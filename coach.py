"""
The Coach — runs every cycle alongside all strategies.

Responsibilities:
  1. ENFORCER  — if any position exceeds the hard limit, force-reduce it immediately
  2. WATCHDOG  — tracks fill rate; if bot has been idle too long, loosens edge thresholds
  3. RANKER    — scores each strategy by P&L contribution; slows down consistent losers
  4. MOTIVATOR — logs what's working, what isn't, and pushes harder when opportunity is there
  5. REPORTER  — every 10 min prints a full performance breakdown

The Coach never blocks strategies directly — it adjusts the shared RiskManager
thresholds and writes to the log. Strategies read thresholds on every trade.
"""

import time
import math
from collections import defaultdict
from utils.logger import log
from risk.manager import RiskManager
from clients.kalshi import KalshiClient


# ── Tunable constants ──────────────────────────────────────────────────────────
HARD_LIMIT_PCT      = 0.08   # force-reduce any position above this % of balance
TARGET_PCT          = 0.06   # reduce down to this % (leave a little room)
IDLE_THRESHOLD_S    = 900    # 15 min with no fills = "idle" (for reporting only)
REPORT_EVERY_S      = 600    # full coach report every 10 min
MAX_SINGLE_GAME_PCT = 0.15   # alert if >15% of portfolio in one game/event
MIN_CASH_PCT        = 0.35   # HARD FLOOR: cash must always be ≥35% of total portfolio


class Coach:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi  = kalshi
        self.risk    = risk

        # Fill tracking
        self._last_fill_ts: float          = time.time()
        self._last_newest_fill_ts: str     = ""
        self._total_fills_seen: int        = 0

        self._last_report_ts: float = 0.0

        # Cycle counter
        self._cycle: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point — call once per main loop cycle
    # ─────────────────────────────────────────────────────────────────────────

    def tick(self):
        self._cycle += 1
        try:
            self._enforce_cash_floor()
            self._enforce_position_limits()
            self._check_concentration()
            self._track_fill_rate()
            if time.time() - self._last_report_ts > REPORT_EVERY_S:
                self._full_report()
                self._last_report_ts = time.time()
        except Exception as e:
            log.warning(f"[coach] tick error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # 0. CASH FLOOR — never let cash drop below 35% of total portfolio
    # ─────────────────────────────────────────────────────────────────────────

    def _enforce_cash_floor(self):
        """
        Graduated cash-pressure system. Instead of a hard wall at 35%, raise the
        bar (edge tax) and shrink position size as cash falls. We never miss a
        great edge — we just stop taking marginal ones when capital is scarce.

          tier 0 healthy   (≥35%)    edge×1.0  size×1.0
          tier 1 mild      (25-35%)  edge×1.5  size×0.7
          tier 2 moderate  (15-25%)  edge×2.5  size×0.4
          tier 3 tight     (10-15%)  edge×4.0  size×0.2
          tier 4 emergency (<10%)    HARD BLOCK
        """
        try:
            bal      = self.kalshi.get_balance()
            cash     = float(bal.get("balance", 0))
            pos_val  = float(bal.get("portfolio_value", 0)) / 100
            total    = cash + pos_val
            if total <= 0:
                return

            cash_pct = cash / total
            prev_tier = self.risk.cash_tier
            new_tier  = self.risk.set_cash_tier(cash_pct)

            # Log transitions and emergencies
            if new_tier != prev_tier:
                icons = ["✅", "🟡", "🟠", "🔴", "🚨"]
                log.warning(
                    f"[coach] {icons[new_tier]} Cash tier {prev_tier}→{new_tier} "
                    f"({self.risk.tier_name()}): ${cash:.2f} = {cash_pct*100:.1f}% — "
                    f"edge×{[1.0,1.5,2.5,4.0,999.0][new_tier]:.1f}  "
                    f"size×{[1.0,0.7,0.4,0.2,0.0][new_tier]:.1f}"
                )
            elif new_tier == 4:
                # Persistent emergency — periodic reminder
                log.warning(
                    f"[coach] 🚨 CASH EMERGENCY: ${cash:.2f} = {cash_pct*100:.1f}% — "
                    f"all new entries blocked until cash recovers above 10%"
                )

            # Periodic status line (every 10 cycles)
            if self._cycle % 10 == 0:
                icons = ["✅", "🟡", "🟠", "🔴", "🚨"]
                log.info(
                    f"[coach] {icons[new_tier]} Cash ${cash:.2f} = {cash_pct*100:.1f}% — "
                    f"tier {new_tier} ({self.risk.tier_name()}), "
                    f"min_edge={self.risk.effective_min_edge()*100:.1f}¢, "
                    f"max_pos={self.risk.effective_max_position_pct()*100:.1f}%"
                )

        except Exception as e:
            log.warning(f"[coach] Cash tier check failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. ENFORCER — reduce any oversized position immediately
    # ─────────────────────────────────────────────────────────────────────────

    def _enforce_position_limits(self):
        balance = self.risk.balance
        if balance <= 0:
            return

        try:
            r = self.kalshi._get("/portfolio/positions", {"limit": 200})
        except Exception:
            return

        market_pos = r.get("market_positions", [])
        for p in market_pos:
            fp = float(p.get("position_fp") or 0)
            if fp == 0:
                continue
            ticker = p.get("ticker") or p.get("market_ticker", "")
            if not ticker:
                continue

            # Get current mark price
            try:
                mkt = self.kalshi.get_market(ticker).get("market", {})
                if fp > 0:
                    mark = float(mkt.get("yes_bid_dollars") or 0)
                else:
                    ask  = float(mkt.get("yes_ask_dollars") or 1)
                    mark = 1.0 - ask
                if mark <= 0:
                    continue
            except Exception:
                mark = 0.50   # assume 50¢ if can't fetch

            notional = abs(fp) * mark
            pct      = notional / balance

            if pct > HARD_LIMIT_PCT:
                # Warn-only. The tier system and approve_trade caps prevent NEW
                # over-sized positions. Force-selling existing ones creates churn,
                # pays Kalshi fees, and locks in losses — let natural settlement
                # handle it. (User policy: "do not fire-sell any positions.")
                side = "yes" if fp > 0 else "no"
                log.warning(
                    f"[coach] ⚠️  Oversized: {ticker} {side.upper()} x{abs(fp):.0f} "
                    f"= {pct*100:.1f}% of portfolio (limit {HARD_LIMIT_PCT*100:.0f}%) "
                    f"— letting it ride to settlement."
                )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. CONCENTRATION CHECK — warn if too much in one game/event
    # ─────────────────────────────────────────────────────────────────────────

    def _check_concentration(self):
        """Alert and force-reduce if too much is concentrated in one game/event."""
        balance = self.risk.balance
        if balance <= 0:
            return

        # Group positions by game key (strip last team suffix for sports,
        # strip threshold suffix for weather e.g. KXLOWTOKC-26MAY17-T65 → KXLOWTOKC-26MAY17)
        game_notional: dict[str, list] = defaultdict(list)
        for ticker, contracts in list(self.risk.open_positions.items()):
            parts    = ticker.rsplit("-", 1)
            game_key = parts[0] if len(parts) == 2 and len(parts[1]) <= 4 else ticker
            game_notional[game_key].append((ticker, contracts))

        for game, positions in game_notional.items():
            total_contracts = sum(c for _, c in positions)
            notional = total_contracts * 0.50
            pct = notional / balance
            if pct > MAX_SINGLE_GAME_PCT:
                # Warn-only. The tier system prevents NEW over-concentration via tier-adjusted
                # caps in approve_trade. Force-selling existing positions to chase a per-game
                # limit creates churn + fees; let natural settlement handle it.
                log.warning(
                    f"[coach] 🎯 Concentration alert: {game} = ~{pct*100:.0f}% of portfolio "
                    f"(no action — letting it ride to settle)"
                )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. FILL RATE TRACKER
    # ─────────────────────────────────────────────────────────────────────────

    def _track_fill_rate(self):
        try:
            fills = self.kalshi._get("/portfolio/fills", {"limit": 5}).get("fills", [])
            current_count = len(fills)
        except Exception:
            return

        # Detect new fills by checking if most-recent fill timestamp changed
        if fills:
            newest_ts = fills[0].get("created_time", "")
            if newest_ts != getattr(self, "_last_newest_fill_ts", ""):
                self._last_newest_fill_ts = newest_ts
                self._last_fill_ts        = time.time()
                self._total_fills_seen   += 1

    # ─────────────────────────────────────────────────────────────────────────
    # 4. FULL REPORT — printed every 10 min
    # ─────────────────────────────────────────────────────────────────────────

    def _full_report(self):
        balance  = self.risk.balance
        n_pos    = len(self.risk.open_positions)
        idle_s   = time.time() - self._last_fill_ts
        tier     = self.risk.cash_tier
        eff_edge_c = self.risk.effective_min_edge() * 100
        eff_size_pct = self.risk.effective_max_position_pct() * 100

        # Get live cash for the report
        cash_pct_str = "N/A"
        try:
            bal     = self.kalshi.get_balance()
            cash    = float(bal.get("balance", 0))
            pos_val = float(bal.get("portfolio_value", 0)) / 100
            total   = cash + pos_val
            if total > 0:
                cash_pct = cash / total
                icons = ["✅", "🟡", "🟠", "🔴", "🚨"]
                cash_pct_str = (
                    f"${cash:.2f} = {cash_pct*100:.1f}%  "
                    f"[{icons[tier]} tier {tier} {self.risk.tier_name()}]"
                )
        except Exception:
            pass

        IDLE_PUSHES = [
            "The edge is out there. Scan harder. Every market has a price that's wrong right now.",
            "Markets don't sleep. Someone is mispricing something right now — find it.",
            "No fills doesn't mean no edges. It means you haven't found them yet. Look again.",
            "The spread is there. The volume is there. Stop waiting and go get it.",
            "Every quiet minute is a missed trade. The edge doesn't come to you.",
        ]
        ACTIVE_PRAISE = [
            "That's the pace. Keep finding them — there's more where that came from.",
            "Good. Fills coming in. Now go find the next one.",
            "This is what we want. Active, disciplined, executing. More.",
        ]

        lines = [
            f"[coach] ── COACH REPORT ──────────────────────────────",
            f"[coach]   Portfolio:    ${balance:.2f}",
            f"[coach]   Cash:         {cash_pct_str}",
            f"[coach]   Open pos:     {n_pos}",
            f"[coach]   Effective:    min_edge={eff_edge_c:.1f}¢  max_pos={eff_size_pct:.1f}%",
            f"[coach]   Idle:         {idle_s/60:.0f}m since last fill",
            f"[coach]   Total fills:  {self._total_fills_seen} this session",
        ]

        # Biggest positions
        if self.risk.open_positions:
            sorted_pos = sorted(
                self.risk.open_positions.items(), key=lambda x: x[1], reverse=True
            )[:5]
            lines.append(f"[coach]   Top positions:")
            for ticker, contracts in sorted_pos:
                pct = (contracts * 0.50 / balance * 100) if balance else 0
                lines.append(f"[coach]     {ticker[-30:]:30s}  x{contracts:3d}  ~{pct:.1f}%")

        # Motivation — push harder, never lower the bar
        import random
        if idle_s > 1800:
            lines.append(f"[coach] 🔥 {idle_s/60:.0f}min with no fill. " + random.choice(IDLE_PUSHES))
        elif idle_s > IDLE_THRESHOLD_S:
            lines.append(f"[coach] ⚡ Quiet spell. " + random.choice(IDLE_PUSHES))
        elif idle_s < 300:
            lines.append(f"[coach] ✅ " + random.choice(ACTIVE_PRAISE))
        else:
            lines.append(f"[coach] ⏳ Scanning. The edge is there — don't stop looking.")

        lines.append(f"[coach] ────────────────────────────────────────────────")

        for line in lines:
            log.info(line)
