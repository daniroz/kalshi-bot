"""Risk management — position sizing, daily loss limits, exposure caps."""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional
from utils.state import save_state, load_state


@dataclass
class RiskConfig:
    starting_balance: float = 500.0
    max_position_pct: float = 0.05      # 5% of balance on initial entry
    max_position_scale_pct: float = 0.12  # total exposure per ticker never exceeds this
    max_daily_loss_pct: float = 1.0
    max_open_positions: int = 999
    min_edge: float = 0.01


_TIER_NAMES = ["healthy", "mild", "moderate", "tight", "emergency"]
# Per-tier multipliers applied to min_edge and max-position caps.
# Tier 4 (emergency) is the only hard block — cash below 10% can't even cover settlement.
_TIER_EDGE_MULT = [1.0, 1.5, 2.5, 4.0, 999.0]
_TIER_SIZE_MULT = [1.0, 0.7, 0.4, 0.2, 0.0]


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.balance: float = cfg.starting_balance
        self.day_start_balance: float = cfg.starting_balance
        self._day_start_ts: float = time.time()
        self.halted: bool = False
        self.halt_reason: str = ""
        # Graduated cash-pressure tier set by Coach each cycle.
        # 0 healthy (≥35% cash), 1 mild (25-35%), 2 moderate (15-25%),
        # 3 tight (10-15%), 4 emergency (<10%, hard block).
        self.cash_tier: int = 0
        self.cash_pct: float = 1.0
        self._lock = threading.Lock()
        # Load persisted positions from last run
        state = load_state()
        self.open_positions: dict[str, int] = {
            k: v for k, v in state.get("open_positions", {}).items() if v > 0
        }
        if self.open_positions:
            from utils.logger import log
            log.info(f"[risk] Restored {len(self.open_positions)} open positions from state")

    # ── Balance updates ───────────────────────────────────────────────────────

    def update_balance(self, new_balance: float):
        self._reset_day_if_needed()
        self.balance = new_balance
        if self._daily_loss() >= self.cfg.max_daily_loss_pct:
            self._halt(f"Daily loss limit hit ({self._daily_loss()*100:.1f}%)")

    def _daily_loss(self) -> float:
        if self.day_start_balance == 0:
            return 0.0
        return max(0.0, (self.day_start_balance - self.balance) / self.day_start_balance)

    def _reset_day_if_needed(self):
        now = time.time()
        # Reset at UTC midnight (86400s per day)
        if now - self._day_start_ts >= 86400:
            self.day_start_balance = self.balance
            self._day_start_ts = now
            self.halted = False
            self.halt_reason = ""

    def _halt(self, reason: str):
        self.halted = True
        self.halt_reason = reason

    # ── Graduated cash tier ──────────────────────────────────────────────────

    def set_cash_tier(self, cash_pct: float) -> int:
        """Coach calls this each cycle with current cash / total_equity ratio.
        Returns the new tier (0-4)."""
        if   cash_pct >= 0.35: tier = 0
        elif cash_pct >= 0.25: tier = 1
        elif cash_pct >= 0.15: tier = 2
        elif cash_pct >= 0.10: tier = 3
        else:                  tier = 4
        self.cash_tier = tier
        self.cash_pct  = cash_pct
        return tier

    def tier_name(self) -> str:
        return _TIER_NAMES[self.cash_tier]

    def effective_min_edge(self) -> float:
        return self.cfg.min_edge * _TIER_EDGE_MULT[self.cash_tier]

    def effective_max_position_pct(self) -> float:
        return self.cfg.max_position_pct * _TIER_SIZE_MULT[self.cash_tier]

    def effective_max_position_scale_pct(self) -> float:
        return self.cfg.max_position_scale_pct * _TIER_SIZE_MULT[self.cash_tier]

    # ── Same-game conflict detection ─────────────────────────────────────────

    @staticmethod
    def _game_key(ticker: str) -> str:
        """Strip trailing team suffix to get the matchup key.
        e.g. KXNHLGAME-26MAY22VGKCOL-VGK → KXNHLGAME-26MAY22VGKCOL

        Threshold suffixes (T90, T88.5) are NOT stripped — those aren't 'sides'
        of the same game, they're independent threshold markets and we want to
        bet multiple of them per city/date (Paruch 3-bin strategy)."""
        parts = ticker.rsplit("-", 1)
        if len(parts) != 2 or len(parts[1]) > 4:
            return ticker
        suffix = parts[1]
        # Threshold markets like T90 / T88.5 are independent — don't merge them.
        if suffix.startswith("T") and len(suffix) >= 2 and suffix[1].isdigit():
            return ticker
        return parts[0]

    def same_game_conflict(self, ticker: str) -> Optional[str]:
        """Return the conflicting ticker if any held position is on the other side
        of the same game. Returns None if clear."""
        game = self._game_key(ticker)
        if game == ticker:
            return None   # no recognisable team suffix — can't detect conflict
        for held in self.open_positions:
            if held == ticker:
                continue
            if self._game_key(held) == game:
                return held
        return None

    # ── Trade approval ────────────────────────────────────────────────────────

    def approve_trade(
        self,
        ticker: str,
        yes_price_cents: int,  # 1-99
        contracts: int,
        edge_dollars: float,
    ) -> tuple[bool, str]:
        """Return (approved, reason). Thread-safe — serializes all approval checks."""
        with self._lock:
            if self.halted:
                return False, f"Bot halted: {self.halt_reason}"

            # Tier 4 (cash <10%) is the only hard block — emergency
            if self.cash_tier >= 4:
                return False, f"Cash emergency ({self.cash_pct*100:.1f}%) — settle out before adding"

            # Block positions on both sides of the same game
            conflict = self.same_game_conflict(ticker)
            if conflict:
                return False, f"Same-game conflict: already holding {conflict}"

            cost = (yes_price_cents / 100) * contracts

            # Minimum trade size — Kalshi fees + slippage make dust trades net-negative.
            # $5 floor means we only place trades that could actually move the P&L needle.
            if cost < 5.0:
                return False, f"Trade too small: ${cost:.2f} < $5 minimum"

            existing_contracts = self.open_positions.get(ticker, 0)
            existing_cost = (yes_price_cents / 100) * existing_contracts
            total_cost = existing_cost + cost

            # Tier-adjusted caps: tighter when cash is scarce, normal at tier 0
            eff_init_pct  = self.effective_max_position_pct()
            eff_scale_pct = self.effective_max_position_scale_pct()
            eff_min_edge  = self.effective_min_edge()

            # Hard cap: total exposure on any ticker never exceeds scale_pct
            if total_cost > self.balance * eff_scale_pct:
                return False, (
                    f"Would exceed tier-{self.cash_tier} ({self.tier_name()}) "
                    f"{eff_scale_pct*100:.1f}% max exposure on {ticker} (${total_cost:.2f})"
                )

            # Initial entry size cap
            if existing_contracts == 0 and cost > self.balance * eff_init_pct:
                return False, (
                    f"Initial position too large for tier-{self.cash_tier} "
                    f"({self.tier_name()}): ${cost:.2f} > {eff_init_pct*100:.1f}%"
                )

            # Per-contract edge — strategies pass total (edge × contracts), so we
            # divide back to per-contract for an apples-to-apples comparison with min_edge.
            # This is what we *actually* mean by "min edge" — every single contract
            # of this trade has to be at least min_edge profitable in expectation.
            per_contract_edge = edge_dollars / contracts if contracts > 0 else 0
            if per_contract_edge < eff_min_edge:
                return False, (
                    f"Per-contract edge too small for tier-{self.cash_tier} "
                    f"({self.tier_name()}): {per_contract_edge*100:.2f}¢ "
                    f"< {eff_min_edge*100:.2f}¢"
                )

            # Reserve the exposure immediately so concurrent threads see it
            self.open_positions[ticker] = existing_contracts + contracts
            return True, "ok"

    def kelly_contracts(
        self,
        yes_price_cents: int,
        true_prob: float,
        max_contracts: int = 50,
    ) -> int:
        """
        Kelly criterion: f* = (bp - q) / b
        where b = odds (payout/cost), p = true_prob, q = 1 - p
        Returns conservative half-Kelly contract count.
        """
        price = yes_price_cents / 100
        if price <= 0 or price >= 1:
            return 0
        b = (1 - price) / price   # net odds on a $1 bet
        p = true_prob
        q = 1 - p
        f = (b * p - q) / b
        if f <= 0:
            return 0
        half_kelly = f * 0.5
        max_cost_per_contract = price
        max_spend = min(self.balance * half_kelly, self.balance * self.cfg.max_position_pct)
        contracts = int(max_spend / max_cost_per_contract)
        return max(0, min(contracts, max_contracts))

    def record_open(self, ticker: str, contracts: int):
        # approve_trade already reserved these contracts in open_positions.
        # This call just persists the state to disk.
        with self._lock:
            save_state(self.open_positions)

    def undo_reservation(self, ticker: str, contracts: int):
        """Call when order placement fails after approve_trade returned True."""
        with self._lock:
            existing = self.open_positions.get(ticker, 0)
            remaining = existing - contracts
            if remaining <= 0:
                self.open_positions.pop(ticker, None)
            else:
                self.open_positions[ticker] = remaining
            save_state(self.open_positions)

    def record_close(self, ticker: str, contracts: int):
        with self._lock:
            existing = self.open_positions.get(ticker, 0)
            remaining = existing - contracts
            if remaining <= 0:
                self.open_positions.pop(ticker, None)
            else:
                self.open_positions[ticker] = remaining
            save_state(self.open_positions)

    def sync_positions(self, kalshi) -> int:
        """
        Replace internal open_positions with ground-truth from Kalshi position_fp.
        Call once per cycle. Returns number of positions synced.
        This prevents the market maker from accumulating positions beyond risk limits
        by re-quoting every cycle while the internal counter resets to 0.
        """
        try:
            r = kalshi._get("/portfolio/positions", {"limit": 200})
            market_pos = r.get("market_positions", [])
            real: dict[str, int] = {}
            for p in market_pos:
                fp = float(p.get("position_fp") or 0)
                if fp == 0:
                    continue
                ticker = p.get("ticker") or p.get("market_ticker", "")
                if not ticker:
                    continue
                # Use absolute contract count regardless of YES/NO direction
                real[ticker] = int(abs(fp)) or 1
            with self._lock:
                self.open_positions = real
                save_state(self.open_positions)
            return len(real)
        except Exception as e:
            from utils.logger import log
            log.warning(f"[risk] Position sync failed: {e}")
            return -1

    def status(self) -> dict:
        self._reset_day_if_needed()
        return {
            "balance": round(self.balance, 2),
            "day_start_balance": round(self.day_start_balance, 2),
            "daily_pnl": round(self.balance - self.day_start_balance, 2),
            "daily_loss_pct": round(self._daily_loss() * 100, 2),
            "open_positions": len(self.open_positions),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }
