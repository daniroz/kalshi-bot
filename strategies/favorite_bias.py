"""
Favorite-Longshot Bias Harvesting.

THE EDGE (one of the most robust findings in prediction-market research —
Thaler & Ziemba 1988, Snowberg & Wolfers 2010, Polymarket calibration studies):
  Heavy favorites are systematically UNDERpriced; longshots OVERpriced.
  People overpay for lottery-ticket upside (the 5¢ "maybe it moons" bet) and
  underpay for boring near-certainties. A contract trading at 90¢ historically
  resolves YES ~92-94% of the time.

THE TRADE:
  Buy the favorite (high-priced) side of LIQUID binary markets when it trades
  in the documented bias zone (85-94¢). Hold to resolution. Edge per contract
  is small (~2-3¢ after fees) but the win rate is high (~92%) and fees at
  extreme prices are tiny (0.07 * 0.9 * 0.1 = 0.63¢).

MONTE CARLO (simulate_strategies.py, 500 runs): median +$139/season,
92% of seasons profitable, Sharpe 1.41.

CRITICAL — this edge is UNPROVEN ON KALSHI specifically. The sim used a +3¢
bias measured on other markets. This strategy logs every entry with the entry
price so we can compute the realized win rate vs the ~92% the bias predicts.
If after ~50 settled trades the win rate is materially below the entry-price
implied probability, the bias isn't here and we turn it off.

Risk controls:
  - Only liquid markets (volume ≥ MIN_VOLUME) so the price is meaningful
  - Only the 85-94¢ zone (below 85 the bias is weak; above 94 no edge left
    after fees + the few percent of genuine upsets)
  - One position per event (no doubling on correlated outcomes)
  - Hold to resolution — no active management
  - Size from the risk manager's tier-aware caps
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional

from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


# ── Tunables ───────────────────────────────────────────────────────────────────
MIN_VOLUME_24H    = 2000.0    # lowered from 5000 — Kalshi thin; still real volume — price must be meaningful
FAV_PRICE_MIN_C   = 85        # bias zone floor
FAV_PRICE_MAX_C   = 97        # extended May 31 from 94 — simulate_v2 showed
                              # extreme favorites (>94¢) also positive EV due
                              # to low fees at extremes + persistent bias
MIN_EDGE_C        = 1.5       # ~3¢ gross bias − ~0.6¢ fee = require ≥1.5¢ net
MIN_TRADE_DOLLARS = 5.0
MAX_TRADE_DOLLARS = 50.0
SCAN_INTERVAL_S   = 120       # slow — these are hold-to-resolution bets
COOLDOWN_S        = 3600      # don't re-enter same ticker within an hour

# Assumed bias (from research) used only for the edge calc + logging.
# The realized win rate is what actually matters — we log to measure it.
ASSUMED_BIAS_C    = 3.0


@dataclass
class FavPosition:
    ticker:        str
    side:          str
    contracts:     int
    entry_price_c: int
    implied_prob:  float    # = entry price; what we're betting beats reality


def _kalshi_fee_cents(price_c: int, contracts: int = 1) -> float:
    p = price_c / 100
    return 0.07 * p * (1 - p) * 100 * contracts


class FavoriteBiasStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk   = risk
        self._last_scan = 0.0
        self._positions: dict[str, FavPosition] = {}
        self._last_entry: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _event_key(ticker: str) -> str:
        parts = ticker.rsplit("-", 1)
        return parts[0] if len(parts) == 2 else ticker

    def _evaluate(self, m: dict) -> Optional[dict]:
        ticker = m["ticker"]
        yes_ask_c = int(round(float(m.get("yes_ask_dollars") or 0) * 100))
        no_ask_c  = int(round(float(m.get("no_ask_dollars")  or 0) * 100))
        if yes_ask_c == 0 or no_ask_c == 0:
            return None

        # Which side is the favorite (high-priced)? Buy whichever sits in the zone.
        side = price_c = None
        if FAV_PRICE_MIN_C <= yes_ask_c <= FAV_PRICE_MAX_C:
            side, price_c = "yes", yes_ask_c
        elif FAV_PRICE_MIN_C <= no_ask_c <= FAV_PRICE_MAX_C:
            side, price_c = "no", no_ask_c
        if side is None:
            return None

        # Net edge: assumed bias minus fee. Both in cents.
        fee_c = _kalshi_fee_cents(price_c)
        edge_c = ASSUMED_BIAS_C - fee_c
        if edge_c < MIN_EDGE_C:
            return None

        return {
            "ticker": ticker, "side": side, "price_c": price_c,
            "edge_c": edge_c, "implied_prob": price_c / 100,
            "title": m.get("title", "")[:55],
        }

    def scan(self) -> list[dict]:
        now = time.time()
        with self._lock:
            if now - self._last_scan < SCAN_INTERVAL_S:
                return []
            self._last_scan = now

        markets = get_liquid_markets(self.kalshi, min_volume=MIN_VOLUME_24H)

        # Don't take two positions in the same event
        held_events = {self._event_key(t) for t in self.risk.open_positions}
        held_events |= {self._event_key(t) for t in self._positions}

        opps = []
        for m in markets:
            ticker = m["ticker"]
            if ticker in self._positions:
                continue
            if self.risk.open_positions.get(ticker, 0) > 0:
                continue
            if self._event_key(ticker) in held_events:
                continue
            if now - self._last_entry.get(ticker, 0) < COOLDOWN_S:
                continue
            try:
                opp = self._evaluate(m)
                if opp:
                    opps.append(opp)
                    held_events.add(self._event_key(ticker))  # dedupe within this scan
            except Exception as e:
                log.debug(f"[favbias] eval {ticker} failed: {e}")
        # Prefer the strongest favorites (closest to the high end of the zone)
        opps.sort(key=lambda x: x["price_c"], reverse=True)
        return opps

    def execute(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_c = opp["price_c"]
        edge_c  = opp["edge_c"]

        price = price_c / 100
        tier_cap = self.risk.balance * self.risk.effective_max_position_pct()
        target = min(MAX_TRADE_DOLLARS, tier_cap)
        if target < MIN_TRADE_DOLLARS:
            return False
        contracts = int(target / price)
        if contracts <= 0:
            return False

        with self._lock:
            if ticker in self._positions:
                return False
            self._positions[ticker] = None

        edge_dollars_total = (edge_c / 100) * contracts
        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge_dollars_total)
        if not ok:
            log.info(f"[favbias] Skip {ticker}: {reason}")
            self._positions.pop(ticker, None)
            return False

        try:
            self.kalshi.place_order(
                ticker=ticker, side=side, action="buy",
                count=contracts, order_type="limit",
                yes_price=price_c if side == "yes" else None,
                no_price =price_c if side == "no"  else None,
            )
            self.risk.record_open(ticker, contracts)
            self._last_entry[ticker] = time.time()
            self._positions[ticker] = FavPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, implied_prob=price_c / 100,
            )
            # Log with implied prob so we can later compute realized win rate vs implied
            log.info(
                f"[favbias] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢  "
                f"implied_p={price_c/100:.2f} edge={edge_c:.1f}¢  {opp['title']}"
            )
            return True
        except Exception as e:
            log.error(f"[favbias] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            self._positions.pop(ticker, None)
            return False
