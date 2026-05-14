"""
Internal arbitrage strategy.

Buys both sides of a Kalshi market when YES_ask + NO_ask < 96 cents,
locking in a guaranteed profit regardless of how the market resolves.

This is the only signal this strategy fires — no near-certain plays,
no directional bets. Pure locked-in arb only.
"""

import time
from dataclasses import dataclass
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


COOLDOWN_S   = 3600   # 1 hour — once we arb a ticker, don't re-enter same day
MIN_ARB_EDGE = 0.04   # only fire if guaranteed profit >= 4 cents per contract


@dataclass
class ArbPosition:
    ticker: str
    contracts: int
    entered_ts: float


class OrderbookStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk = risk
        self._last_entry: dict[str, float] = {}
        self._positions:  dict[str, ArbPosition] = {}

    def _check_exits(self):
        """Close out positions when the market has finalized."""
        for ticker, pos in list(self._positions.items()):
            try:
                m = self.kalshi.get_market(ticker).get("market", {})
                status = m.get("status", "")
                if status in ("finalized", "settled"):
                    self.risk.record_close(ticker, pos.contracts)
                    log.info(f"[ob] ARB settled {ticker} x{pos.contracts}")
                    self._positions.pop(ticker, None)
                # Also clean up positions older than 7 days (market must have settled)
                elif time.time() - pos.entered_ts > 7 * 86400:
                    self.risk.record_close(ticker, pos.contracts)
                    self._positions.pop(ticker, None)
            except Exception as e:
                log.debug(f"[ob] exit check {ticker}: {e}")

    def scan(self) -> list[dict]:
        self._check_exits()
        markets = get_liquid_markets(self.kalshi, min_volume=50)
        signals = []

        for m in markets:
            ticker = m["ticker"]
            if ticker in self._positions:
                continue
            if time.time() - self._last_entry.get(ticker, 0) < COOLDOWN_S:
                continue

            yes_ask = float(m.get("yes_ask_dollars") or 0)
            no_ask  = float(m.get("no_ask_dollars")  or 0)
            if not yes_ask or not no_ask:
                continue

            yes_ask_c = int(yes_ask * 100)
            no_ask_c  = int(no_ask  * 100)
            edge = round(1.0 - yes_ask - no_ask, 3)

            if edge >= MIN_ARB_EDGE:
                signals.append({
                    "ticker":    ticker,
                    "title":     m.get("title", "")[:60],
                    "price_c":   yes_ask_c,
                    "no_price_c": no_ask_c,
                    "edge":      edge,
                })

        signals.sort(key=lambda x: x["edge"], reverse=True)
        return signals

    def execute(self, signal: dict) -> bool:
        ticker    = signal["ticker"]
        yes_c     = signal["price_c"]
        no_c      = signal["no_price_c"]
        edge      = signal["edge"]
        contracts = max(1, min(20, self.risk.kelly_contracts(yes_c, 0.5 + edge / 2, 20)))

        ok, reason = self.risk.approve_trade(ticker, yes_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[ob] Skipped {ticker}: {reason}")
            return False

        placed = 0
        for side, price_c in (("yes", yes_c), ("no", no_c)):
            try:
                self.kalshi.place_order(
                    ticker=ticker, side=side, action="buy",
                    count=contracts, order_type="limit",
                    yes_price=price_c if side == "yes" else None,
                    no_price=price_c  if side == "no"  else None,
                )
                placed += 1
            except Exception as e:
                log.error(f"[ob] {side} leg failed {ticker}: {e}")

        if placed:
            self.risk.record_open(ticker, contracts)
            self._last_entry[ticker] = time.time()
            self._positions[ticker] = ArbPosition(
                ticker=ticker, contracts=contracts, entered_ts=time.time()
            )
            log.info(f"[ob] ARB {ticker} both x{contracts}  yes={yes_c}c no={no_c}c  edge={edge*100:.1f}c")
        else:
            self.risk.undo_reservation(ticker, contracts)
        return placed > 0
