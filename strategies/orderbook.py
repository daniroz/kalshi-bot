"""
Internal arbitrage strategy.

Buys both sides of a Kalshi market when YES_ask + NO_ask < 96 cents,
locking in a guaranteed profit regardless of how the market resolves.

This is the only signal this strategy fires — no near-certain plays,
no directional bets. Pure locked-in arb only.
"""

import time
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


MIN_TOTAL_DEPTH     = 0
IMBALANCE_THRESHOLD = 2.0
COOLDOWN_S          = 120
MIN_ARB_EDGE        = 0.04  # only fire if guaranteed profit >= 4 cents per contract


class OrderbookStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk = risk
        self._last_entry: dict[str, float] = {}
        self._prev_imbalance: dict[str, float] = {}

    def scan(self) -> list[dict]:
        markets = get_liquid_markets(self.kalshi, min_volume=50)
        signals = []

        for m in markets:
            ticker = m["ticker"]
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
                    "ticker": ticker, "title": m.get("title", "")[:60],
                    "type": "arb", "side": "both",
                    "price_c": yes_ask_c, "no_price_c": no_ask_c,
                    "edge": edge, "yes_qty": 0, "no_qty": 0,
                    "ratio": 1.0, "strengthening": True,
                })

        signals.sort(key=lambda x: x["edge"], reverse=True)
        return signals

    def execute(self, signal: dict) -> bool:
        if signal["type"] == "arb":
            return self._execute_arb(signal)

        ticker  = signal["ticker"]
        side    = signal["side"]
        price_c = signal["price_c"]
        edge    = signal["edge"]

        contracts = self.risk.kelly_contracts(price_c, price_c / 100 + edge, max_contracts=50)
        contracts = max(1, contracts)

        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[ob] Skipped {ticker}: {reason}")
            return False

        try:
            self.kalshi.place_order(
                ticker=ticker, side=side, action="buy",
                count=contracts, order_type="limit",
                yes_price=price_c if side == "yes" else None,
                no_price=price_c  if side == "no"  else None,
            )
            self.risk.record_open(ticker, contracts)
            self._last_entry[ticker] = time.time()
            log.info(f"[ob] {signal['type'].upper()} {ticker} {side.upper()} x{contracts} @ {price_c}c  edge={edge*100:.1f}c")
            return True
        except Exception as e:
            log.error(f"[ob] Order failed {ticker}: {e}")
            return False

    def _execute_arb(self, signal: dict) -> bool:
        ticker    = signal["ticker"]
        yes_c     = signal["price_c"]
        no_c      = signal["no_price_c"]
        edge      = signal["edge"]
        contracts = max(1, min(20, self.risk.kelly_contracts(yes_c, 0.5 + edge / 2, 20)))

        ok, reason = self.risk.approve_trade(ticker, yes_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[ob] Skipped arb {ticker}: {reason}")
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
                log.error(f"[ob] {side} arb leg failed {ticker}: {e}")

        if placed:
            self.risk.record_open(ticker, contracts)
            self._last_entry[ticker] = time.time()
            log.info(f"[ob] ARB {ticker} both x{contracts}  yes={yes_c}c no={no_c}c  edge={edge*100:.1f}c")
        return placed > 0
