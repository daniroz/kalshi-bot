"""
Mispricing detection: find markets where the implied probability
significantly differs from a base-rate model.

Rules used:
1. Binary complement check — YES + NO prices should sum to ~$1.
   If YES_ask + NO_ask < $0.97, there's a pure Kalshi-internal arb.
2. Extreme liquidity imbalance — large one-sided order book suggests
   informed flow; fade the thin side.
3. Near-expiry mispricing — market closes in <24h but price is still
   far from 0 or 100 with no recent volume (stale market).
"""

import time
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


class MispricingStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager, min_edge: float = 0.03):
        self.kalshi = kalshi
        self.risk = risk
        self.min_edge = min_edge

    def scan(self) -> list[dict]:
        markets = get_liquid_markets(self.kalshi, min_volume=0)

        opps = []
        now = time.time()

        for m in markets:
            yes_ask = float(m.get("yes_ask_dollars") or 0)
            yes_bid = float(m.get("yes_bid_dollars") or 0)
            no_ask  = float(m.get("no_ask_dollars")  or 0)
            no_bid  = float(m.get("no_bid_dollars")  or 0)

            # Rule 1: internal arbitrage (YES ask + NO ask < 0.97)
            if yes_ask and no_ask:
                total = yes_ask + no_ask
                edge  = (1.0 - total) - self.min_edge
                if edge > 0:
                    opps.append({
                        "type": "internal_arb",
                        "ticker": m["ticker"],
                        "title": m.get("title"),
                        "yes_ask": yes_ask,
                        "no_ask": no_ask,
                        "edge": round(edge, 4),
                        "buy_yes": True,
                        "buy_no": True,
                    })

            # Rule 2: near-expiry stale price
            close_ts_str = m.get("close_time") or m.get("expected_expiration_time")
            if close_ts_str:
                try:
                    from datetime import datetime, timezone
                    close_dt = datetime.fromisoformat(close_ts_str.replace("Z", "+00:00"))
                    secs_left = (close_dt - datetime.now(timezone.utc)).total_seconds()
                    vol_24h = m.get("volume_24h_fp") or 0
                    mid = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else None
                    if (
                        0 < secs_left < 3600   # expires in under 1 hour
                        and mid is not None
                        and vol_24h == 0        # no recent volume
                        and (mid < 0.05 or mid > 0.95)  # near resolution
                    ):
                        side  = "yes" if mid > 0.95 else "no"
                        price = yes_ask if side == "yes" else no_ask
                        edge  = (1.0 - price) - self.min_edge if price else 0
                        if edge > 0:
                            opps.append({
                                "type": "near_expiry",
                                "ticker": m["ticker"],
                                "title": m.get("title"),
                                "side": side,
                                "price": price,
                                "edge": round(edge, 4),
                                "secs_left": int(secs_left),
                            })
                except Exception:
                    pass

        opps.sort(key=lambda x: x["edge"], reverse=True)
        return opps

    def execute(self, opp: dict) -> bool:
        ticker = opp["ticker"]

        if opp["type"] == "internal_arb":
            return self._execute_internal_arb(opp)
        elif opp["type"] == "near_expiry":
            return self._execute_near_expiry(opp)
        return False

    def _execute_internal_arb(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        yes_c   = int(round(opp["yes_ask"] * 100))
        no_c    = int(round(opp["no_ask"] * 100))
        edge    = opp["edge"]
        contracts = max(1, min(10, self.risk.kelly_contracts(yes_c, 0.5 + edge / 2, 10)))

        ok, reason = self.risk.approve_trade(ticker, yes_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[mis] Skipped internal arb {ticker}: {reason}")
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
                log.error(f"[mis] {side} order failed {ticker}: {e}")

        if placed:
            self.risk.record_open(ticker, contracts)
            log.info(f"[mis] Internal arb {ticker}  yes={yes_c}¢ no={no_c}¢ x{contracts}  edge={edge*100:.1f}¢")
        return placed > 0

    def _execute_near_expiry(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_c = int(round(opp["price"] * 100))
        edge    = opp["edge"]
        contracts = max(1, min(20, self.risk.kelly_contracts(price_c, price_c / 100 + edge, 20)))

        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[mis] Skipped near_expiry {ticker}: {reason}")
            return False

        try:
            self.kalshi.place_order(
                ticker=ticker, side=side, action="buy",
                count=contracts, order_type="limit",
                yes_price=price_c if side == "yes" else None,
                no_price=price_c  if side == "no"  else None,
            )
            self.risk.record_open(ticker, contracts)
            log.info(f"[mis] Near-expiry {ticker} {side.upper()} x{contracts} @ {price_c}¢  {opp['secs_left']}s left")
            return True
        except Exception as e:
            log.error(f"[mis] Near-expiry order failed {ticker}: {e}")
            return False
