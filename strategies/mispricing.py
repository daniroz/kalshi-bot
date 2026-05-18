"""
Mispricing detection — focused on the one rule that's actually +EV.

Rule (kept): YES_ask + NO_ask < $1 minus fees. Pure internal Kalshi arb.
  Buy YES + buy NO, guaranteed $1 payout regardless of resolution.
  Required edge AFTER both legs of fees. At 50¢ mid that's ~3.5¢ each side
  = 7¢ round trip of fees, so min_edge must be ≥6¢ for a viable trade.

Rule (REMOVED 2026-05-18): "near-expiry stale price" — buying at ≥95¢ for
  3¢ of upside is negative EV after the 5% tail risk of opposite resolution.
  EV math: 0.95*3 + 0.05*(-97) = -2¢ per contract. The strategy was
  systematically losing money on this rule.
"""

import time
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


COOLDOWN_S = 600  # 10 min between re-entries per ticker


class MispricingStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager, min_edge: float = 0.06):
        # min_edge raised from 0.03 → 0.06 because internal arb pays Kalshi fees
        # on BOTH legs. At 50¢ that's 1.75¢ × 2 = 3.5¢ in fees just for entry,
        # so we need at least 6¢ headline edge to net ~2-3¢ realised profit.
        self.kalshi = kalshi
        self.risk = risk
        self.min_edge = min_edge
        self._last_entry: dict[str, float] = {}

    def scan(self) -> list[dict]:
        markets = get_liquid_markets(self.kalshi, min_volume=0)

        opps = []
        now = time.time()

        now = time.time()
        for m in markets:
            ticker = m["ticker"]
            if now - self._last_entry.get(ticker, 0) < COOLDOWN_S:
                continue

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

        opps.sort(key=lambda x: x["edge"], reverse=True)
        return opps

    def execute(self, opp: dict) -> bool:
        if opp["type"] == "internal_arb":
            return self._execute_internal_arb(opp)
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

        yes_ok = no_ok = False
        try:
            self.kalshi.place_order(
                ticker=ticker, side="yes", action="buy",
                count=contracts, order_type="limit", yes_price=yes_c,
            )
            yes_ok = True
        except Exception as e:
            log.error(f"[mis] yes order failed {ticker}: {e}")

        try:
            self.kalshi.place_order(
                ticker=ticker, side="no", action="buy",
                count=contracts, order_type="limit", no_price=no_c,
            )
            no_ok = True
        except Exception as e:
            log.error(f"[mis] no order failed {ticker}: {e}")

        if yes_ok and no_ok:
            self.risk.record_open(ticker, contracts)
            self._last_entry[ticker] = time.time()
            log.info(f"[mis] Internal arb {ticker}  yes={yes_c}¢ no={no_c}¢ x{contracts}  edge={edge*100:.1f}¢")
            return True
        elif yes_ok or no_ok:
            # One leg placed — we have a naked directional position. Cancel it immediately.
            filled_side = "yes" if yes_ok else "no"
            log.error(f"[mis] Partial arb fill {ticker} ({filled_side} only) — will attempt cancel")
            try:
                resting = self.risk.kalshi.get_orders(status="resting").get("orders", []) \
                    if hasattr(self.risk, "kalshi") else \
                    self.kalshi.get_orders(status="resting").get("orders", [])
                for o in resting:
                    if o.get("ticker") == ticker:
                        self.kalshi.cancel_order(o["order_id"])
            except Exception as e:
                log.error(f"[mis] Cancel partial arb failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            return False
        else:
            self.risk.undo_reservation(ticker, contracts)
            return False

