"""
Track pending limit orders and cancel stale ones.

Usage in a strategy:
    self._fills = FillTracker(kalshi)
    ...
    order_id = place_order(...)
    self._fills.track(order_id, ticker, contracts)
    ...
    cancelled = self._fills.check_stale(self.risk)   # call each cycle
"""

import time
from utils.logger import log


STALE_AFTER = 300   # cancel unfilled limit orders after 5 minutes


class FillTracker:
    def __init__(self, kalshi, stale_after: int = STALE_AFTER):
        self._kalshi     = kalshi
        self._stale_after = stale_after
        self._pending: dict[str, dict] = {}   # order_id -> {ticker, contracts, ts}

    def track(self, order_id: str, ticker: str, contracts: int):
        if order_id:
            self._pending[order_id] = {
                "ticker":    ticker,
                "contracts": contracts,
                "ts":        time.time(),
            }

    def check_stale(self, risk=None) -> list[dict]:
        """
        For each pending order older than stale_after:
          - If still resting → cancel it, reverse the position in RiskManager
          - If already filled → leave it alone
        Returns list of cancelled order infos.
        """
        now = time.time()
        cancelled = []

        for oid, info in list(self._pending.items()):
            if now - info["ts"] < self._stale_after:
                continue
            try:
                r = self._kalshi._get(f"/portfolio/orders/{oid}")
                order  = r.get("order", {})
                status = order.get("status", "")
                filled = int(float(order.get("filled_count_fp") or order.get("count_fp") or 0))
                remaining = int(float(order.get("remaining_count") or 0))

                if status == "resting" or remaining > 0:
                    self._kalshi._delete(f"/portfolio/orders/{oid}")
                    if risk and remaining > 0:
                        risk.record_close(info["ticker"], remaining)
                    log.info(
                        f"[fills] Cancelled stale order {oid[:8]} "
                        f"{info['ticker']}  filled={filled}  remaining={remaining}"
                    )
                    cancelled.append(info)
            except Exception as e:
                log.debug(f"[fills] check_stale {oid[:8]}: {e}")
            finally:
                self._pending.pop(oid, None)

        return cancelled
