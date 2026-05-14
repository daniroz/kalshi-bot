"""
Market-making strategy: post limit orders on both sides of the spread
on liquid Kalshi markets and collect the bid-ask spread.

Targets markets with:
- 24h volume > $500
- Spread >= 4 cents
- Price between 15¢ and 85¢ (avoid near-resolution markets)
"""

from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


MIN_VOLUME_24H   = 500.0
MIN_SPREAD_CENTS = 4
MIN_PRICE_CENTS  = 3
MAX_PRICE_CENTS  = 97
CONTRACTS_PER_SIDE = 5


class MarketMakerStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk = risk
        self._active_quotes: dict[str, tuple[str, str]] = {}  # ticker -> (bid_id, ask_id)

    def _is_good_market(self, m: dict) -> bool:
        vol  = float(m.get("volume_24h_fp") or 0)
        bid  = float(m.get("yes_bid_dollars") or 0)
        ask  = float(m.get("yes_ask_dollars") or 0)
        spread_c = round((ask - bid) * 100)
        price_c  = round(bid * 100)
        return (
            vol >= MIN_VOLUME_24H
            and spread_c >= MIN_SPREAD_CENTS
            and MIN_PRICE_CENTS <= price_c <= MAX_PRICE_CENTS
        )

    def refresh_quotes(self):
        """Cancel stale quotes and re-quote best markets."""
        markets = get_liquid_markets(self.kalshi, min_volume=MIN_VOLUME_24H)
        good = [m for m in markets if self._is_good_market(m)]
        good.sort(key=lambda m: m.get("volume_24h_fp", 0), reverse=True)
        targets = good[:5]  # quote top-5 by volume

        # Cancel existing quotes not in targets
        target_tickers = {m["ticker"] for m in targets}
        for ticker in list(self._active_quotes.keys()):
            if ticker not in target_tickers:
                self._cancel_quotes(ticker)

        # Post new quotes
        for m in targets:
            ticker = m["ticker"]
            if ticker in self._active_quotes:
                continue
            self._post_quotes(m)

    def _post_quotes(self, m: dict):
        ticker   = m["ticker"]
        bid_c    = int(round(float(m["yes_bid_dollars"] or 0) * 100))
        ask_c    = int(round(float(m["yes_ask_dollars"] or 0) * 100))
        mid_c    = (bid_c + ask_c) // 2
        our_bid  = mid_c - 1   # 1 cent inside mid
        our_ask  = mid_c + 1

        if our_bid < 1 or our_ask > 99:
            return

        ok_b, reason = self.risk.approve_trade(ticker, our_bid, CONTRACTS_PER_SIDE, 0.02 * CONTRACTS_PER_SIDE)
        if not ok_b:
            log.info(f"[mm] Skipping {ticker}: {reason}")
            return

        bid_id = ask_id = None
        try:
            r = self.kalshi.place_order(
                ticker=ticker, side="yes", action="buy",
                count=CONTRACTS_PER_SIDE, order_type="limit",
                yes_price=our_bid,
            )
            bid_id = r.get("order", {}).get("order_id")
        except Exception as e:
            log.error(f"[mm] Bid failed {ticker}: {e}")
            return

        try:
            r = self.kalshi.place_order(
                ticker=ticker, side="yes", action="sell",
                count=CONTRACTS_PER_SIDE, order_type="limit",
                yes_price=our_ask,
            )
            ask_id = r.get("order", {}).get("order_id")
        except Exception as e:
            log.error(f"[mm] Ask failed {ticker}: {e}")

        if bid_id or ask_id:
            self._active_quotes[ticker] = (bid_id, ask_id)
            log.info(f"[mm] Quoted {ticker}  bid={our_bid}¢  ask={our_ask}¢  x{CONTRACTS_PER_SIDE}")

    def _cancel_quotes(self, ticker: str):
        bid_id, ask_id = self._active_quotes.pop(ticker, (None, None))
        for oid in (bid_id, ask_id):
            if oid:
                try:
                    self.kalshi.cancel_order(oid)
                except Exception:
                    pass

    def cancel_all(self):
        for ticker in list(self._active_quotes.keys()):
            self._cancel_quotes(ticker)
