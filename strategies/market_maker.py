"""
Market-making strategy: post limit orders on both sides of the spread
on liquid Kalshi markets and collect the bid-ask spread.

Targets markets with:
- 24h volume > $500
- Spread >= 4 cents
- Price between 15¢ and 85¢ (avoid near-resolution markets)
"""

import time
from dataclasses import dataclass
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


MIN_VOLUME_24H     = 500.0
MIN_SPREAD_CENTS   = 4
MIN_PRICE_CENTS    = 3
MAX_PRICE_CENTS    = 97
CONTRACTS_PER_SIDE = 3
MAX_MARKETS        = 3
HALF_FILL_EXIT_S   = 300   # exit a one-sided fill after 5 min if not resolved


@dataclass
class HalfFill:
    ticker: str
    side: str        # "yes" or "no" — the side that filled
    contracts: int
    fill_ts: float
    resting_order_id: str   # the OTHER side still resting


class MarketMakerStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk = risk
        self._active_quotes: dict[str, tuple[str, str]] = {}   # ticker -> (bid_id, ask_id)
        self._half_fills:    dict[str, HalfFill] = {}          # ticker -> HalfFill

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
        """Cancel stale quotes, check for one-sided fills, re-quote best markets."""
        self._check_half_fills()

        markets = get_liquid_markets(self.kalshi, min_volume=MIN_VOLUME_24H)
        good = [m for m in markets if self._is_good_market(m)]
        good.sort(key=lambda m: m.get("volume_24h_fp", 0), reverse=True)
        targets = good[:MAX_MARKETS]

        # Get all currently-resting order IDs from Kalshi
        try:
            resting_ids = {
                o["order_id"]
                for o in self.kalshi.get_orders(status="resting").get("orders", [])
            }
        except Exception:
            resting_ids = None   # API failure — skip fill check this cycle

        # Check each active quote pair for one-sided fills
        if resting_ids is not None:
            for ticker, (bid_id, ask_id) in list(self._active_quotes.items()):
                bid_resting = bid_id in resting_ids if bid_id else False
                ask_resting = ask_id in resting_ids if ask_id else False

                if bid_id and not bid_resting and ask_id and ask_resting:
                    # Bid filled, ask still open — we're long YES
                    log.info(f"[mm] Half-fill detected {ticker}: bid filled, ask resting")
                    self._active_quotes.pop(ticker, None)
                    self._half_fills[ticker] = HalfFill(
                        ticker=ticker, side="yes", contracts=CONTRACTS_PER_SIDE,
                        fill_ts=time.time(), resting_order_id=ask_id,
                    )
                elif ask_id and not ask_resting and bid_id and bid_resting:
                    # Ask filled, bid still open — we're short YES (long NO)
                    log.info(f"[mm] Half-fill detected {ticker}: ask filled, bid resting")
                    self._active_quotes.pop(ticker, None)
                    self._half_fills[ticker] = HalfFill(
                        ticker=ticker, side="no", contracts=CONTRACTS_PER_SIDE,
                        fill_ts=time.time(), resting_order_id=bid_id,
                    )

        # Cancel existing quotes not in target list
        target_tickers = {m["ticker"] for m in targets}
        for ticker in list(self._active_quotes.keys()):
            if ticker not in target_tickers:
                self._cancel_quotes(ticker)

        # Post new quotes
        for m in targets:
            ticker = m["ticker"]
            if ticker in self._active_quotes or ticker in self._half_fills:
                continue
            self._post_quotes(m)

    def _check_half_fills(self):
        """Exit half-filled positions that have been open too long."""
        now = time.time()
        for ticker, hf in list(self._half_fills.items()):
            age = now - hf.fill_ts
            if age < HALF_FILL_EXIT_S:
                continue

            # Cancel the resting other leg
            try:
                self.kalshi.cancel_order(hf.resting_order_id)
            except Exception:
                pass

            # Exit the filled position at market (limit at current bid/ask)
            try:
                m = self.kalshi.get_market(ticker).get("market", {})
                if hf.side == "yes":
                    exit_price = int(float(m.get("yes_bid_dollars") or 0.01) * 100)
                    self.kalshi.place_order(
                        ticker=ticker, side="yes", action="sell",
                        count=hf.contracts, order_type="limit",
                        yes_price=max(1, exit_price),
                    )
                else:
                    exit_price = int(float(m.get("no_bid_dollars") or 0.01) * 100)
                    self.kalshi.place_order(
                        ticker=ticker, side="no", action="sell",
                        count=hf.contracts, order_type="limit",
                        no_price=max(1, exit_price),
                    )
                self.risk.record_close(ticker, hf.contracts)
                log.info(f"[mm] Half-fill exit {ticker} {hf.side.upper()} x{hf.contracts} after {age/60:.1f}m")
            except Exception as e:
                log.error(f"[mm] Half-fill exit failed {ticker}: {e}")
            finally:
                self._half_fills.pop(ticker, None)

    def _post_quotes(self, m: dict):
        ticker   = m["ticker"]
        bid_c    = int(round(float(m.get("yes_bid_dollars") or 0) * 100))
        ask_c    = int(round(float(m.get("yes_ask_dollars") or 0) * 100))
        mid_c    = (bid_c + ask_c) // 2
        our_bid  = mid_c - 1
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
            self.risk.record_open(ticker, CONTRACTS_PER_SIDE)
        except Exception as e:
            log.error(f"[mm] Bid failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, CONTRACTS_PER_SIDE)
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
        self.risk.record_close(ticker, CONTRACTS_PER_SIDE)

    def cancel_all(self):
        for ticker in list(self._active_quotes.keys()):
            self._cancel_quotes(ticker)
        # Also cancel resting legs from half-fills
        for hf in self._half_fills.values():
            try:
                self.kalshi.cancel_order(hf.resting_order_id)
                self.risk.record_close(hf.ticker, hf.contracts)
            except Exception:
                pass
        self._half_fills.clear()
