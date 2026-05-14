"""
Intraday momentum on KXINX / KXNDAQ same-day contracts.

These are the most liquid markets on Kalshi. They ask "will the S&P 500
close above X today?" As the market moves, their prices swing 10-30 cents
within a session — far more edge than any other strategy.

Focus: markets priced 20-80 cents (real uncertainty, real movement).
Entry: price moved 3+ cents in same direction across 3 consecutive readings.
Exit: 8-cent profit target OR 4-cent stop loss OR 30 min before expiry.
"""

import time
from collections import deque
from datetime import datetime, timezone
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.logger import log
from dataclasses import dataclass


SERIES          = ["KXINX", "KXNDAQ", "KXBTC", "KXETH"]
MIN_PRICE       = 20    # cents — only trade markets with real uncertainty
MAX_PRICE       = 80
MIN_VOLUME      = 500   # daily volume floor
LOOKBACK        = 3     # price readings needed
MIN_MOVE        = 3     # cents of consistent movement to trigger
PROFIT_TARGET   = 8     # cents
STOP_LOSS       = 4     # cents
EXPIRY_BUFFER   = 1800  # exit 30 min before market closes
COOLDOWN_S      = 120


TRAIL_DISTANCE = 4   # cents below high water to trigger trailing stop

@dataclass
class IntradayPosition:
    ticker: str
    side: str
    contracts: int
    entry_price_c: int
    entry_time: float
    close_time: float
    high_water_c: int = 0


class IntradayStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk = risk
        self._price_history: dict[str, deque] = {}
        self._positions: dict[str, IntradayPosition] = {}
        self._last_entry: dict[str, float] = {}
        self._markets_cache: list[dict] = []
        self._cache_ts: float = 0

    def _get_markets(self) -> list[dict]:
        if time.time() - self._cache_ts < 60 and self._markets_cache:
            return self._markets_cache

        markets = []
        for series in SERIES:
            try:
                r = self.kalshi._get("/markets", {"limit": 50, "series_ticker": series})
                for m in r.get("markets", []):
                    if m.get("status") != "active":
                        continue
                    bid = float(m.get("yes_bid_dollars") or 0)
                    ask = float(m.get("yes_ask_dollars") or 0)
                    vol = float(m.get("volume_24h_fp") or 0)
                    if bid > 0 and ask > 0 and vol >= MIN_VOLUME:
                        markets.append(m)
            except Exception as e:
                log.warning(f"[intra] {series} fetch failed: {e}")

        self._markets_cache = markets
        self._cache_ts = time.time()
        return markets

    def _mid(self, m: dict) -> float:
        bid = float(m.get("yes_bid_dollars") or 0)
        ask = float(m.get("yes_ask_dollars") or 0)
        return (bid + ask) / 2 if bid and ask else 0

    def _close_ts(self, m: dict) -> float:
        ts_str = m.get("close_time") or m.get("expected_expiration_time") or ""
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return 0

    def _check_exits(self, markets: list[dict]):
        price_map = {m["ticker"]: self._mid(m) for m in markets}
        now = time.time()

        for ticker, pos in list(self._positions.items()):
            current = price_map.get(ticker)
            if not current:
                continue
            current_c = int(current * 100)

            # 30 min before expiry — get out regardless
            if pos.close_time and now > pos.close_time - EXPIRY_BUFFER:
                self._exit(pos, "expiry approaching")
                continue

            # Update high water mark
            if current_c > pos.high_water_c:
                pos.high_water_c = current_c

            pnl = current_c - pos.entry_price_c
            if pos.side == "no":
                pnl = -pnl

            trail_stop = pos.high_water_c - TRAIL_DISTANCE
            if pnl >= PROFIT_TARGET:
                self._exit(pos, f"profit target +{pnl}c")
            elif pnl <= -STOP_LOSS:
                self._exit(pos, f"stop loss {pnl}c")
            elif current_c < trail_stop and pos.high_water_c > pos.entry_price_c + 3:
                self._exit(pos, f"trailing stop (peak={pos.high_water_c}¢ now={current_c}¢)")

    def _exit(self, pos: IntradayPosition, reason: str):
        try:
            self.kalshi.place_order(
                ticker=pos.ticker, side=pos.side, action="sell",
                count=pos.contracts, order_type="market",
            )
            self.risk.record_close(pos.ticker, pos.contracts)
            log.info(f"[intra] EXIT {pos.ticker} {pos.side.upper()} x{pos.contracts}  {reason}")
        except Exception as e:
            log.error(f"[intra] Exit failed {pos.ticker}: {e}")
        finally:
            self._positions.pop(pos.ticker, None)

    def scan(self) -> list[dict]:
        markets = self._get_markets()
        self._check_exits(markets)

        for m in markets:
            ticker = m["ticker"]
            mid = self._mid(m)
            if mid == 0:
                continue
            if ticker not in self._price_history:
                self._price_history[ticker] = deque(maxlen=LOOKBACK + 1)
            self._price_history[ticker].append(mid)

        signals = []
        now = time.time()

        for m in markets:
            ticker = m["ticker"]
            if ticker in self._positions:
                continue
            if now - self._last_entry.get(ticker, 0) < COOLDOWN_S:
                continue

            mid = self._mid(m)
            mid_c = int(mid * 100)
            if not (MIN_PRICE <= mid_c <= MAX_PRICE):
                continue

            close_ts = self._close_ts(m)
            if close_ts and now > close_ts - EXPIRY_BUFFER:
                continue

            history = self._price_history.get(ticker)
            if not history or len(history) < LOOKBACK:
                continue

            prices = list(history)
            moves = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
            total_move = (prices[-1] - prices[0]) * 100

            if all(d > 0 for d in moves) and total_move >= MIN_MOVE:
                yes_ask_c = int(float(m.get("yes_ask_dollars") or 0) * 100)
                signals.append({
                    "ticker":     ticker,
                    "title":      m.get("title", "")[:60],
                    "side":       "yes",
                    "price_c":    yes_ask_c,
                    "move_c":     round(total_move, 1),
                    "edge":       min(0.08, total_move / 100 * 0.6),
                    "close_time": close_ts,
                    "volume":     float(m.get("volume_24h_fp") or 0),
                })
            elif all(d < 0 for d in moves) and abs(total_move) >= MIN_MOVE:
                no_ask = float(m.get("no_ask_dollars") or 0)
                no_ask_c = int(no_ask * 100) if no_ask else 100 - int(float(m.get("yes_bid_dollars") or 0) * 100)
                signals.append({
                    "ticker":     ticker,
                    "title":      m.get("title", "")[:60],
                    "side":       "no",
                    "price_c":    no_ask_c,
                    "move_c":     round(abs(total_move), 1),
                    "edge":       min(0.08, abs(total_move) / 100 * 0.6),
                    "close_time": close_ts,
                    "volume":     float(m.get("volume_24h_fp") or 0),
                })

        signals.sort(key=lambda x: (x["volume"], x["move_c"]), reverse=True)
        return signals

    def execute(self, signal: dict) -> bool:
        ticker  = signal["ticker"]
        side    = signal["side"]
        price_c = signal["price_c"]
        edge    = signal["edge"]

        contracts = self.risk.kelly_contracts(price_c, price_c / 100 + edge, max_contracts=50)
        contracts = max(1, contracts)

        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[intra] Skipped {ticker}: {reason}")
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
            close_ts = signal.get("close_time", 0)
            self._positions[ticker] = IntradayPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, entry_time=time.time(),
                close_time=close_ts, high_water_c=price_c,
            )
            log.info(
                f"[intra] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}c"
                f"  move={signal['move_c']}c  vol={signal['volume']:.0f}"
            )
            return True
        except Exception as e:
            log.error(f"[intra] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            return False
