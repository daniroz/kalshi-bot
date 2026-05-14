"""
Smart money via volume-spike detection.

The trades endpoint is unavailable on Kalshi, so we infer informed
activity from sudden volume surges. When a market's rolling 24h volume
jumps 3x above its own recent average, someone with conviction is
moving it — we follow the direction of the most recent price shift.

Entry: volume ratio >= 3x AND price moved 2+ cents in last window
Exit: 5¢ profit target | 3¢ stop loss | 1-hour time stop
"""

import time
from collections import deque
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log
from dataclasses import dataclass


VOLUME_SPIKE_RATIO  = 3.0   # current vol must be 3x rolling average
VOLUME_HISTORY      = 12    # track last 12 readings (~60s at 5s cycles)
MIN_PRICE_CENTS     = 3
MAX_PRICE_CENTS     = 97
COOLDOWN_S          = 120
PROFIT_TARGET_CENTS = 5
STOP_LOSS_CENTS     = 3
MAX_HOLD_SECONDS    = 3600


@dataclass
class SmartPosition:
    ticker: str
    side: str
    contracts: int
    entry_price_c: int
    entry_time: float


class SmartMoneyStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager, whale_threshold: int = 25):
        self.kalshi = kalshi
        self.risk = risk
        self._vol_history: dict[str, deque] = {}
        self._price_history: dict[str, deque] = {}
        self._last_entry: dict[str, float] = {}
        self._positions: dict[str, SmartPosition] = {}

    def _update_history(self, markets: list[dict]):
        for m in markets:
            ticker = m["ticker"]
            vol = float(m.get("volume_24h_fp") or 0)
            yes_ask = float(m.get("yes_ask_dollars") or 0)
            yes_bid = float(m.get("yes_bid_dollars") or 0)
            mid = (yes_ask + yes_bid) / 2 if yes_ask and yes_bid else yes_ask or yes_bid

            if ticker not in self._vol_history:
                self._vol_history[ticker] = deque(maxlen=VOLUME_HISTORY)
                self._price_history[ticker] = deque(maxlen=4)

            self._vol_history[ticker].append(vol)
            if mid > 0:
                self._price_history[ticker].append(mid)

    def _check_exits(self, markets: list[dict]):
        price_map = {}
        for m in markets:
            yes_bid = float(m.get("yes_bid_dollars") or 0)
            no_bid  = float(m.get("no_bid_dollars") or 0)
            price_map[m["ticker"]] = (yes_bid, no_bid)

        for ticker, pos in list(self._positions.items()):
            now = time.time()

            if now - pos.entry_time > MAX_HOLD_SECONDS:
                self._exit(pos, "time stop (1h)")
                continue

            bids = price_map.get(ticker)
            if not bids:
                continue
            yes_bid, no_bid = bids
            current_c = int(yes_bid * 100) if pos.side == "yes" else int(no_bid * 100)
            if current_c == 0:
                continue

            pnl = current_c - pos.entry_price_c
            if pnl >= PROFIT_TARGET_CENTS:
                self._exit(pos, f"profit target (+{pnl}¢)")
            elif pnl <= -STOP_LOSS_CENTS:
                self._exit(pos, f"stop loss ({pnl}¢)")

    def _exit(self, pos: SmartPosition, reason: str):
        try:
            self.kalshi.place_order(
                ticker=pos.ticker, side=pos.side, action="sell",
                count=pos.contracts, order_type="market",
            )
            self.risk.record_close(pos.ticker, pos.contracts)
            log.info(f"[smart] EXIT {pos.ticker} {pos.side.upper()} x{pos.contracts}  {reason}")
        except Exception as e:
            log.error(f"[smart] Exit failed {pos.ticker}: {e}")
        finally:
            self._positions.pop(pos.ticker, None)

    def scan(self) -> list[dict]:
        markets = get_liquid_markets(self.kalshi, min_volume=100)
        self._update_history(markets)
        self._check_exits(markets)

        signals = []
        for m in markets:
            ticker = m["ticker"]

            if ticker in self._positions:
                continue
            if time.time() - self._last_entry.get(ticker, 0) < COOLDOWN_S:
                continue

            vols = self._vol_history.get(ticker)
            prices = self._price_history.get(ticker)
            if not vols or len(vols) < 4 or not prices or len(prices) < 2:
                continue

            rolling_avg = sum(list(vols)[:-1]) / (len(vols) - 1)
            current_vol = vols[-1]
            if rolling_avg < 10 or current_vol / rolling_avg < VOLUME_SPIKE_RATIO:
                continue

            price_list = list(prices)
            price_move = (price_list[-1] - price_list[0]) * 100

            yes_ask = float(m.get("yes_ask_dollars") or 0)
            no_ask  = float(m.get("no_ask_dollars") or 0)

            if price_move >= 2 and yes_ask:
                price_c = int(yes_ask * 100)
                if MIN_PRICE_CENTS <= price_c <= MAX_PRICE_CENTS:
                    signals.append({
                        "ticker":       ticker,
                        "title":        m.get("title", "")[:60],
                        "side":         "yes",
                        "price_c":      price_c,
                        "whale_size":   int(current_vol),
                        "vol_ratio":    round(current_vol / rolling_avg, 1),
                    })
            elif price_move <= -2 and no_ask:
                price_c = int(no_ask * 100) if no_ask else 100 - int(float(m.get("yes_bid_dollars") or 0) * 100)
                if MIN_PRICE_CENTS <= price_c <= MAX_PRICE_CENTS:
                    signals.append({
                        "ticker":       ticker,
                        "title":        m.get("title", "")[:60],
                        "side":         "no",
                        "price_c":      price_c,
                        "whale_size":   int(current_vol),
                        "vol_ratio":    round(current_vol / rolling_avg, 1),
                    })

        signals.sort(key=lambda x: x["vol_ratio"], reverse=True)
        return signals

    def execute(self, signal: dict) -> bool:
        ticker  = signal["ticker"]
        side    = signal["side"]
        price_c = signal["price_c"]

        contracts = self.risk.kelly_contracts(price_c, price_c / 100 + 0.03, max_contracts=50)
        contracts = max(1, contracts)

        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, 0.03 * contracts)
        if not ok:
            log.info(f"[smart] Skipped {ticker}: {reason}")
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
            self._positions[ticker] = SmartPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, entry_time=time.time(),
            )
            log.info(
                f"[smart] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢  "
                f"vol_ratio={signal['vol_ratio']}x"
            )
            return True
        except Exception as e:
            log.error(f"[smart] Order failed {ticker}: {e}")
            return False
