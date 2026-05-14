"""
Price momentum strategy.

Tracks mid-price changes across bot cycles. If a market's price has
moved consistently in one direction over the last N cycles, follow it.

Real traders use this because Kalshi prices often lag news — once a
move starts it continues as more traders react.

Entry: price moved 3+ cents in same direction over last 3 checks
Exit: price reverses 2+ cents from peak
"""

import time
from collections import deque
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log
from dataclasses import dataclass, field


LOOKBACK        = 4      # number of price snapshots to track
MIN_MOVE_CENTS  = 3      # minimum total move to qualify as momentum
MIN_PRICE_CENTS = 3
MAX_PRICE_CENTS = 97
COOLDOWN_S      = 180
REVERSAL_CENTS  = 2      # exit if price reverses this much from entry


@dataclass
class MomentumPosition:
    ticker: str
    side: str
    contracts: int
    entry_price_c: int
    peak_price_c: int


class MomentumStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk = risk
        self._price_history: dict[str, deque] = {}
        self._last_entry: dict[str, float] = {}
        self._positions: dict[str, MomentumPosition] = {}

    def _mid(self, m: dict) -> float:
        bid = float(m.get("yes_bid_dollars") or 0)
        ask = float(m.get("yes_ask_dollars") or 0)
        if bid and ask:
            return (bid + ask) / 2
        return ask or bid

    def _update_prices(self, markets: list[dict]):
        for m in markets:
            ticker = m["ticker"]
            mid = self._mid(m)
            if mid == 0:
                continue
            if ticker not in self._price_history:
                self._price_history[ticker] = deque(maxlen=LOOKBACK)
            self._price_history[ticker].append(mid)

    def _check_exits(self, markets: list[dict]):
        price_map = {m["ticker"]: self._mid(m) for m in markets}
        for ticker, pos in list(self._positions.items()):
            current = price_map.get(ticker)
            if not current:
                continue
            current_c = int(current * 100)

            # Update peak
            if pos.side == "yes" and current_c > pos.peak_price_c:
                pos.peak_price_c = current_c
            elif pos.side == "no":
                # For NO, we profit when YES price falls
                if current_c < pos.peak_price_c:
                    pos.peak_price_c = current_c

            # Exit on reversal
            if pos.side == "yes":
                drawdown = pos.peak_price_c - current_c
                if drawdown >= REVERSAL_CENTS:
                    self._exit(pos, f"reversal -{drawdown}¢ from peak")
            else:
                drawdown = current_c - pos.peak_price_c
                if drawdown >= REVERSAL_CENTS:
                    self._exit(pos, f"reversal +{drawdown}¢ from peak (NO trade)")

    def _exit(self, pos: MomentumPosition, reason: str):
        try:
            self.kalshi.place_order(
                ticker=pos.ticker, side=pos.side, action="sell",
                count=pos.contracts, order_type="market",
            )
            self.risk.record_close(pos.ticker, pos.contracts)
            log.info(f"[mom] EXIT {pos.ticker} {pos.side.upper()} x{pos.contracts}  {reason}")
        except Exception as e:
            log.error(f"[mom] Exit failed {pos.ticker}: {e}")
        finally:
            self._positions.pop(pos.ticker, None)

    def scan(self, markets: list[dict] = None) -> list[dict]:
        if markets is None:
            markets = get_liquid_markets(self.kalshi, min_volume=50)

        self._update_prices(markets)
        self._check_exits(markets)

        signals = []
        for m in markets:
            ticker = m["ticker"]
            if ticker in self._positions:
                continue
            if time.time() - self._last_entry.get(ticker, 0) < COOLDOWN_S:
                continue

            history = self._price_history.get(ticker)
            if not history or len(history) < LOOKBACK:
                continue

            prices = list(history)
            moves  = [prices[i+1] - prices[i] for i in range(len(prices)-1)]

            # All moves in same direction = momentum
            if all(d > 0 for d in moves):
                total_move = (prices[-1] - prices[0]) * 100
                if total_move >= MIN_MOVE_CENTS:
                    yes_ask = float(m.get("yes_ask_dollars") or 0)
                    price_c = int(yes_ask * 100)
                    if MIN_PRICE_CENTS <= price_c <= MAX_PRICE_CENTS:
                        signals.append({
                            "ticker":  ticker,
                            "title":   m.get("title", "")[:60],
                            "side":    "yes",
                            "price_c": price_c,
                            "move_c":  round(total_move, 1),
                            "edge":    min(0.04, total_move / 100 * 0.5),
                        })

            elif all(d < 0 for d in moves):
                total_move = (prices[0] - prices[-1]) * 100
                if total_move >= MIN_MOVE_CENTS:
                    no_ask  = float(m.get("no_ask_dollars") or 0)
                    price_c = int(no_ask * 100) if no_ask else 100 - int(float(m.get("yes_bid_dollars") or 0) * 100)
                    if MIN_PRICE_CENTS <= price_c <= MAX_PRICE_CENTS:
                        signals.append({
                            "ticker":  ticker,
                            "title":   m.get("title", "")[:60],
                            "side":    "no",
                            "price_c": price_c,
                            "move_c":  round(total_move, 1),
                            "edge":    min(0.04, total_move / 100 * 0.5),
                        })

        signals.sort(key=lambda x: x["move_c"], reverse=True)
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
            log.info(f"[mom] Skipped {ticker}: {reason}")
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
            self._positions[ticker] = MomentumPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, peak_price_c=price_c,
            )
            log.info(f"[mom] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢  move={signal['move_c']}¢")
            return True
        except Exception as e:
            log.error(f"[mom] Order failed {ticker}: {e}")
            return False
