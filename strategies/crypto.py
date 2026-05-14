"""
Crypto live price strategy.

Pulls live BTC/ETH prices from Binance (free, no auth) and compares
to Kalshi's KXBTC/KXETH market prices. Uses a log-normal model to
compute the probability that the price closes above/below a given
level by the market's expiry time.

Edge: Kalshi crypto markets often lag real price moves by minutes.
When BTC drops 2%, the "above $X" markets take time to reprice.

Entry: model probability differs from Kalshi ask by 8+ cents
Exit:  8-cent profit | 5-cent stop | 30 min before expiry
"""

import math
import time
import re
import httpx
from datetime import datetime, timezone
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.logger import log
from dataclasses import dataclass


BINANCE_URL     = "https://api.binance.com/api/v3/ticker/price"
ANNUAL_VOL_BTC  = 0.65   # 65% annualized volatility
ANNUAL_VOL_ETH  = 0.80   # 80% annualized volatility
MIN_EDGE        = 0.08   # 8 cent minimum
COOLDOWN_S      = 180
EXPIRY_BUFFER   = 1800   # exit 30 min before expiry
PROFIT_TARGET   = 8      # cents
STOP_LOSS       = 5      # cents
PRICE_POLL_S    = 20     # poll Binance every 20s


TRAIL_DISTANCE = 4   # cents

@dataclass
class CryptoPosition:
    ticker: str
    side: str
    contracts: int
    entry_price_c: int
    entry_time: float
    expiry_ts: float
    high_water_c: int = 0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def above_prob(current: float, strike: float, hours: float, annual_vol: float) -> float:
    """Probability that asset closes above strike given current price and hours to expiry."""
    if hours <= 0:
        return 1.0 if current > strike else 0.0
    T = hours / (24 * 365)
    if T <= 0 or current <= 0 or strike <= 0:
        return 0.5
    d = math.log(current / strike) / (annual_vol * math.sqrt(T))
    return _norm_cdf(d)


class CryptoStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk = risk
        self._positions: dict[str, CryptoPosition] = {}
        self._last_entry: dict[str, float] = {}
        self._last_price_poll: float = 0
        self._prices: dict[str, float] = {}   # symbol -> price
        self._markets: list[dict] = []
        self._markets_ts: float = 0

    def _fetch_prices(self):
        if time.time() - self._last_price_poll < PRICE_POLL_S:
            return
        self._last_price_poll = time.time()
        for symbol, key in [("BTCUSDT", "BTC"), ("ETHUSDT", "ETH")]:
            try:
                r = httpx.get(BINANCE_URL, params={"symbol": symbol}, timeout=5)
                self._prices[key] = float(r.json()["price"])
            except Exception as e:
                log.warning(f"[crypto] Price fetch {symbol}: {e}")

    def _fetch_markets(self):
        if time.time() - self._markets_ts < 60 and self._markets:
            return
        markets = []
        for series in ["KXBTC", "KXETH"]:
            try:
                r = self.kalshi._get("/markets", {"limit": 50, "series_ticker": series})
                for m in r.get("markets", []):
                    if m.get("status") != "active":
                        continue
                    ask = float(m.get("yes_ask_dollars") or 0)
                    bid = float(m.get("yes_bid_dollars") or 0)
                    if ask > 0 and bid > 0:
                        markets.append(m)
            except Exception as e:
                log.warning(f"[crypto] Market fetch {series}: {e}")
        self._markets = markets
        self._markets_ts = time.time()

    def _parse_ticker(self, ticker: str):
        """Extract asset, direction (T=above, B=below), and strike from ticker."""
        # KXBTC-26MAY1600-T88799.99
        m = re.match(r"KX(BTC|ETH)-\d+\w+-([TB])([\d.]+)", ticker)
        if not m:
            return None, None, None
        asset     = m.group(1)
        direction = m.group(2)   # T=above, B=below
        strike    = float(m.group(3))
        return asset, direction, strike

    def _expiry_hours(self, m: dict) -> float:
        ts_str = m.get("close_time") or m.get("expected_expiration_time") or ""
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            secs = (dt - datetime.now(timezone.utc)).total_seconds()
            return max(0, secs / 3600)
        except Exception:
            return 0

    def _check_exits(self):
        now = time.time()
        for ticker, pos in list(self._positions.items()):
            try:
                if pos.expiry_ts and now > pos.expiry_ts - EXPIRY_BUFFER:
                    self._exit(pos, "expiry approaching")
                    continue
                m = self.kalshi.get_market(ticker).get("market", {})
                bid_c = int(float(m.get("yes_bid_dollars" if pos.side == "yes" else "no_bid_dollars") or 0) * 100)
                if bid_c:
                    if bid_c > pos.high_water_c:
                        pos.high_water_c = bid_c
                    trail_stop = pos.high_water_c - TRAIL_DISTANCE
                    pnl = bid_c - pos.entry_price_c
                    if pnl >= PROFIT_TARGET:
                        self._exit(pos, f"profit +{pnl}c")
                    elif pnl <= -STOP_LOSS:
                        self._exit(pos, f"stop {pnl}c")
                    elif bid_c < trail_stop and pos.high_water_c > pos.entry_price_c + 3:
                        self._exit(pos, f"trailing stop (peak={pos.high_water_c}¢)")
            except Exception as e:
                log.warning(f"[crypto] Exit check {ticker}: {e}")

    def _exit(self, pos: CryptoPosition, reason: str):
        try:
            self.kalshi.place_order(
                ticker=pos.ticker, side=pos.side, action="sell",
                count=pos.contracts, order_type="market",
            )
            self.risk.record_close(pos.ticker, pos.contracts)
            log.info(f"[crypto] EXIT {pos.ticker} {pos.side.upper()} x{pos.contracts}  {reason}")
        except Exception as e:
            log.error(f"[crypto] Exit failed {pos.ticker}: {e}")
        finally:
            self._positions.pop(pos.ticker, None)

    def scan(self) -> list[dict]:
        self._fetch_prices()
        self._fetch_markets()
        self._check_exits()

        if not self._prices:
            return []

        signals = []
        now = time.time()

        for m in self._markets:
            ticker = m["ticker"]
            if ticker in self._positions:
                continue
            if now - self._last_entry.get(ticker, 0) < COOLDOWN_S:
                continue

            asset, direction, strike = self._parse_ticker(ticker)
            if not asset:
                continue

            vol_map = {"BTC": ANNUAL_VOL_BTC, "ETH": ANNUAL_VOL_ETH}
            current_price = self._prices.get(asset)
            if not current_price:
                continue

            hours = self._expiry_hours(m)
            if hours <= 0 or hours > 48:
                continue

            annual_vol = vol_map[asset]
            prob_above = above_prob(current_price, strike, hours, annual_vol)

            yes_ask_c = int(float(m.get("yes_ask_dollars") or 0) * 100)
            yes_bid_c = int(float(m.get("yes_bid_dollars") or 0) * 100)
            no_ask_c  = int(float(m.get("no_ask_dollars")  or 0) * 100)

            if direction == "T":   # YES = closes ABOVE strike
                model_yes_c = int(prob_above * 100)
            else:                  # YES = closes BELOW strike (B markets)
                model_yes_c = int((1 - prob_above) * 100)

            edge_yes = model_yes_c - yes_ask_c
            edge_no  = (100 - model_yes_c) - no_ask_c if no_ask_c else -999

            if edge_yes >= int(MIN_EDGE * 100):
                signals.append({
                    "ticker":    ticker, "title": m.get("title","")[:60],
                    "side":      "yes",  "price_c": yes_ask_c,
                    "edge":      edge_yes / 100,
                    "model_c":   model_yes_c,
                    "current":   current_price,
                    "strike":    strike,
                    "hours":     round(hours, 1),
                    "expiry_ts": now + hours * 3600,
                })
            elif edge_no >= int(MIN_EDGE * 100) and no_ask_c:
                signals.append({
                    "ticker":    ticker, "title": m.get("title","")[:60],
                    "side":      "no",   "price_c": no_ask_c,
                    "edge":      edge_no / 100,
                    "model_c":   model_yes_c,
                    "current":   current_price,
                    "strike":    strike,
                    "hours":     round(hours, 1),
                    "expiry_ts": now + hours * 3600,
                })

        signals.sort(key=lambda x: x["edge"], reverse=True)
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
            log.info(f"[crypto] Skipped {ticker}: {reason}")
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
            self._positions[ticker] = CryptoPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, entry_time=time.time(),
                expiry_ts=signal["expiry_ts"], high_water_c=price_c,
            )
            log.info(
                f"[crypto] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}c"
                f"  model={signal['model_c']}c  {signal['current']:,.0f} vs strike {signal['strike']:,.0f}"
                f"  {signal['hours']}h left"
            )
            return True
        except Exception as e:
            log.error(f"[crypto] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            return False
