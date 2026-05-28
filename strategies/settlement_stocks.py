"""
Verified Stock Settlement — S&P 500 end-of-day close markets.

Thesis: In the final minutes before the 4pm EDT close, the live SPX value is
a near-perfect estimate of where the index will settle. We only trade when
the live index is comfortably on one side of the threshold AND the market
still prices the locked-in outcome below 92¢.

Ticker format (Kalshi):
  KXINX-26MAY22H1600-T7849.9999  → "Will S&P 500 close above 7849.9999 on May 22?"
  (Note: despite the "INX" series name, this IS the S&P 500 per the rules.)

Data source: Yahoo Finance free quote endpoint (no auth). Caches 10s.

Safety margins (the amount index must beat threshold by, in index points):
  ≥10 minutes to close: SPX moves can be ±15 → require 8 point buffer
  3-10 minutes:                                    4 points
  1-3 minutes:                                     2 points
  <1 minute:                                       1 point

Only operates 3:30 PM – 4:00 PM ET on weekdays — outside that window, no edge.
"""

import re
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


# ── Tunables ───────────────────────────────────────────────────────────────────
SCAN_INTERVAL_S      = 20
MIN_EDGE_C           = 10
MIN_TRADE_DOLLARS    = 5.0        # honors global risk-manager floor
MAX_TRADE_DOLLARS    = 70.0       # sized up 2x (May 28)
MAX_PRICE_C          = 89         # ≥10¢ edge → 90+ unreachable; matches math
MIN_PRICE_C          = 40         # POST-BACKTEST raised 20→40 — require market agreement
RESOLVE_MIN_S        = 180        # was 60 — last 3min is where crashes happen + Yahoo lag bites
RESOLVE_MAX_S        = 20 * 60    # 20 min (was 30 — stocks reprice fast near close)
QUOTE_CACHE_TTL_S    = 10

# MARKET-SANITY FLOOR: if our model says "locked at 99%" but the order book
# prices it below MIN_PRICE_C, the market knows something we don't (typically
# our data feed is lagging a real-time move). Refuse the trade — listen to
# the market. This is THE check that would have prevented the KXINX-MAY19
# disaster: yes_ask was 1¢, our model said locked, market was correct, we lost.

# Kalshi series → Yahoo Finance symbol
SUPPORTED_SERIES = {
    "KXINX":      "^GSPC",   # S&P 500
    "KXSPXCLOSE": "^GSPC",   # also S&P 500 (older series name)
    "KXNDAQ":     "^NDX",    # NASDAQ-100
    "KXDJI":      "^DJI",    # Dow
}

_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
           "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}


@dataclass
class StockPosition:
    ticker:        str
    side:          str
    contracts:     int
    entry_price_c: int
    spot_at_entry: float
    threshold:     float


def _kalshi_fee_cents(price_c: int, contracts: int) -> float:
    p = price_c / 100
    return 0.07 * p * (1 - p) * 100 * contracts


# ── Yahoo Finance fetcher ──────────────────────────────────────────────────────
_quote_cache: dict[str, tuple[float, float]] = {}

def _fetch_quote(symbol: str) -> Optional[float]:
    """Fetch live price from Yahoo's /v8/chart endpoint (no auth required)."""
    cached = _quote_cache.get(symbol)
    if cached and time.time() - cached[0] < QUOTE_CACHE_TTL_S:
        return cached[1]
    try:
        r = httpx.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"interval": "1m", "range": "1d"},
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("chart", {}).get("result", [])
        if not results:
            return None
        meta = results[0].get("meta", {})
        # regularMarketPrice is the live print during RTH; falls back to chartPreviousClose off-hours.
        price = meta.get("regularMarketPrice")
        if price is None:
            return None
        price = float(price)
        _quote_cache[symbol] = (time.time(), price)
        return price
    except Exception as e:
        log.debug(f"[settle-stocks] quote fetch failed {symbol}: {e}")
        return None


# ── Market-hours check ─────────────────────────────────────────────────────────
def _us_market_open() -> bool:
    """Is the US stock market currently open? (rough check — no holiday awareness)."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:
            return False
        minutes = now_et.hour * 60 + now_et.minute
        return 9*60 + 30 <= minutes < 16*60
    except Exception:
        return True   # assume open if TZ lookup fails


def _safety_margin_points(seconds_left: float, symbol: str) -> float:
    """Required index-point buffer between spot and threshold."""
    # Different indices have different scales — buffer scales accordingly
    scale = {"^GSPC": 1.0, "^NDX": 3.0, "^DJI": 7.0}.get(symbol, 1.0)
    if seconds_left >= 10 * 60: return 8.0 * scale
    if seconds_left >=  3 * 60: return 4.0 * scale
    if seconds_left >=  1 * 60: return 2.0 * scale
    return 1.0 * scale


def _parse_stock_ticker(ticker: str) -> Optional[dict]:
    """KXINX-26MAY22H1600-T7849.9999 → metadata or None."""
    parts = ticker.split("-")
    if len(parts) != 3:
        return None
    series = parts[0]
    if series not in SUPPORTED_SERIES:
        return None
    date_str = parts[1]
    thr_str  = parts[2]
    if not thr_str.startswith("T"):
        return None
    try:
        threshold = float(thr_str[1:])
    except ValueError:
        return None
    # Format: YYMMMDDH<HHMM>  e.g. 26MAY22H1600
    m = re.match(r"(\d{2})([A-Z]{3})(\d{2})H(\d{4})$", date_str)
    if not m:
        # Some series might not use the H prefix — fall back to YYMMMDD with implicit 4pm
        m2 = re.match(r"(\d{2})([A-Z]{3})(\d{2})$", date_str)
        if not m2:
            return None
        yy, mon, dd = m2.groups()
        hhmm = "1600"
    else:
        yy, mon, dd, hhmm = m.groups()
    month = _MONTHS.get(mon)
    if not month:
        return None
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        resolve_dt = datetime(2000 + int(yy), month, int(dd),
                              int(hhmm[:2]), int(hhmm[2:]), tzinfo=et)
        return {
            "series": series,
            "symbol": SUPPORTED_SERIES[series],
            "threshold": threshold,
            "resolve_dt": resolve_dt.astimezone(timezone.utc),
        }
    except Exception:
        return None


# ── Strategy ───────────────────────────────────────────────────────────────────
class StockSettlementStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk   = risk
        self._last_scan = 0.0
        self._positions: dict[str, StockPosition] = {}
        self._lock = threading.Lock()

    def _evaluate(self, m: dict) -> Optional[dict]:
        ticker = m["ticker"]
        if not _us_market_open():
            return None
        parsed = _parse_stock_ticker(ticker)
        if not parsed:
            return None

        close_ts_str = m.get("close_time") or ""
        try:
            resolve_dt = datetime.fromisoformat(close_ts_str.replace("Z", "+00:00"))
        except Exception:
            resolve_dt = parsed["resolve_dt"]
        seconds_left = (resolve_dt - datetime.now(timezone.utc)).total_seconds()
        if not (RESOLVE_MIN_S <= seconds_left <= RESOLVE_MAX_S):
            return None

        quote = _fetch_quote(parsed["symbol"])
        if quote is None:
            return None

        threshold = parsed["threshold"]
        margin = _safety_margin_points(seconds_left, parsed["symbol"])

        yes_ask_c = int(round(float(m.get("yes_ask_dollars") or 0) * 100))
        no_ask_c  = int(round(float(m.get("no_ask_dollars")  or 0) * 100))
        if yes_ask_c == 0 or no_ask_c == 0:
            return None

        side = price_c = reason = None
        if quote >= threshold + margin and MIN_PRICE_C <= yes_ask_c <= MAX_PRICE_C:
            side, price_c = "yes", yes_ask_c
            reason = (f"{parsed['symbol']}={quote:.2f} ≥ {threshold:.2f}+{margin:.1f} "
                      f"({seconds_left/60:.1f}min)")
        elif quote <= threshold - margin and MIN_PRICE_C <= no_ask_c <= MAX_PRICE_C:
            side, price_c = "no", no_ask_c
            reason = (f"{parsed['symbol']}={quote:.2f} ≤ {threshold:.2f}-{margin:.1f} "
                      f"({seconds_left/60:.1f}min)")
        # Catch the dangerous case where our model says LOCKED but market disagrees.
        # Log it so we can debug — but DO NOT trade.
        elif quote >= threshold + margin and yes_ask_c < MIN_PRICE_C:
            log.warning(f"[settle-stocks] SKIP {ticker}: model says YES locked "
                        f"({parsed['symbol']}={quote:.2f}≥{threshold:.2f}) but market "
                        f"prices it at {yes_ask_c}¢ — market wins, our data is likely lagging.")
        elif quote <= threshold - margin and no_ask_c < MIN_PRICE_C:
            log.warning(f"[settle-stocks] SKIP {ticker}: model says NO locked "
                        f"({parsed['symbol']}={quote:.2f}≤{threshold:.2f}) but market "
                        f"prices NO at {no_ask_c}¢ — market wins, our data is likely lagging.")

        if side is None:
            return None

        edge_c = (100 - price_c) - _kalshi_fee_cents(price_c, 1)
        if edge_c < MIN_EDGE_C:
            return None

        return {
            "ticker": ticker, "side": side, "price_c": price_c,
            "spot": quote, "threshold": threshold,
            "seconds_left": seconds_left, "edge_c": edge_c, "reason": reason,
        }

    def scan(self) -> list[dict]:
        now = time.time()
        with self._lock:
            if now - self._last_scan < SCAN_INTERVAL_S:
                return []
            self._last_scan = now
        if not _us_market_open():
            return []

        markets = [m for m in get_liquid_markets(self.kalshi, min_volume=0)
                   if m["ticker"].split("-")[0] in SUPPORTED_SERIES]

        opps = []
        for m in markets:
            ticker = m["ticker"]
            if ticker in self._positions:
                continue
            if self.risk.open_positions.get(ticker, 0) > 0:
                continue
            try:
                opp = self._evaluate(m)
                if opp:
                    opps.append(opp)
            except Exception as e:
                log.debug(f"[settle-stocks] eval {ticker} failed: {e}")
        opps.sort(key=lambda x: x["edge_c"], reverse=True)
        return opps

    def execute(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_c = opp["price_c"]
        edge_c  = opp["edge_c"]

        # Tier-aware sizing: target min(MAX_TRADE_DOLLARS, tier-adjusted cap).
        price = price_c / 100
        tier_cap_dollars = self.risk.balance * self.risk.effective_max_position_pct()
        target_dollars = min(MAX_TRADE_DOLLARS, tier_cap_dollars)
        if target_dollars < MIN_TRADE_DOLLARS:
            log.info(f"[settle-stocks] Skip {ticker}: tier cap ${tier_cap_dollars:.2f} below floor ${MIN_TRADE_DOLLARS}")
            return False
        contracts = int(target_dollars / price)
        if contracts <= 0:
            return False

        with self._lock:
            if ticker in self._positions:
                return False
            self._positions[ticker] = None

        edge_dollars_total = (edge_c / 100) * contracts
        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge_dollars_total)
        if not ok:
            log.info(f"[settle-stocks] Skip {ticker}: {reason}")
            self._positions.pop(ticker, None)
            return False

        try:
            self.kalshi.place_order(
                ticker=ticker, side=side, action="buy",
                count=contracts, order_type="limit",
                yes_price=price_c if side == "yes" else None,
                no_price =price_c if side == "no"  else None,
            )
            self.risk.record_open(ticker, contracts)
            self._positions[ticker] = StockPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, spot_at_entry=opp["spot"],
                threshold=opp["threshold"],
            )
            log.info(
                f"[settle-stocks] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢  "
                f"edge={edge_c:.1f}¢  {opp['reason']}"
            )
            return True
        except Exception as e:
            log.error(f"[settle-stocks] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            self._positions.pop(ticker, None)
            return False
