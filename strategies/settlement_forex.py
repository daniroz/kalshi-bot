"""
Verified Forex Settlement — EUR/USD and USD/JPY at 10am ET open price.

Thesis: Kalshi forex markets resolve on the OPEN price at 10:00 AM ET on a
specific date. In the final 30 minutes before that print, the live pair price
from Yahoo (EURUSD=X / USDJPY=X) is a very tight predictor of where it'll
print — forex is liquid 24/5 and rarely moves more than 20-30 pips in a
30-minute window absent a scheduled news release.

Why this is the lowest-confidence of the verified strategies (and we treat it
with extra care):
  - Resolution is a POINT-IN-TIME print, not an average. A single tick can
    fall on either side of the threshold.
  - Macro releases (NFP at 8:30, CPI at 8:30, housing data at 10:00) can spike
    pairs 50+ pips in seconds.
  - 10:00 AM ET is itself a release window — risk of release-driven volatility
    AT the exact resolution moment.

Defenses:
  - Wider absolute safety margins than other strategies (40 pips for EURUSD,
    30 pips for USDJPY) at >10 min.
  - Hard stop: no trading inside the final 90 seconds before resolution.
  - The MARKET-SANITY 20¢ floor catches data lag (same as other strategies).
  - Skip trading when known macro releases are within ±15min of resolution.

Ticker format (Kalshi):
  KXEURUSD-26MAY2010-T1.17599 → "Will EUR/USD open above 1.17599 at 10am ET May 20?"
  Date format YYMMMDDHH: 26MAY20 + 10 (hour 10 ET = 10am)
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
SCAN_INTERVAL_S      = 15
MIN_EDGE_C           = 10
MIN_TRADE_DOLLARS    = 5.0
MAX_TRADE_DOLLARS    = 25.0       # smaller cap — forex resolution is point-in-time
MAX_PRICE_C          = 89
MIN_PRICE_C          = 25         # tighter MARKET-SANITY floor for forex (more news risk)
RESOLVE_MIN_S        = 90         # 90s hard stop — last 90s is news-spike-deadly
RESOLVE_MAX_S        = 30 * 60    # 30 min before resolution
QUOTE_CACHE_TTL_S    = 5

# Kalshi series → Yahoo Finance forex symbol + pip-margin baseline
SUPPORTED_SERIES = {
    "KXEURUSD": {"symbol": "EURUSD=X", "pip_base": 0.0040},   # 40-pip margin baseline
    "KXUSDJPY": {"symbol": "USDJPY=X", "pip_base": 0.30},     # 30-pip margin baseline (JPY is in whole numbers)
    # Add KXGBPUSD if Kalshi adds it: {"symbol": "GBPUSD=X", "pip_base": 0.0045}
}

_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
           "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}


@dataclass
class ForexPosition:
    ticker:        str
    side:          str
    contracts:     int
    entry_price_c: int
    spot_at_entry: float
    threshold:     float


def _kalshi_fee_cents(price_c: int, contracts: int) -> float:
    p = price_c / 100
    return 0.07 * p * (1 - p) * 100 * contracts


# ── Yahoo Finance forex fetcher ────────────────────────────────────────────────
_quote_cache: dict[str, tuple[float, float]] = {}

def _fetch_fx(symbol: str) -> Optional[float]:
    """Fetch live forex price from Yahoo /v8/chart (no auth)."""
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
        # Prefer last 1-min bar close over potentially-stale regularMarketPrice
        ind = results[0].get("indicators", {}).get("quote", [{}])[0]
        closes = ind.get("close") or []
        recent = [c for c in closes if c is not None]
        if recent:
            price = float(recent[-1])
        else:
            meta = results[0].get("meta", {})
            price = meta.get("regularMarketPrice")
            if price is None:
                return None
            price = float(price)
        _quote_cache[symbol] = (time.time(), price)
        return price
    except Exception as e:
        log.debug(f"[settle-fx] fetch failed {symbol}: {e}")
        return None


# ── Macro-release blackout window ──────────────────────────────────────────────
# US economic releases that hit forex hard. We refuse to trade within ±15 min of these.
# Times in ET. Source: BLS / Fed standard release calendar (most releases).
HIGH_IMPACT_RELEASE_HOURS_ET = [
    (8, 30),    # NFP, CPI, PPI, Retail Sales, GDP advance
    (10, 0),    # ISM, Consumer Confidence, JOLTS — RIGHT at our resolution hour!
    (14, 0),    # FOMC statement days
]

def _macro_blackout(now_et: datetime, resolve_et: datetime) -> bool:
    """Is now within ±15 minutes of a high-impact macro release?
    Also blocks if resolution itself coincides with a release window."""
    BLACKOUT_MIN = 15
    for hh, mm in HIGH_IMPACT_RELEASE_HOURS_ET:
        release_today = now_et.replace(hour=hh, minute=mm, second=0, microsecond=0)
        # Check both 'now' and 'resolve_et' against release window
        for ref in (now_et, resolve_et):
            delta_min = abs((ref - release_today).total_seconds() / 60)
            if delta_min < BLACKOUT_MIN:
                return True
    return False


def _parse_forex_ticker(ticker: str) -> Optional[dict]:
    """KXEURUSD-26MAY2010-T1.17599 → metadata."""
    parts = ticker.split("-")
    if len(parts) != 3:
        return None
    series = parts[0]
    if series not in SUPPORTED_SERIES:
        return None
    cfg = SUPPORTED_SERIES[series]
    date_str = parts[1]
    thr_str  = parts[2]
    if not thr_str.startswith("T"):
        return None
    try:
        threshold = float(thr_str[1:])
    except ValueError:
        return None
    m = re.match(r"(\d{2})([A-Z]{3})(\d{2})(\d{2})$", date_str)
    if not m:
        return None
    yy, mon, dd, hh = m.groups()
    month = _MONTHS.get(mon)
    if not month:
        return None
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        resolve_dt = datetime(2000 + int(yy), month, int(dd), int(hh), 0, tzinfo=et)
        return {
            "series": series, "symbol": cfg["symbol"], "pip_base": cfg["pip_base"],
            "threshold": threshold,
            "resolve_dt_et": resolve_dt,
            "resolve_dt": resolve_dt.astimezone(timezone.utc),
        }
    except Exception:
        return None


def _safety_margin(seconds_left: float, pip_base: float) -> float:
    """Currency-units buffer between live price and threshold."""
    if seconds_left >= 20 * 60: return pip_base * 1.50
    if seconds_left >= 10 * 60: return pip_base * 1.00
    if seconds_left >=  3 * 60: return pip_base * 0.60
    return pip_base * 0.30   # last 90s blocked entirely via RESOLVE_MIN_S anyway


# ── Strategy ───────────────────────────────────────────────────────────────────
class ForexSettlementStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk   = risk
        self._last_scan = 0.0
        self._positions: dict[str, ForexPosition] = {}
        self._lock = threading.Lock()

    def _evaluate(self, m: dict) -> Optional[dict]:
        ticker = m["ticker"]
        parsed = _parse_forex_ticker(ticker)
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

        # Macro blackout: skip if a high-impact release is within ±15 min
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
            now_et = datetime.now(et)
            if _macro_blackout(now_et, parsed["resolve_dt_et"]):
                return None
        except Exception:
            pass

        quote = _fetch_fx(parsed["symbol"])
        if quote is None:
            return None

        threshold = parsed["threshold"]
        margin = _safety_margin(seconds_left, parsed["pip_base"])

        yes_ask_c = int(round(float(m.get("yes_ask_dollars") or 0) * 100))
        no_ask_c  = int(round(float(m.get("no_ask_dollars")  or 0) * 100))
        if yes_ask_c == 0 or no_ask_c == 0:
            return None

        side = price_c = reason = None
        if quote >= threshold + margin and MIN_PRICE_C <= yes_ask_c <= MAX_PRICE_C:
            side, price_c = "yes", yes_ask_c
            reason = (f"{parsed['symbol']}={quote:.5f} ≥ {threshold:.5f}+{margin:.5f} "
                      f"({seconds_left/60:.1f}min)")
        elif quote <= threshold - margin and MIN_PRICE_C <= no_ask_c <= MAX_PRICE_C:
            side, price_c = "no", no_ask_c
            reason = (f"{parsed['symbol']}={quote:.5f} ≤ {threshold:.5f}-{margin:.5f} "
                      f"({seconds_left/60:.1f}min)")
        elif quote >= threshold + margin and yes_ask_c < MIN_PRICE_C:
            log.warning(f"[settle-fx] SKIP {ticker}: model says YES locked "
                        f"({parsed['symbol']}={quote:.5f}≥{threshold:.5f}) but market "
                        f"prices YES at {yes_ask_c}¢ — listen to market.")
            return None
        elif quote <= threshold - margin and no_ask_c < MIN_PRICE_C:
            log.warning(f"[settle-fx] SKIP {ticker}: model says NO locked "
                        f"but market prices NO at {no_ask_c}¢ — listen to market.")
            return None

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
                log.debug(f"[settle-fx] eval {ticker} failed: {e}")
        opps.sort(key=lambda x: x["edge_c"], reverse=True)
        return opps

    def execute(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_c = opp["price_c"]
        edge_c  = opp["edge_c"]

        price = price_c / 100
        tier_cap_dollars = self.risk.balance * self.risk.effective_max_position_pct()
        target_dollars = min(MAX_TRADE_DOLLARS, tier_cap_dollars)
        if target_dollars < MIN_TRADE_DOLLARS:
            log.info(f"[settle-fx] Skip {ticker}: tier cap ${tier_cap_dollars:.2f} below floor")
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
            log.info(f"[settle-fx] Skip {ticker}: {reason}")
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
            self._positions[ticker] = ForexPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, spot_at_entry=opp["spot"],
                threshold=opp["threshold"],
            )
            log.info(
                f"[settle-fx] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢  "
                f"edge={edge_c:.1f}¢  {opp['reason']}"
            )
            return True
        except Exception as e:
            log.error(f"[settle-fx] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            self._positions.pop(ticker, None)
            return False
