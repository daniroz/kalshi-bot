"""
Verified Commodities Settlement — WTI crude oil daily settle markets.

Thesis: WTI crude oil settles on NYMEX at 2:30 PM ET. In the final 30 minutes
before that settle, the live front-month futures price (CL=F on Yahoo) is a
near-perfect predictor of the official settlement value. When live price is
comfortably above/below a Kalshi threshold AND the order book still prices
the locked outcome ≤89¢ AND ≥20¢ (sanity), we trade.

Ticker format (Kalshi):
  KXWTI-26MAY2014-T99.99
    → "Will WTI crude oil settle above $99.99 USD/Bbl on May 20, 2026?"
  Date format YYMMMDDHH: 26MAY20 + 14 (hour 14 ET = 2pm = settle window)

Why this should be safe + profitable:
  - WTI futures are deeply liquid; intraday moves in last 30min are usually <$0.50
  - Yahoo CL=F is live during NYMEX hours
  - Settlement window (14:28-14:30 ET) averages, so we're approximating that with
    the spot near 14:00 — minor model error but very small (typically <$0.25)
  - Safety margins scale with time-to-settle and asset volatility

Safety margins (USD per barrel WTI):
  ≥20 minutes to settle: $1.50
  10-20 minutes:         $1.00
  3-10 minutes:          $0.50
  <3 minutes:            $0.25

We only operate during NYMEX intraday hours (9am ET → 2:30pm ET) — settles only
happen during these windows. Off-hours, no live price = no trade.

All defensive layers shared with sibling settlement strategies:
  - MARKET-SANITY floor at 20¢ (refuse if market disagrees by >50¢)
  - Tier-aware position sizing
  - Same-game conflict block via risk manager
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
MIN_TRADE_DOLLARS    = 5.0
MAX_TRADE_DOLLARS    = 70.0       # sized up 2x (May 28)
MAX_PRICE_C          = 89
MIN_PRICE_C          = 40         # POST-BACKTEST raised 20→40 — require market agreement
RESOLVE_MIN_S        = 180        # at least 3min cushion — final 3min is most dangerous
RESOLVE_MAX_S        = 20 * 60    # 20 min (was 30)
QUOTE_CACHE_TTL_S    = 10

# Kalshi series → Yahoo Finance futures symbol
SUPPORTED_SERIES = {
    "KXWTI":  "CL=F",   # WTI crude oil front-month
    # Add gold / nat gas / silver here if Kalshi adds markets we want
    # "KXGOLD":   "GC=F",
    # "KXNATGAS": "NG=F",
    # "KXSILVER": "SI=F",
}

_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
           "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}


@dataclass
class CommodityPosition:
    ticker:        str
    side:          str
    contracts:     int
    entry_price_c: int
    spot_at_entry: float
    threshold:     float


def _kalshi_fee_cents(price_c: int, contracts: int) -> float:
    p = price_c / 100
    return 0.07 * p * (1 - p) * 100 * contracts


# ── Yahoo Finance futures fetcher ──────────────────────────────────────────────
_quote_cache: dict[str, tuple[float, float]] = {}

def _fetch_futures(symbol: str) -> Optional[float]:
    """Get last 1-min bar close from Yahoo chart endpoint — bypasses the stale
    regularMarketPrice issue that hit settlement_stocks earlier."""
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

        # Prefer the most-recent 1-min bar close if available — fresher than
        # regularMarketPrice during fast markets.
        ind = results[0].get("indicators", {}).get("quote", [{}])[0]
        closes = ind.get("close") or []
        recent = [c for c in closes if c is not None]
        if recent:
            price = float(recent[-1])
        else:
            price = meta.get("regularMarketPrice")
            if price is None:
                return None
            price = float(price)

        _quote_cache[symbol] = (time.time(), price)
        return price
    except Exception as e:
        log.debug(f"[settle-comm] quote fetch failed {symbol}: {e}")
        return None


# ── NYMEX-hours check (rough — no holiday calendar) ────────────────────────────
def _nymex_intraday() -> bool:
    """Is NYMEX in intraday trading? Crude settles at 14:30 ET; pit hours are
    roughly 9:00-14:30. Outside those hours we don't trust the futures price."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:   # Sat/Sun
            return False
        minutes = now_et.hour * 60 + now_et.minute
        return 9*60 <= minutes < 14*60 + 30
    except Exception:
        return True


# ── Ticker parsing ─────────────────────────────────────────────────────────────
def _parse_commodity_ticker(ticker: str) -> Optional[dict]:
    """
    KXWTI-26MAY2014-T99.99 → metadata
    Date format YYMMMDDHH (e.g. 26MAY20 + 14 → May 20 2026 at 2pm ET = settle window)
    """
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
        # NYMEX settle is at HH:30 of the indicated hour (e.g. KXWTI ...14 → 14:30 ET)
        resolve_dt = datetime(2000 + int(yy), month, int(dd), int(hh), 30, tzinfo=et)
        return {
            "series": series,
            "symbol": SUPPORTED_SERIES[series],
            "threshold": threshold,
            "resolve_dt": resolve_dt.astimezone(timezone.utc),
        }
    except Exception:
        return None


def _safety_margin_usd(seconds_left: float, symbol: str) -> float:
    """USD buffer between live price and threshold to call outcome 'locked'.
    Tuned per symbol — CL trades in dollars, GC in dollars (much higher base)."""
    # Default tuned for WTI (CL=F) where intraday ATR is ~$1-2
    base = {"CL=F": 1.0, "GC=F": 8.0, "NG=F": 0.15, "SI=F": 0.30}.get(symbol, 1.0)
    if seconds_left >= 20 * 60: return base * 1.50
    if seconds_left >= 10 * 60: return base * 1.00
    if seconds_left >=  3 * 60: return base * 0.50
    return base * 0.25


# ── Strategy ───────────────────────────────────────────────────────────────────
class CommoditySettlementStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk   = risk
        self._last_scan = 0.0
        self._positions: dict[str, CommodityPosition] = {}
        self._lock = threading.Lock()

    def _evaluate(self, m: dict) -> Optional[dict]:
        ticker = m["ticker"]
        if not _nymex_intraday():
            return None
        parsed = _parse_commodity_ticker(ticker)
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

        quote = _fetch_futures(parsed["symbol"])
        if quote is None:
            return None

        threshold = parsed["threshold"]
        margin = _safety_margin_usd(seconds_left, parsed["symbol"])

        yes_ask_c = int(round(float(m.get("yes_ask_dollars") or 0) * 100))
        no_ask_c  = int(round(float(m.get("no_ask_dollars")  or 0) * 100))
        if yes_ask_c == 0 or no_ask_c == 0:
            return None

        side = price_c = reason = None
        if quote >= threshold + margin and MIN_PRICE_C <= yes_ask_c <= MAX_PRICE_C:
            side, price_c = "yes", yes_ask_c
            reason = (f"{parsed['symbol']}=${quote:.2f} ≥ ${threshold:.2f}+${margin:.2f} "
                      f"({seconds_left/60:.1f}min)")
        elif quote <= threshold - margin and MIN_PRICE_C <= no_ask_c <= MAX_PRICE_C:
            side, price_c = "no", no_ask_c
            reason = (f"{parsed['symbol']}=${quote:.2f} ≤ ${threshold:.2f}-${margin:.2f} "
                      f"({seconds_left/60:.1f}min)")
        elif quote >= threshold + margin and yes_ask_c < MIN_PRICE_C:
            log.warning(f"[settle-comm] SKIP {ticker}: model says YES locked "
                        f"({parsed['symbol']}=${quote:.2f}≥${threshold:.2f}) but market "
                        f"prices YES at {yes_ask_c}¢ — listen to market.")
            return None
        elif quote <= threshold - margin and no_ask_c < MIN_PRICE_C:
            log.warning(f"[settle-comm] SKIP {ticker}: model says NO locked "
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
        if not _nymex_intraday():
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
                log.debug(f"[settle-comm] eval {ticker} failed: {e}")
        opps.sort(key=lambda x: x["edge_c"], reverse=True)
        return opps

    def execute(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_c = opp["price_c"]
        edge_c  = opp["edge_c"]

        # Tier-aware sizing
        price = price_c / 100
        tier_cap_dollars = self.risk.balance * self.risk.effective_max_position_pct()
        target_dollars = min(MAX_TRADE_DOLLARS, tier_cap_dollars)
        if target_dollars < MIN_TRADE_DOLLARS:
            log.info(f"[settle-comm] Skip {ticker}: tier cap ${tier_cap_dollars:.2f} below floor")
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
            log.info(f"[settle-comm] Skip {ticker}: {reason}")
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
            self._positions[ticker] = CommodityPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, spot_at_entry=opp["spot"],
                threshold=opp["threshold"],
            )
            log.info(
                f"[settle-comm] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢  "
                f"edge={edge_c:.1f}¢  {opp['reason']}"
            )
            return True
        except Exception as e:
            log.error(f"[settle-comm] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            self._positions.pop(ticker, None)
            return False
