"""
Verified Crypto Settlement — BTC and ETH hourly/daily price markets.

Thesis: With minutes left until resolution, the spot price of BTC/ETH from
a public exchange (Coinbase) is a near-perfect estimate of where the CF
Benchmarks RTI will settle. We only trade when the spot price is comfortably
on one side of the threshold AND the market still prices the locked-in
outcome below 92¢.

Ticker format (Kalshi):
  KXBTC-26MAY1917-T86249.99   → "Will BTC be > $86,249.99 at 5pm EDT May 19?"
  KXETH-26MAY1917-T2859.99    → "Will ETH be > $2,859.99 at 5pm EDT May 19?"

Resolution: simple average of CF Benchmarks RTI over 60 seconds before close.
Approximation: Coinbase spot is within ~0.05% of CF Benchmarks RTI in normal
markets — well inside our safety margin.

Safety margins (the amount spot must beat the threshold by):
  ≥30 minutes to resolve: 2.0% margin
  10-30 minutes:          1.0%
  3-10 minutes:           0.5%
  <3 minutes:             0.2%
"""

import re
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


# ── Tunables ───────────────────────────────────────────────────────────────────
SCAN_INTERVAL_S      = 30           # tight — crypto moves
MIN_EDGE_C           = 10           # 10¢ per contract minimum AFTER fees
MIN_TRADE_DOLLARS    = 20.0
MAX_TRADE_DOLLARS    = 40.0
MAX_PRICE_C          = 92           # don't pay >92¢ even verified
RESOLVE_MIN_S        = 60           # need ≥1 min — slippage cushion
RESOLVE_MAX_S        = 90 * 60      # 90 min out, spot drift is too risky
PRICE_CACHE_TTL_S    = 5            # refresh spot every 5s


# Pair currently tradeable on Kalshi (extend as new series appear)
SUPPORTED_SERIES = {
    "KXBTC":  "BTC-USD",
    "KXETH":  "ETH-USD",
    # Add KXSOL / KXXRP / etc. here as Kalshi adds them; just need the Coinbase pair.
}

# Month parsing for the YYMMMDDHH date format
_MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
           "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}


@dataclass
class CryptoPosition:
    ticker:        str
    side:          str
    contracts:     int
    entry_price_c: int
    spot_at_entry: float
    threshold:     float


def _kalshi_fee_cents(price_c: int, contracts: int) -> float:
    p = price_c / 100
    return 0.07 * p * (1 - p) * 100 * contracts


# ── Coinbase spot fetch (free, no auth, fast) ──────────────────────────────────
_spot_cache: dict[str, tuple[float, float]] = {}   # pair -> (ts, price)

def _fetch_spot(pair: str) -> Optional[float]:
    cached = _spot_cache.get(pair)
    if cached and time.time() - cached[0] < PRICE_CACHE_TTL_S:
        return cached[1]
    try:
        r = httpx.get(f"https://api.coinbase.com/v2/prices/{pair}/spot", timeout=8)
        r.raise_for_status()
        price = float(r.json()["data"]["amount"])
        _spot_cache[pair] = (time.time(), price)
        return price
    except Exception as e:
        log.debug(f"[settle-crypto] spot fetch failed {pair}: {e}")
        return None


# ── Ticker parsing ─────────────────────────────────────────────────────────────
def _parse_crypto_ticker(ticker: str) -> Optional[dict]:
    """
    KXBTC-26MAY1917-T86249.99 → {series, resolve_dt_utc, threshold, kind: 'T'}
    Returns None if not a parseable threshold market.
    """
    parts = ticker.split("-")
    if len(parts) != 3:
        return None
    series = parts[0]
    if series not in SUPPORTED_SERIES:
        return None
    date_str = parts[1]
    thr_str  = parts[2]
    # Only handle T (above) markets; skip B (between/range) for now
    if not thr_str.startswith("T"):
        return None
    try:
        threshold = float(thr_str[1:])
    except ValueError:
        return None
    # Date: YYMMMDDHH  (2-digit year, 3-letter month, day, hour-of-day)
    m = re.match(r"(\d{2})([A-Z]{3})(\d{2})(\d{2})$", date_str)
    if not m:
        return None
    yy, mon, dd, hh = m.groups()
    month = _MONTHS.get(mon)
    if not month:
        return None
    # Kalshi crypto markets resolve at the given hour, EDT/EST (US/Eastern).
    # 5pm EDT = 21:00 UTC. We approximate as UTC-4 (EDT); in standard time it
    # would be UTC-5. close_time on the market is authoritative — see fallback below.
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        resolve_dt = datetime(2000 + int(yy), month, int(dd), int(hh), 0, tzinfo=et)
        return {
            "series": series,
            "resolve_dt": resolve_dt.astimezone(timezone.utc),
            "threshold": threshold,
            "pair": SUPPORTED_SERIES[series],
        }
    except Exception:
        return None


def _safety_margin_pct(seconds_left: float) -> float:
    """Required percentage buffer between spot and threshold to count as locked."""
    if seconds_left >= 30 * 60:  return 0.020   # 2.0%
    if seconds_left >= 10 * 60:  return 0.010   # 1.0%
    if seconds_left >=  3 * 60:  return 0.005   # 0.5%
    return 0.002                                # 0.2%


# ── Strategy ───────────────────────────────────────────────────────────────────
class CryptoSettlementStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk   = risk
        self._last_scan = 0.0
        self._positions: dict[str, CryptoPosition] = {}
        self._lock = threading.Lock()

    def _evaluate(self, m: dict) -> Optional[dict]:
        ticker = m["ticker"]
        parsed = _parse_crypto_ticker(ticker)
        if not parsed:
            return None

        # Prefer Kalshi's authoritative close_time if present
        close_ts_str = m.get("close_time") or ""
        try:
            resolve_dt = datetime.fromisoformat(close_ts_str.replace("Z", "+00:00"))
        except Exception:
            resolve_dt = parsed["resolve_dt"]
        seconds_left = (resolve_dt - datetime.now(timezone.utc)).total_seconds()
        if not (RESOLVE_MIN_S <= seconds_left <= RESOLVE_MAX_S):
            return None

        spot = _fetch_spot(parsed["pair"])
        if spot is None:
            return None

        threshold = parsed["threshold"]
        margin_required = threshold * _safety_margin_pct(seconds_left)

        yes_ask_c = int(round(float(m.get("yes_ask_dollars") or 0) * 100))
        no_ask_c  = int(round(float(m.get("no_ask_dollars")  or 0) * 100))
        if yes_ask_c == 0 or no_ask_c == 0:
            return None

        side = price_c = reason = None

        # Spot comfortably above threshold → YES locked
        if spot >= threshold + margin_required and yes_ask_c <= MAX_PRICE_C:
            side, price_c = "yes", yes_ask_c
            reason = (f"spot=${spot:,.2f} ≥ threshold=${threshold:,.2f} + "
                      f"{margin_required:,.2f} ({seconds_left/60:.1f}min)")
        # Spot comfortably below threshold → NO locked
        elif spot <= threshold - margin_required and no_ask_c <= MAX_PRICE_C:
            side, price_c = "no", no_ask_c
            reason = (f"spot=${spot:,.2f} ≤ threshold=${threshold:,.2f} - "
                      f"{margin_required:,.2f} ({seconds_left/60:.1f}min)")

        if side is None:
            return None

        fee_c = _kalshi_fee_cents(price_c, 1)
        edge_c = (100 - price_c) - fee_c
        if edge_c < MIN_EDGE_C:
            return None

        return {
            "ticker":       ticker,
            "side":         side,
            "price_c":      price_c,
            "spot":         spot,
            "threshold":    threshold,
            "edge_c":       edge_c,
            "seconds_left": seconds_left,
            "reason":       reason,
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
                log.debug(f"[settle-crypto] eval {ticker} failed: {e}")
        opps.sort(key=lambda x: x["edge_c"], reverse=True)
        return opps

    def execute(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_c = opp["price_c"]
        edge_c  = opp["edge_c"]

        price = price_c / 100
        contracts = int(MAX_TRADE_DOLLARS / price)
        if contracts * price < MIN_TRADE_DOLLARS:
            contracts = max(contracts, int(MIN_TRADE_DOLLARS / price) + 1)
        if contracts <= 0:
            return False

        with self._lock:
            if ticker in self._positions:
                return False
            self._positions[ticker] = None

        edge_dollars_total = (edge_c / 100) * contracts
        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge_dollars_total)
        if not ok:
            log.info(f"[settle-crypto] Skip {ticker}: {reason}")
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
            self._positions[ticker] = CryptoPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, spot_at_entry=opp["spot"],
                threshold=opp["threshold"],
            )
            log.info(
                f"[settle-crypto] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢  "
                f"edge={edge_c:.1f}¢  {opp['reason']}"
            )
            return True
        except Exception as e:
            log.error(f"[settle-crypto] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            self._positions.pop(ticker, None)
            return False
