"""
Weather strategy — built specifically for Kalshi's temperature threshold markets.

Kalshi weather markets are binary threshold markets:
  KXHIGHNY-26MAY17-T90  → "Will NYC high temp be >90°F on May 17?"
  KXLOWTCHI-26MAY17-T64 → "Will Chicago low temp be >64°F on May 17?"

Multiple thresholds exist per city per day. This gives us the "3-bin" equivalent
from @theparuchh's Polymarket strategy: scan all thresholds for a city/date and
bet on every one where our model finds edge.

Key upgrades vs v1:
  1. Targets Kalshi series directly — no guessing from market titles
  2. Correct airport/observatory station coordinates (not city centers)
  3. Three models (ICON, GFS, ECMWF) with bias_correction=true
  4. Entry window: 18–30 hours before resolution
  5. Skip if 3 models disagree by >3°C
  6. Normal-CDF probability model using model spread as uncertainty
  7. Price filters: skip if YES ask > 45¢ (market already priced it in)
"""

import httpx
import math
import re
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.logger import log


# ── Kalshi weather series → station info ───────────────────────────────────────
# kind: "high" = daily max temp, "low" = daily min temp
# All US markets resolve in °F
SERIES = {
    # ── High temperature ──────────────────────────────────────────────────────
    "KXHIGHNY":    {"city": "NYC",         "lat": 40.7772,  "lon": -73.8726,  "station": "KLGA", "tz": "America/New_York",    "kind": "high"},
    "KXHIGHCHI":   {"city": "Chicago",     "lat": 41.9742,  "lon": -87.9073,  "station": "KORD", "tz": "America/Chicago",     "kind": "high"},
    "KXHIGHMIA":   {"city": "Miami",       "lat": 25.7959,  "lon": -80.2870,  "station": "KMIA", "tz": "America/New_York",    "kind": "high"},
    "KXHIGHLAX":   {"city": "LA",          "lat": 33.9425,  "lon": -118.4081, "station": "KLAX", "tz": "America/Los_Angeles", "kind": "high"},
    "KXHIGHDEN":   {"city": "Denver",      "lat": 39.8561,  "lon": -104.6737, "station": "KDEN", "tz": "America/Denver",      "kind": "high"},
    "KXHIGHTATL":  {"city": "Atlanta",     "lat": 33.6407,  "lon": -84.4277,  "station": "KATL", "tz": "America/New_York",    "kind": "high"},
    "KXHIGHTBOS":  {"city": "Boston",      "lat": 42.3629,  "lon": -71.0064,  "station": "KBOS", "tz": "America/New_York",    "kind": "high"},
    "KXHIGHTDC":   {"city": "DC",          "lat": 38.9531,  "lon": -77.4565,  "station": "KIAD", "tz": "America/New_York",    "kind": "high"},
    "KXHIGHTHOU":  {"city": "Houston",     "lat": 29.9902,  "lon": -95.3368,  "station": "KIAH", "tz": "America/Chicago",     "kind": "high"},
    "KXHIGHTDAL":  {"city": "Dallas",      "lat": 32.8471,  "lon": -96.8518,  "station": "KDAL", "tz": "America/Chicago",     "kind": "high"},
    "KXHIGHTPHX":  {"city": "Phoenix",     "lat": 33.4373,  "lon": -112.0078, "station": "KPHX", "tz": "America/Phoenix",     "kind": "high"},
    "KXHIGHTSEA":  {"city": "Seattle",     "lat": 47.4502,  "lon": -122.3088, "station": "KSEA", "tz": "America/Los_Angeles", "kind": "high"},
    "KXHIGHTSFO":  {"city": "SF",          "lat": 37.6213,  "lon": -122.3790, "station": "KSFO", "tz": "America/Los_Angeles", "kind": "high"},
    "KXHIGHPHIL":  {"city": "Philly",      "lat": 39.8721,  "lon": -75.2411,  "station": "KPHL", "tz": "America/New_York",    "kind": "high"},
    "KXHIGHAUS":   {"city": "Austin",      "lat": 30.1975,  "lon": -97.6664,  "station": "KAUS", "tz": "America/Chicago",     "kind": "high"},
    "KXHIGHTNOLA": {"city": "NOLA",        "lat": 29.9942,  "lon": -90.2580,  "station": "KMSY", "tz": "America/Chicago",     "kind": "high"},
    "KXHIGHTMIN":  {"city": "Minneapolis", "lat": 44.8848,  "lon": -93.2223,  "station": "KMSP", "tz": "America/Chicago",     "kind": "high"},
    "KXHIGHTOKC":  {"city": "OKC",         "lat": 35.3931,  "lon": -97.6007,  "station": "KOKC", "tz": "America/Chicago",     "kind": "high"},
    "KXHIGHTLV":   {"city": "Las Vegas",   "lat": 36.0840,  "lon": -115.1537, "station": "KLAS", "tz": "America/Los_Angeles", "kind": "high"},
    "KXHIGHTSATX": {"city": "San Antonio", "lat": 29.5337,  "lon": -98.4698,  "station": "KSAT", "tz": "America/Chicago",     "kind": "high"},
    # ── Low temperature ───────────────────────────────────────────────────────
    "KXLOWTNYC":   {"city": "NYC",         "lat": 40.7772,  "lon": -73.8726,  "station": "KLGA", "tz": "America/New_York",    "kind": "low"},
    "KXLOWTCHI":   {"city": "Chicago",     "lat": 41.9742,  "lon": -87.9073,  "station": "KORD", "tz": "America/Chicago",     "kind": "low"},
    "KXLOWTMIA":   {"city": "Miami",       "lat": 25.7959,  "lon": -80.2870,  "station": "KMIA", "tz": "America/New_York",    "kind": "low"},
    "KXLOWTLAX":   {"city": "LA",          "lat": 33.9425,  "lon": -118.4081, "station": "KLAX", "tz": "America/Los_Angeles", "kind": "low"},
    "KXLOWTDEN":   {"city": "Denver",      "lat": 39.8561,  "lon": -104.6737, "station": "KDEN", "tz": "America/Denver",      "kind": "low"},
    "KXLOWTATL":   {"city": "Atlanta",     "lat": 33.6407,  "lon": -84.4277,  "station": "KATL", "tz": "America/New_York",    "kind": "low"},
    "KXLOWTBOS":   {"city": "Boston",      "lat": 42.3629,  "lon": -71.0064,  "station": "KBOS", "tz": "America/New_York",    "kind": "low"},
    "KXLOWTDC":    {"city": "DC",          "lat": 38.9531,  "lon": -77.4565,  "station": "KIAD", "tz": "America/New_York",    "kind": "low"},
    "KXLOWTHOU":   {"city": "Houston",     "lat": 29.9902,  "lon": -95.3368,  "station": "KIAH", "tz": "America/Chicago",     "kind": "low"},
    "KXLOWTDAL":   {"city": "Dallas",      "lat": 32.8471,  "lon": -96.8518,  "station": "KDAL", "tz": "America/Chicago",     "kind": "low"},
    "KXLOWTPHX":   {"city": "Phoenix",     "lat": 33.4373,  "lon": -112.0078, "station": "KPHX", "tz": "America/Phoenix",     "kind": "low"},
    "KXLOWTSEA":   {"city": "Seattle",     "lat": 47.4502,  "lon": -122.3088, "station": "KSEA", "tz": "America/Los_Angeles", "kind": "low"},
    "KXLOWTSFO":   {"city": "SF",          "lat": 37.6213,  "lon": -122.3790, "station": "KSFO", "tz": "America/Los_Angeles", "kind": "low"},
    "KXLOWTPHIL":  {"city": "Philly",      "lat": 39.8721,  "lon": -75.2411,  "station": "KPHL", "tz": "America/New_York",    "kind": "low"},
    "KXLOWTAUS":   {"city": "Austin",      "lat": 30.1975,  "lon": -97.6664,  "station": "KAUS", "tz": "America/Chicago",     "kind": "low"},
    "KXLOWTNOLA":  {"city": "NOLA",        "lat": 29.9942,  "lon": -90.2580,  "station": "KMSY", "tz": "America/Chicago",     "kind": "low"},
    "KXLOWTMIN":   {"city": "Minneapolis", "lat": 44.8848,  "lon": -93.2223,  "station": "KMSP", "tz": "America/Chicago",     "kind": "low"},
    "KXLOWTOKC":   {"city": "OKC",         "lat": 35.3931,  "lon": -97.6007,  "station": "KOKC", "tz": "America/Chicago",     "kind": "low"},
    "KXLOWTLV":    {"city": "Las Vegas",   "lat": 36.0840,  "lon": -115.1537, "station": "KLAS", "tz": "America/Los_Angeles", "kind": "low"},
    "KXLOWTSATX":  {"city": "San Antonio", "lat": 29.5337,  "lon": -98.4698,  "station": "KSAT", "tz": "America/Chicago",     "kind": "low"},
    "KXLOWTATL":   {"city": "Atlanta",     "lat": 33.6407,  "lon": -84.4277,  "station": "KATL", "tz": "America/New_York",    "kind": "low"},
}

# Model bias corrections (°F) — cities where models systematically miss
CITY_BIAS_F = {
    # Add if you find a city consistently losing; start with 0 and track results
}

MIN_EDGE       = 0.10   # per-contract edge required — high bar, weather is hard
MAX_YES_ASK    = 0.85   # only skip near-resolved markets (>85¢ already priced in)
MIN_YES_ASK    = 0.05   # skip ultra-cheap lottery tickets (≤5¢) — most go to zero
MODEL_AGREE_C  = 3.0    # skip if 3 models spread > 3°C
PAIR_AGREE_F   = 1.8    # 2 models within 1°C (=1.8°F) → drop 3rd as outlier
WIN_MIN_H      = 18.0   # earliest entry: 18h before resolution
WIN_MAX_H      = 30.0   # 18–30h sweet spot, before market reprices on fresh models
SCAN_INTERVAL  = 90     # scan every 90 seconds — weather edges are time-sensitive
EXIT_EDGE_MIN  = 0.02   # exit when edge collapses to 2¢
EXIT_HOURS     = 4 + 7/60   # always exit 4h 7min before resolution
# REDESIGN (2026-05-18): The "3-bin" idea worked on Polymarket range markets where
# bins are mutually exclusive — exactly one resolves. Kalshi nested thresholds are
# DIFFERENT: each threshold is an independent bet. Buying 4 cheap extremes = spray
# and pray, ~80% hit zero. Back to ONE threshold per city/date, chosen as the
# threshold closest to the consensus forecast where the per-contract edge clears
# the 10¢ bar. This is the high-conviction setup.
MAX_PER_CITY_DATE = 1


# ── Forecast cache (avoid hammering Open-Meteo for same city/date) ─────────────
_forecast_cache: dict = {}   # (lat, lon, date) -> forecast dict
_forecast_ts:   dict = {}    # (lat, lon, date) -> fetch timestamp
FORECAST_TTL = 1800          # re-fetch every 30 min


def _fetch_forecast(lat: float, lon: float, target_date: str, tz_name: str) -> object:
    key = (round(lat, 3), round(lon, 3), target_date)
    if key in _forecast_cache and time.time() - _forecast_ts.get(key, 0) < FORECAST_TTL:
        return _forecast_cache[key]

    for attempt in range(3):   # retry up to 3 times on SSL/timeout failures
        try:
            r = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":         lat,
                    "longitude":        lon,
                    "models":           "icon_seamless,gfs_seamless,ecmwf_ifs025",
                    "hourly":           "temperature_2m",
                    "bias_correction":  "true",       # ← the parameter most bots miss
                    "temperature_unit": "fahrenheit",  # get °F directly
                    "start_date":       target_date,
                    "end_date":         target_date,
                    "timezone":         tz_name,       # local timezone so min/max is local day
                },
                timeout=20,
            )
            r.raise_for_status()
            hourly = r.json().get("hourly", {})

            icon_h  = [t for t in hourly.get("temperature_2m_icon_seamless",  []) if t is not None]
            gfs_h   = [t for t in hourly.get("temperature_2m_gfs_seamless",   []) if t is not None]
            ecmwf_h = [t for t in hourly.get("temperature_2m_ecmwf_ifs025",   []) if t is not None]

            if not icon_h or not gfs_h or not ecmwf_h:
                return None

            result = {
                # Daily high (max) per model
                "icon_high":  max(icon_h),  "gfs_high":  max(gfs_h),  "ecmwf_high":  max(ecmwf_h),
                # Daily low (min) per model
                "icon_low":   min(icon_h),  "gfs_low":   min(gfs_h),  "ecmwf_low":   min(ecmwf_h),
            }
            for kind in ("high", "low"):
                vals = [result[f"icon_{kind}"], result[f"gfs_{kind}"], result[f"ecmwf_{kind}"]]
                spread = max(vals) - min(vals)
                result[f"{kind}_spread"] = round(spread, 2)
                result[f"{kind}_agree"]  = spread <= MODEL_AGREE_C * 9/5   # all 3 within 3°C

                # Paruch V3: if any 2 models agree within 1°C, drop the 3rd as outlier
                # and use the pair's mean as consensus. Otherwise mean all three.
                pairs = [(0,1,2), (0,2,1), (1,2,0)]   # (a, b, outlier_idx)
                best_pair = None
                for i, j, k in pairs:
                    d = abs(vals[i] - vals[j])
                    if d <= PAIR_AGREE_F and (best_pair is None or d < best_pair[0]):
                        best_pair = (d, i, j, k)
                if best_pair:
                    _, i, j, k = best_pair
                    result[f"{kind}_consensus"]  = round((vals[i] + vals[j]) / 2, 1)
                    result[f"{kind}_used"]       = 2
                    result[f"{kind}_outlier"]    = round(vals[k], 1)
                else:
                    result[f"{kind}_consensus"]  = round(sum(vals) / 3, 1)
                    result[f"{kind}_used"]       = 3
                    result[f"{kind}_outlier"]    = None

            _forecast_cache[key] = result
            _forecast_ts[key]    = time.time()
            return result

        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))   # 2s, 4s back-off
                continue
            log.warning(f"[weather] Forecast failed ({lat:.2f},{lon:.2f} {target_date}): {e}")
            return None


# ── Probability calculation ─────────────────────────────────────────────────────

def _prob_above(forecast_temp: float, threshold: float, spread_f: float) -> float:
    """P(actual_temp > threshold) using normal CDF. Spread = model uncertainty (°F)."""
    sigma = max(1.5, spread_f / 2)   # 1.5°F floor on uncertainty
    z = (forecast_temp - threshold) / sigma
    return round(max(0.02, min(0.98, 0.5 * (1 + math.erf(z / math.sqrt(2))))), 3)


# ── Ticker parsing ──────────────────────────────────────────────────────────────

def _parse_ticker(ticker: str):
    """
    Parse series, date, threshold from a Kalshi weather ticker.
    e.g. KXHIGHNY-26MAY17-T90 → series='KXHIGHNY', date='2026-05-17', threshold=90.0
    """
    parts = ticker.split("-")
    if len(parts) < 3:
        return None, None, None

    series = parts[0]
    date_str = parts[1]    # e.g. '26MAY17'
    thresh_str = parts[2]  # e.g. 'T90' or 'T88.5'

    # Parse date
    try:
        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,  "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10,"NOV": 11, "DEC": 12
        }
        m = re.match(r'(\d{2})([A-Z]{3})(\d{2})', date_str)
        if not m:
            return series, None, None
        year  = 2000 + int(m.group(1))
        month = month_map.get(m.group(2).upper())
        day   = int(m.group(3))
        if not month:
            return series, None, None
        date = f"{year:04d}-{month:02d}-{day:02d}"
    except Exception:
        return series, None, None

    # Parse threshold
    try:
        threshold = float(thresh_str.lstrip("T"))
    except Exception:
        return series, date, None

    return series, date, threshold


# ── Entry window check ──────────────────────────────────────────────────────────

def _hours_until_resolution(target_date: str, tz_name: str) -> float:
    """Hours until end of the target day in the city's local timezone."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        try:
            from backports.zoneinfo import ZoneInfo
        except ImportError:
            return 24.0   # can't check, assume in window
    try:
        local_tz   = ZoneInfo(tz_name)
        resolve_dt = datetime.strptime(target_date, "%Y-%m-%d").replace(
            hour=23, minute=59, tzinfo=local_tz
        )
        return (resolve_dt - datetime.now(local_tz)).total_seconds() / 3600
    except Exception:
        return 24.0


# ── Position tracking ───────────────────────────────────────────────────────────

@dataclass
class WeatherPosition:
    ticker:        str
    series:        str
    side:          str
    contracts:     int
    entry_price_c: int
    model_prob:    float
    target_date:   str
    threshold_f:   float
    kind:          str   # "high" or "low"
    tz_name:       str
    lat:           float
    lon:           float


# ── Strategy ────────────────────────────────────────────────────────────────────

class WeatherStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk   = risk
        self._last_scan:  float = 0
        self._positions:  dict[str, WeatherPosition] = {}
        self._last_entry: dict[str, float] = {}
        self._lock = threading.Lock()   # prevents concurrent double-entry

    # ── Exit management ────────────────────────────────────────────────────────

    def check_exits(self):
        # Rebuild in-memory tracking from Kalshi each call so exit logic also runs
        # on positions opened in previous bot sessions (otherwise they sit forever
        # because self._positions is in-memory and resets on every restart).
        self._rehydrate_positions()
        for ticker, pos in list(self._positions.items()):
            if pos is None:   # reserved slot for a pending entry
                continue
            reason = self._should_exit(pos)
            if reason:
                self._exit(pos, reason)

    def _rehydrate_positions(self):
        """Reconstruct WeatherPosition objects from Kalshi's current holdings
        for every weather ticker we hold but aren't already tracking. Only the
        ticker + side + contracts come from Kalshi; everything else is parsed
        from the ticker itself (date, threshold, city → tz/lat/lon)."""
        try:
            r = self.kalshi._get("/portfolio/positions", {"limit": 200})
            market_pos = r.get("market_positions", [])
        except Exception:
            return

        for p in market_pos:
            fp = float(p.get("position_fp") or 0)
            if fp == 0:
                continue
            ticker = p.get("ticker") or p.get("market_ticker", "")
            if not ticker or ticker in self._positions and self._positions[ticker] is not None:
                continue

            series, target_date, threshold_f = _parse_ticker(ticker)
            if not series or not target_date or threshold_f is None:
                continue
            cfg = SERIES.get(series)
            if not cfg:
                continue

            # Skip past-resolution markets — they'll settle on their own and the
            # exit logic just hammers a closed market with 400s. Kalshi handles
            # settlement; we don't need to "sell" anything.
            if _hours_until_resolution(target_date, cfg["tz"]) < 0:
                continue

            self._positions[ticker] = WeatherPosition(
                ticker        = ticker,
                series        = series,
                side          = "yes" if fp > 0 else "no",
                contracts     = int(abs(fp)),
                entry_price_c = 0,         # unknown without fill history; not needed for time/edge exits
                model_prob    = 0.0,       # only used by edge-collapse exit; safe default skips that path
                target_date   = target_date,
                threshold_f   = threshold_f,
                kind          = cfg["kind"],
                tz_name       = cfg["tz"],
                lat           = cfg["lat"],
                lon           = cfg["lon"],
            )

    def _should_exit(self, pos: WeatherPosition) -> object:
        hours_left = _hours_until_resolution(pos.target_date, pos.tz_name)
        if hours_left < 0:
            return "past resolution"
        if hours_left < EXIT_HOURS:
            return f"{hours_left:.1f}h to resolution"

        # Re-check edge — exit if model and market have converged
        fc = _fetch_forecast(pos.lat, pos.lon, pos.target_date, pos.tz_name)
        if fc and fc.get(f"{pos.kind}_agree"):
            consensus = fc[f"{pos.kind}_consensus"]
            bias      = CITY_BIAS_F.get(pos.series, 0.0)
            spread    = fc[f"{pos.kind}_spread"]
            model_prob = _prob_above(consensus + bias, pos.threshold_f, spread)
            try:
                m = self.kalshi.get_market(pos.ticker).get("market", {})
                yes_ask = float(m.get("yes_ask_dollars") or 0)
                yes_bid = float(m.get("yes_bid_dollars") or 0)
                kalshi_prob = (yes_ask + yes_bid) / 2 if yes_ask and yes_bid else yes_ask
                edge = abs(model_prob - kalshi_prob)
                if edge < EXIT_EDGE_MIN:
                    return f"edge collapsed to {edge*100:.1f}¢"
            except Exception:
                pass
        return None

    def _exit(self, pos: WeatherPosition, reason: str):
        # Kalshi rejects market orders — use a limit at the current bid for our side,
        # which fills against the resting bid immediately (effectively a market sell).
        try:
            m = self.kalshi.get_market(pos.ticker).get("market", {})
            if pos.side == "yes":
                bid_c = int(round(float(m.get("yes_bid_dollars") or 0) * 100))
            else:
                bid_c = int(round(float(m.get("no_bid_dollars")  or 0) * 100))
            bid_c = max(1, min(99, bid_c))   # Kalshi requires 1-99¢

            self.kalshi.place_order(
                ticker=pos.ticker, side=pos.side, action="sell",
                count=pos.contracts, order_type="limit",
                yes_price=bid_c if pos.side == "yes" else None,
                no_price =bid_c if pos.side == "no"  else None,
            )
            self.risk.record_close(pos.ticker, pos.contracts)
            log.info(f"[weather] EXIT {pos.ticker} {pos.side.upper()} x{pos.contracts} @ {bid_c}¢  {reason}")
        except Exception as e:
            log.error(f"[weather] Exit failed {pos.ticker}: {e}")
        finally:
            self._positions.pop(pos.ticker, None)

    # ── Scan ────────────────────────────────────────────────────────────────────

    def scan(self) -> list[dict]:
        self.check_exits()

        now = time.time()
        with self._lock:
            if now - self._last_scan < SCAN_INTERVAL:
                return []
            self._last_scan = now

        opps = []

        for series_ticker, info in SERIES.items():
            # Fetch active markets for this series
            try:
                r = self.kalshi._get("/markets", {"limit": 20, "series_ticker": series_ticker})
                markets = [m for m in r.get("markets", []) if m.get("status") == "active"]
            except Exception:
                continue

            if not markets:
                continue

            # Group by date — fetch forecast once per city/date
            by_date: dict = {}
            for m in markets:
                _, date, threshold = _parse_ticker(m["ticker"])
                if date and threshold is not None:
                    by_date.setdefault(date, []).append((m, threshold))

            for date, mkt_list in by_date.items():
                # ── Entry window check ─────────────────────────────────────
                hours_left = _hours_until_resolution(date, info["tz"])
                if not (WIN_MIN_H <= hours_left <= WIN_MAX_H):
                    continue

                # ── Fetch 3-model forecast ─────────────────────────────────
                fc = _fetch_forecast(info["lat"], info["lon"], date, info["tz"])
                if not fc:
                    continue

                kind = info["kind"]   # "high" or "low"

                # ── Model agreement check ──────────────────────────────────
                if not fc[f"{kind}_agree"]:
                    log.info(
                        f"[weather] Skip {series_ticker} {date} — models disagree "
                        f"{fc[f'{kind}_spread']:.1f}°F  "
                        f"(ICON={fc[f'icon_{kind}']:.1f} GFS={fc[f'gfs_{kind}']:.1f} "
                        f"ECMWF={fc[f'ecmwf_{kind}']:.1f})"
                    )
                    continue

                consensus = fc[f"{kind}_consensus"]
                spread    = fc[f"{kind}_spread"]
                bias      = CITY_BIAS_F.get(series_ticker, 0.0)
                forecast  = consensus + bias

                # ── Evaluate each threshold market ─────────────────────────
                for m, threshold in sorted(mkt_list, key=lambda x: x[1]):
                    ticker = m["ticker"]

                    if ticker in self._positions:
                        continue
                    if now - self._last_entry.get(ticker, 0) < 3600:
                        continue

                    # Per-city-date cap: all thresholds resolve on the same temperature,
                    # so multiple positions per city/date are fully correlated — cap at 1.
                    city_date_key = f"{series_ticker}-{date}"
                    city_date_open = sum(
                        1 for t in self._positions
                        if t and t.startswith(series_ticker) and f"-{date}-" in t
                    )
                    if city_date_open >= MAX_PER_CITY_DATE:
                        continue

                    yes_ask = float(m.get("yes_ask_dollars") or 0)
                    yes_bid = float(m.get("yes_bid_dollars") or 0)
                    no_ask  = float(m.get("no_ask_dollars")  or 0)

                    if not yes_ask or yes_ask < MIN_YES_ASK:
                        continue

                    model_prob = _prob_above(forecast, threshold, spread)

                    # YES edge: model says more likely than market prices
                    yes_edge = model_prob - yes_ask
                    # NO edge: model says less likely than market prices
                    no_edge  = (1 - model_prob) - no_ask if no_ask else 0

                    # Price filter: skip if market has already priced it in
                    if yes_edge >= MIN_EDGE and yes_ask <= MAX_YES_ASK:
                        side  = "yes"
                        price = yes_ask
                        edge  = yes_edge
                    elif no_edge >= MIN_EDGE and no_ask <= MAX_YES_ASK:
                        side  = "no"
                        price = no_ask
                        edge  = no_edge
                    else:
                        continue

                    opps.append({
                        "ticker":      ticker,
                        "series":      series_ticker,
                        "city":        info["city"],
                        "station":     info["station"],
                        "kind":        kind,
                        "threshold_f": threshold,
                        "date":        date,
                        "hours_left":  round(hours_left, 1),
                        "side":        side,
                        "price":       price,
                        "price_c":     int(price * 100),
                        "model_prob":  model_prob,
                        "kalshi_prob": round(yes_ask, 3),
                        "edge":        round(edge, 3),
                        "forecast_f":  round(forecast, 1),
                        "spread_f":    round(spread, 1),
                        "icon_f":      round(fc[f"icon_{kind}"], 1),
                        "gfs_f":       round(fc[f"gfs_{kind}"],  1),
                        "ecmwf_f":     round(fc[f"ecmwf_{kind}"],1),
                        "used_models": fc.get(f"{kind}_used", 3),
                        "tz_name":     info["tz"],
                        "lat":         info["lat"],
                        "lon":         info["lon"],
                    })

        opps.sort(key=lambda x: x["edge"], reverse=True)

        if opps:
            log.info(
                f"[weather] {len(opps)} edges found: "
                + ", ".join(
                    f"{o['city']} {o['kind']} {'>'+str(int(o['threshold_f']))+'°F'} "
                    f"{o['side'].upper()} {o['edge']*100:.0f}¢"
                    for o in opps[:4]
                )
            )

        return opps

    # ── Execute ─────────────────────────────────────────────────────────────────

    def execute(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_c = opp["price_c"]
        edge    = opp["edge"]

        with self._lock:
            if ticker in self._positions:
                return False
            # Also block if risk manager already shows an open position (e.g. after bot restart)
            if self.risk.open_positions.get(ticker, 0) > 0:
                self._positions[ticker] = None  # suppress future scans of this ticker
                return False
            # Reserve the slot immediately to block concurrent callers
            self._positions[ticker] = None  # type: ignore — filled below if order succeeds

        contracts = self.risk.kelly_contracts(price_c, opp["model_prob"], max_contracts=50)
        contracts = max(1, contracts)

        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[weather] Skipped {ticker}: {reason}")
            self._positions.pop(ticker, None)  # release the slot reservation
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
            self._positions[ticker] = WeatherPosition(
                ticker=ticker, series=opp["series"], side=side,
                contracts=contracts, entry_price_c=price_c,
                model_prob=opp["model_prob"],
                target_date=opp["date"], threshold_f=opp["threshold_f"],
                kind=opp["kind"], tz_name=opp["tz_name"],
                lat=opp["lat"], lon=opp["lon"],
            )
            used_note = f" [used {opp.get('used_models',3)}/3]" if opp.get('used_models',3) < 3 else ""
            log.info(
                f"[weather] ENTER {ticker}  {side.upper()} x{contracts} @ {price_c}¢  "
                f"model={opp['model_prob']*100:.0f}%  edge={edge*100:.0f}¢  "
                f"forecast={opp['forecast_f']}°F  threshold={opp['threshold_f']}°F  "
                f"ICON={opp['icon_f']} GFS={opp['gfs_f']} ECMWF={opp['ecmwf_f']}{used_note}  "
                f"spread={opp['spread_f']}°F  {opp['hours_left']}h left  [{opp['station']}]"
            )
            return True
        except Exception as e:
            log.error(f"[weather] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            self._positions.pop(ticker, None)  # release slot on failure
            return False
