"""
Verified Settlement strategy (weather v1).

Thesis: A few hours before a weather market resolves, the day's actual high/low
is often already locked in by observed temperatures so far. Open-Meteo gives us
hourly observed temps. If the market still prices an outcome at extreme odds
against what the observations already show, that's a real, structural edge.

This is fundamentally different from forecast-based weather trading:
  - Forecast strategy bets on model accuracy (model risk dominates).
  - Settlement strategy bets on physical certainty (the temp already happened).

We never override physics. If observed max today is already 92°F, and the
threshold is "> 89°F", the market WILL resolve YES regardless of what happens
in the remaining hours. The only risk is observation-station data corrections,
which are very rare.

Rules:
  1. Market resolves in 0.5 - 8 hours (close enough that observed data is decisive).
  2. We have hourly observed temp from start-of-day in city's local TZ.
  3. For HIGH market with threshold T:
       observed_max ≥ T + 1°F → buy YES (locked YES) at price ≤ 90¢ if edge ≥ 10¢
       remaining hourly forecast max + observed_max < T - 2°F → buy NO at ≤ 90¢
  4. For LOW market with threshold T:
       observed_min ≤ T - 1°F → buy YES at price ≤ 90¢
       remaining hourly forecast min > T + 2°F → buy NO at ≤ 90¢
  5. Min trade $20 (fees become noise).
  6. Position fully resolves within hours — single exit at settlement.

Why this works: We're not predicting, we're observing. The "edge" comes from
the market not having fully priced in observations that haven't yet been
captured by sportsbook-style automated price feeds.
"""

import time
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx

from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log

# Reuse the same series→city mapping the existing weather strategy uses.
from strategies.weather import SERIES, _parse_ticker, _hours_until_resolution


# ── Tunables ───────────────────────────────────────────────────────────────────
RESOLVE_WINDOW_MIN_H = 0.5       # don't bother less than 30 min from settle (slippage)
RESOLVE_WINDOW_MAX_H = 8.0       # past 8h out, observed data isn't yet decisive
MIN_EDGE_C           = 10        # 10¢ per contract minimum edge AFTER fees
MIN_TRADE_DOLLARS    = 20.0
MAX_TRADE_DOLLARS    = 35.0      # cap any single bet; this is concentrated risk
MAX_PRICE_C          = 90        # don't pay >90¢ even when verified — limited upside
SCAN_INTERVAL_S      = 60        # tight loop — observed data updates hourly
OBS_CACHE_TTL_S      = 600       # re-fetch observations every 10 min


@dataclass
class SettlementPosition:
    ticker:        str
    side:          str          # "yes" or "no"
    contracts:     int
    entry_price_c: int
    observed_temp: float        # what we saw at entry
    threshold_f:   float
    series:        str


def _kalshi_fee_cents(price_c: int, contracts: int) -> float:
    """Kalshi fee in cents: 0.07 * price * (1 - price) per contract."""
    p = price_c / 100
    return 0.07 * p * (1 - p) * 100 * contracts


# ── Open-Meteo observed-temperature fetch ──────────────────────────────────────
_obs_cache: dict[tuple, tuple[float, dict]] = {}   # (lat,lon,date) -> (ts, payload)


def _fetch_observations(lat: float, lon: float, target_date: str, tz_name: str) -> Optional[dict]:
    """Fetch hourly temps for `target_date` in the city's local TZ.
    Returns dict with observed (past hours) and forecast (future hours) max/min."""
    key = (round(lat, 3), round(lon, 3), target_date)
    cached = _obs_cache.get(key)
    if cached and time.time() - cached[0] < OBS_CACHE_TTL_S:
        return cached[1]

    for attempt in range(3):
        try:
            r = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":         lat,
                    "longitude":        lon,
                    "hourly":           "temperature_2m",
                    "temperature_unit": "fahrenheit",
                    "timezone":         tz_name,
                    "start_date":       target_date,
                    "end_date":         target_date,
                    "bias_correction":  "true",
                },
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            times = data["hourly"]["time"]
            temps = data["hourly"]["temperature_2m"]
            if not times or not temps:
                return None

            # Split into observed (past) vs forecast (future) using the LOCAL hour.
            try:
                from zoneinfo import ZoneInfo
            except ImportError:
                from backports.zoneinfo import ZoneInfo
            local_now = datetime.now(ZoneInfo(tz_name))
            observed = [(t, v) for t, v in zip(times, temps)
                        if datetime.fromisoformat(t) <= local_now and v is not None]
            forecast = [(t, v) for t, v in zip(times, temps)
                        if datetime.fromisoformat(t) > local_now and v is not None]

            result = {
                "observed_max":  max((v for _, v in observed), default=None),
                "observed_min":  min((v for _, v in observed), default=None),
                "forecast_max":  max((v for _, v in forecast), default=None),
                "forecast_min":  min((v for _, v in forecast), default=None),
                "hours_seen":    len(observed),
                "hours_remaining": len(forecast),
            }
            _obs_cache[key] = (time.time(), result)
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            log.warning(f"[settle] Obs fetch failed for {target_date}: {e}")
            return None


# ── Strategy ───────────────────────────────────────────────────────────────────
class SettlementStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk   = risk
        self._last_scan: float = 0.0
        self._positions: dict[str, SettlementPosition] = {}
        self._lock = threading.Lock()

    # ── Entry decision ─────────────────────────────────────────────────────────
    def _evaluate(self, m: dict) -> Optional[dict]:
        """Return an opp dict if this market has a verified edge, else None."""
        ticker = m["ticker"]
        series, target_date, threshold_f = _parse_ticker(ticker)
        if not series or not target_date or threshold_f is None:
            return None
        cfg = SERIES.get(series)
        if not cfg:
            return None

        # Time-to-resolve window
        h_left = _hours_until_resolution(target_date, cfg["tz"])
        if not (RESOLVE_WINDOW_MIN_H <= h_left <= RESOLVE_WINDOW_MAX_H):
            return None

        obs = _fetch_observations(cfg["lat"], cfg["lon"], target_date, cfg["tz"])
        if not obs:
            return None

        kind = cfg["kind"]
        yes_ask_c = int(round(float(m.get("yes_ask_dollars") or 0) * 100))
        no_ask_c  = int(round(float(m.get("no_ask_dollars")  or 0) * 100))
        if yes_ask_c == 0 or no_ask_c == 0:
            return None

        side = price_c = reason = observed_temp = None

        if kind == "high":
            obs_max = obs["observed_max"] or -999
            forecast_remaining_max = obs["forecast_max"] or obs_max
            # YES is locked: observed max already exceeds threshold by ≥1°F
            if obs_max >= threshold_f + 1.0 and yes_ask_c <= MAX_PRICE_C:
                side, price_c, observed_temp = "yes", yes_ask_c, obs_max
                reason = f"observed_max={obs_max:.1f}°F ≥ threshold={threshold_f}°F + 1"
            # NO is locked: both observed and rest-of-day forecast won't reach threshold
            elif (max(obs_max, forecast_remaining_max) < threshold_f - 2.0
                  and no_ask_c <= MAX_PRICE_C):
                side, price_c, observed_temp = "no", no_ask_c, max(obs_max, forecast_remaining_max)
                reason = f"max(obs,fcst)={max(obs_max, forecast_remaining_max):.1f}°F < threshold={threshold_f}°F - 2"

        else:   # low
            obs_min = obs["observed_min"]
            forecast_remaining_min = obs["forecast_min"]
            if obs_min is None:
                return None
            if forecast_remaining_min is None:
                forecast_remaining_min = obs_min
            # YES locked: observed min already below threshold by ≥1°F
            if obs_min <= threshold_f - 1.0 and yes_ask_c <= MAX_PRICE_C:
                side, price_c, observed_temp = "yes", yes_ask_c, obs_min
                reason = f"observed_min={obs_min:.1f}°F ≤ threshold={threshold_f}°F - 1"
            # NO locked: observed AND remaining forecast both stay above threshold + 2
            elif (min(obs_min, forecast_remaining_min) > threshold_f + 2.0
                  and no_ask_c <= MAX_PRICE_C):
                side, price_c, observed_temp = "no", no_ask_c, min(obs_min, forecast_remaining_min)
                reason = f"min(obs,fcst)={min(obs_min, forecast_remaining_min):.1f}°F > threshold={threshold_f}°F + 2"

        if side is None:
            return None

        # Edge after fees: $1 - price_c paid - fee_paid
        contracts_for_fee = 1
        fee_c = _kalshi_fee_cents(price_c, contracts_for_fee)
        # If verified outcome will resolve to $1, our profit per contract is
        # (100 - price_c - fee_c). We require ≥ MIN_EDGE_C.
        edge_c = (100 - price_c) - fee_c
        if edge_c < MIN_EDGE_C:
            return None

        return {
            "ticker":       ticker,
            "series":       series,
            "side":         side,
            "price_c":      price_c,
            "threshold_f":  threshold_f,
            "observed":     observed_temp,
            "hours_left":   h_left,
            "edge_c":       edge_c,
            "reason":       reason,
            "title":        m.get("title", "")[:60],
        }

    # ── Public scan/execute ────────────────────────────────────────────────────
    def scan(self) -> list[dict]:
        now = time.time()
        with self._lock:
            if now - self._last_scan < SCAN_INTERVAL_S:
                return []
            self._last_scan = now

        markets = [m for m in get_liquid_markets(self.kalshi, min_volume=0)
                   if m["ticker"].split("-")[0] in SERIES]

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
                log.debug(f"[settle] eval {ticker} failed: {e}")
        opps.sort(key=lambda x: x["edge_c"], reverse=True)
        return opps

    def execute(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_c = opp["price_c"]
        edge_c  = opp["edge_c"]

        # Position size: hit the $5 floor in approve_trade by spending at least MIN_TRADE_DOLLARS,
        # cap at MAX_TRADE_DOLLARS for concentration control.
        price_dollars = price_c / 100
        contracts = max(int(MIN_TRADE_DOLLARS / price_dollars),
                        int(MAX_TRADE_DOLLARS / price_dollars))
        contracts = min(contracts, int(MAX_TRADE_DOLLARS / price_dollars))
        if contracts <= 0:
            return False

        with self._lock:
            if ticker in self._positions:
                return False
            self._positions[ticker] = None  # reserve slot

        edge_dollars_total = (edge_c / 100) * contracts
        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge_dollars_total)
        if not ok:
            log.info(f"[settle] Skip {ticker}: {reason}")
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
            self._positions[ticker] = SettlementPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, observed_temp=opp["observed"],
                threshold_f=opp["threshold_f"], series=opp["series"],
            )
            log.info(
                f"[settle] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢  "
                f"edge={edge_c:.1f}¢  {opp['reason']}  ({opp['hours_left']:.1f}h to settle)"
            )
            return True
        except Exception as e:
            log.error(f"[settle] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            self._positions.pop(ticker, None)
            return False
