"""
Weather strategy: compare Kalshi's implied probability on weather markets
against free forecast data from Open-Meteo (no API key required).

Targets markets like:
  - "Will NYC reach 90°F on [date]?"
  - "Will it rain more than 1 inch in Chicago this week?"

Edge source: Kalshi prices often lag behind updated forecast models.
"""

import httpx
import re
import time
from datetime import datetime, timezone, timedelta
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.logger import log


# City name → (lat, lon)
CITY_COORDS = {
    "new york":    (40.71, -74.01),
    "nyc":         (40.71, -74.01),
    "los angeles": (34.05, -118.24),
    "chicago":     (41.88, -87.63),
    "houston":     (29.76, -95.37),
    "miami":       (25.77, -80.19),
    "dallas":      (32.78, -96.80),
    "phoenix":     (33.45, -112.07),
    "seattle":     (47.61, -122.33),
    "boston":      (42.36, -71.06),
    "atlanta":     (33.75, -84.39),
    "denver":      (39.74, -104.98),
}

MIN_EDGE = 0.06   # weather needs bigger edge — model uncertainty is high


def _get_forecast(lat: float, lon: float, target_date: str) -> object:
    """Fetch daily forecast from Open-Meteo for a given lat/lon and date (YYYY-MM-DD)."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "start_date": target_date,
            "end_date": target_date,
        }
        r = httpx.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        daily = data.get("daily", {})
        return {
            "high_f":        (daily.get("temperature_2m_max") or [None])[0],
            "low_f":         (daily.get("temperature_2m_min") or [None])[0],
            "precip_in":     (daily.get("precipitation_sum")  or [None])[0],
            "precip_prob":   (daily.get("precipitation_probability_max") or [None])[0],
        }
    except Exception as e:
        log.warning(f"[weather] Forecast fetch failed: {e}")
        return None


def _extract_city(title: str) -> object:
    title_lower = title.lower()
    for name, coords in CITY_COORDS.items():
        if name in title_lower:
            return coords
    return None


def _extract_temp_threshold(title: str) -> object:
    """Extract temperature threshold from title like 'reach 90°F' or 'above 85 degrees'."""
    m = re.search(r'(\d+)\s*(?:°f|degrees|°)', title, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _extract_date(title: str) -> object:
    """Try to extract a date from a market title. Returns YYYY-MM-DD or None."""
    today = datetime.now(timezone.utc)
    # Match patterns like "May 15", "June 3", "on 5/15"
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    m = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})', title, re.IGNORECASE)
    if m:
        month = months[m.group(1).lower()[:3]]
        day   = int(m.group(2))
        year  = today.year if month >= today.month else today.year + 1
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


EXIT_EDGE_THRESHOLD = 0.02   # exit when our edge shrinks to 2¢ (model converged with market)
EXIT_HOURS_BEFORE   = 6      # always exit at least 6h before event resolves


from dataclasses import dataclass

@dataclass
class WeatherPosition:
    ticker: str
    side: str
    contracts: int
    entry_price_c: int
    model_prob: float
    target_date: str
    lat: float
    lon: float
    title: str


class WeatherStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk = risk
        self._last_scan = 0
        self._scan_interval = 600   # only re-scan weather every 10 min (forecasts don't change faster)
        self._positions: dict[str, WeatherPosition] = {}

    def check_exits(self):
        """Exit positions when edge has collapsed or event is approaching."""
        for ticker, pos in list(self._positions.items()):
            reason = self._should_exit(pos)
            if reason:
                self._exit_position(pos, reason)

    def _should_exit(self, pos: WeatherPosition) -> object:
        # Time-based: exit 6h before the event date
        try:
            from datetime import datetime, timezone, timedelta
            event_dt = datetime.strptime(pos.target_date, "%Y-%m-%d").replace(
                hour=18, tzinfo=timezone.utc
            )
            hours_left = (event_dt - datetime.now(timezone.utc)).total_seconds() / 3600
            if hours_left < EXIT_HOURS_BEFORE:
                return f"event approaching ({hours_left:.1f}h left)"
        except Exception:
            pass

        # Edge collapse: re-fetch forecast and re-check the edge
        forecast = _get_forecast(pos.lat, pos.lon, pos.target_date)
        if forecast:
            new_model_prob = self._estimate_prob_static(pos.title, forecast)
            if new_model_prob is not None:
                try:
                    m = self.kalshi.get_market(ticker).get("market", {})
                    yes_bid = float(m.get("yes_bid_dollars") or 0)
                    yes_ask = float(m.get("yes_ask_dollars") or 0)
                    kalshi_prob = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else yes_ask
                    current_edge = abs(new_model_prob - kalshi_prob)
                    if current_edge < EXIT_EDGE_THRESHOLD:
                        return f"edge collapsed ({current_edge*100:.1f}¢)"
                except Exception:
                    pass

        return None

    def _exit_position(self, pos: WeatherPosition, reason: str):
        try:
            self.kalshi.place_order(
                ticker=pos.ticker, side=pos.side, action="sell",
                count=pos.contracts, order_type="market",
            )
            self.risk.record_close(pos.ticker, pos.contracts)
            log.info(f"[weather] EXIT {pos.ticker} {pos.side.upper()} x{pos.contracts}  reason={reason}")
        except Exception as e:
            log.error(f"[weather] Exit failed {pos.ticker}: {e}")
        finally:
            self._positions.pop(pos.ticker, None)

    @staticmethod
    def _estimate_prob_static(title: str, forecast: dict) -> object:
        return WeatherStrategy._estimate_prob(None, title, forecast)  # type: ignore

    def scan(self) -> list[dict]:
        self.check_exits()  # always check exits first

        now = time.time()
        if now - self._last_scan < self._scan_interval:
            return []
        self._last_scan = now

        try:
            resp = self.kalshi.get_markets(limit=200)
        except Exception as e:
            log.warning(f"[weather] Market fetch failed: {e}")
            return []

        opps = []
        for m in resp.get("markets", []):
            if m.get("status") != "active":
                continue
            title = m.get("title", "")
            if not any(w in title.lower() for w in ["temperature", "degrees", "°f", "rain", "snow", "precip", "high of"]):
                continue

            coords = _extract_city(title)
            if not coords:
                continue

            target_date = _extract_date(title)
            if not target_date:
                continue

            forecast = _get_forecast(coords[0], coords[1], target_date)
            if not forecast:
                continue

            yes_ask = float(m.get("yes_ask_dollars") or 0)
            yes_bid = float(m.get("yes_bid_dollars") or 0)
            if not yes_ask:
                continue

            model_prob = self._estimate_prob(title, forecast)
            if model_prob is None:
                continue

            kalshi_prob = yes_ask  # ask = implied probability for YES buyer
            edge = model_prob - kalshi_prob

            if abs(edge) < MIN_EDGE:
                continue

            side = "yes" if edge > 0 else "no"
            price = yes_ask if side == "yes" else float(m.get("no_ask_dollars") or (1 - yes_bid))

            opps.append({
                "type": "weather",
                "ticker": m["ticker"],
                "title": title,
                "side": side,
                "price": price,
                "price_c": int(price * 100),
                "model_prob": round(model_prob, 3),
                "kalshi_prob": round(kalshi_prob, 3),
                "edge": round(abs(edge), 3),
                "forecast": forecast,
                "target_date": target_date,
                "coords": coords,
            })

        opps.sort(key=lambda x: x["edge"], reverse=True)
        return opps

    def _estimate_prob(self, title: str, forecast: dict) -> object:
        title_lower = title.lower()
        high = forecast.get("high_f")
        precip = forecast.get("precip_in")
        precip_prob = forecast.get("precip_prob")

        # Temperature above threshold
        if high is not None and ("above" in title_lower or "reach" in title_lower or "high of" in title_lower or "exceed" in title_lower):
            threshold = _extract_temp_threshold(title)
            if threshold:
                diff = high - threshold
                # Sigmoid-like: if forecast high == threshold, ~50%; ±5°F ~ ±30%
                import math
                return max(0.02, min(0.98, 0.5 + 0.06 * diff))

        # Precipitation
        if "rain" in title_lower or "precip" in title_lower or "snow" in title_lower:
            if precip_prob is not None:
                return precip_prob / 100.0

        return None

    def execute(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_c = opp["price_c"]
        edge    = opp["edge"]

        contracts = self.risk.kelly_contracts(price_c, opp["model_prob"], max_contracts=50)
        contracts = max(1, contracts)

        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[weather] Skipped {ticker}: {reason}")
            return False

        coords = opp.get("coords", (0.0, 0.0))
        try:
            self.kalshi.place_order(
                ticker=ticker, side=side, action="buy",
                count=contracts, order_type="limit",
                yes_price=price_c if side == "yes" else None,
                no_price=price_c  if side == "no"  else None,
            )
            self.risk.record_open(ticker, contracts)
            self._positions[ticker] = WeatherPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, model_prob=opp["model_prob"],
                target_date=opp.get("target_date", ""),
                lat=coords[0], lon=coords[1],
                title=opp.get("title", ""),
            )
            log.info(
                f"[weather] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢  "
                f"model={opp['model_prob']*100:.0f}%  kalshi={opp['kalshi_prob']*100:.0f}%  "
                f"edge={edge*100:.0f}¢"
            )
            return True
        except Exception as e:
            log.error(f"[weather] Order failed {ticker}: {e}")
            return False
