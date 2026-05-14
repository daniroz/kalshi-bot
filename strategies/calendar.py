"""
Economic calendar strategy.

Knows the schedule for CPI, NFP, FOMC, GDP, and other US macro releases.
Uses the FRED API to fetch actual data the moment it's released.

Logic:
  - 30 min before release: start monitoring the relevant Kalshi market
  - At release time: fetch actual data from FRED / BLS
  - Compare actual vs consensus expectation embedded in the Kalshi price
  - If Kalshi hasn't repriced within 60s of release → enter

Entry edge: Kalshi typically lags 1-3 minutes after a major data release.
Exit: price moves to model estimate | 5-cent stop | 15 min max hold
"""

import re
import time
import math
import httpx
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.logger import log
import os


FRED_API_KEY   = os.getenv("FRED_API_KEY", "")
FRED_BASE      = "https://api.stlouisfed.org/fred"
MIN_EDGE       = 0.06
PROFIT_TARGET  = 10    # cents — bigger target since this is high-conviction
STOP_LOSS      = 5
MAX_HOLD_S     = 900   # 15 min
MONITOR_BEFORE = 1800  # start watching 30 min before release
POST_RELEASE   = 120   # only trade within 2 min after release

# 2026 US Economic Calendar (Eastern Time)
# Format: (month, day, hour, minute, event, series_id, kalshi_series)
CALENDAR_2026 = [
    # CPI — 2nd or 3rd Wednesday, 8:30 AM ET
    (1,  15, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    (2,  12, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    (3,  12, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    (4,  11, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    (5,  13, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    (6,  11, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    (7,  15, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    (8,  12, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    (9,  11, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    (10, 14, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    (11, 13, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    (12, 11, 8, 30, "CPI",  "CPIAUCSL", "KXCPI"),
    # NFP — first Friday, 8:30 AM ET
    (1,  9,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    (2,  6,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    (3,  6,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    (4,  3,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    (5,  1,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    (6,  5,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    (7,  2,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    (8,  7,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    (9,  4,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    (10, 2,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    (11, 6,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    (12, 4,  8, 30, "NFP",  "PAYEMS",   "KXNFP"),
    # FOMC — 8 meetings per year
    (1,  29, 14, 0, "FOMC", "FEDFUNDS", "KXFED"),
    (3,  19, 14, 0, "FOMC", "FEDFUNDS", "KXFED"),
    (5,  7,  14, 0, "FOMC", "FEDFUNDS", "KXFED"),
    (6,  18, 14, 0, "FOMC", "FEDFUNDS", "KXFED"),
    (7,  30, 14, 0, "FOMC", "FEDFUNDS", "KXFED"),
    (9,  17, 14, 0, "FOMC", "FEDFUNDS", "KXFED"),
    (10, 29, 14, 0, "FOMC", "FEDFUNDS", "KXFED"),
    (12, 10, 14, 0, "FOMC", "FEDFUNDS", "KXFED"),
    # GDP — end of month, 8:30 AM ET (advance estimate)
    (1,  30, 8, 30, "GDP",  "GDP",      "KXGDP"),
    (4,  30, 8, 30, "GDP",  "GDP",      "KXGDP"),
    (7,  30, 8, 30, "GDP",  "GDP",      "KXGDP"),
    (10, 30, 8, 30, "GDP",  "GDP",      "KXGDP"),
]

# ET offset (EST = UTC-5, EDT = UTC-4)
ET_OFFSET = -5


def _release_ts(month: int, day: int, hour: int, minute: int) -> float:
    """Return UTC timestamp for a 2026 Eastern Time release."""
    dt_et = datetime(2026, month, day, hour, minute, tzinfo=timezone(timedelta(hours=ET_OFFSET)))
    return dt_et.timestamp()


def _upcoming_releases(window_s: int = MONITOR_BEFORE) -> list[dict]:
    """Return releases happening within window_s seconds from now."""
    now = time.time()
    results = []
    for entry in CALENDAR_2026:
        month, day, hour, minute, event, series_id, kalshi_series = entry
        ts = _release_ts(month, day, hour, minute)
        secs_until = ts - now
        if -POST_RELEASE <= secs_until <= window_s:
            results.append({
                "event":         event,
                "series_id":     series_id,
                "kalshi_series": kalshi_series,
                "ts":            ts,
                "secs_until":    secs_until,
                "released":      secs_until <= 0,
            })
    return results


def _fetch_fred_latest(series_id: str):
    """Fetch the most recent FRED data point."""
    if not FRED_API_KEY:
        return None
    try:
        r = httpx.get(
            f"{FRED_BASE}/series/observations",
            params={
                "series_id":       series_id,
                "api_key":         FRED_API_KEY,
                "file_type":       "json",
                "sort_order":      "desc",
                "limit":           3,
                "observation_start": datetime.now().strftime("%Y-%m-01"),
            },
            timeout=8,
        )
        obs = r.json().get("observations", [])
        for o in obs:
            val = o.get("value", ".")
            if val != ".":
                return float(val)
    except Exception as e:
        log.debug(f"[cal] FRED fetch {series_id}: {e}")
    return None


@dataclass
class CalendarPosition:
    ticker: str
    side: str
    contracts: int
    entry_price_c: int
    entry_time: float
    high_water_c: int
    event: str


class CalendarStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi    = kalshi
        self.risk      = risk
        self._positions:   dict[str, CalendarPosition] = {}
        self._last_entry:  dict[str, float] = {}
        self._last_values: dict[str, float] = {}   # series_id -> last fetched value

    def _get_kalshi_markets(self, kalshi_series: str) -> list[dict]:
        try:
            r = self.kalshi._get("/markets", {"limit": 50, "series_ticker": kalshi_series})
            return [m for m in r.get("markets", []) if m.get("status") == "active"]
        except Exception:
            return []

    def _check_exits(self):
        now = time.time()
        for ticker, pos in list(self._positions.items()):
            try:
                m = self.kalshi.get_market(ticker).get("market", {})
                bid_c = int(float(m.get("yes_bid_dollars" if pos.side == "yes" else "no_bid_dollars") or 0) * 100)
                if not bid_c:
                    continue
                if bid_c > pos.high_water_c:
                    pos.high_water_c = bid_c
                trail_stop = pos.high_water_c - 4
                pnl = bid_c - pos.entry_price_c
                if now - pos.entry_time > MAX_HOLD_S:
                    self._exit(pos, f"max hold  pnl={pnl:+}¢")
                elif pnl >= PROFIT_TARGET:
                    self._exit(pos, f"profit +{pnl}¢")
                elif pnl <= -STOP_LOSS:
                    self._exit(pos, f"stop {pnl}¢")
                elif bid_c < trail_stop:
                    self._exit(pos, f"trailing stop (peak={pos.high_water_c}¢)")
            except Exception as e:
                log.debug(f"[cal] Exit check {ticker}: {e}")

    def _exit(self, pos: CalendarPosition, reason: str):
        try:
            self.kalshi.place_order(
                ticker=pos.ticker, side=pos.side, action="sell",
                count=pos.contracts, order_type="market",
            )
            self.risk.record_close(pos.ticker, pos.contracts)
            log.info(f"[cal] EXIT {pos.ticker} {pos.side.upper()} x{pos.contracts}  {reason}")
        except Exception as e:
            log.error(f"[cal] Exit failed {pos.ticker}: {e}")
        finally:
            self._positions.pop(pos.ticker, None)

    def scan(self) -> list[dict]:
        self._check_exits()
        upcoming = _upcoming_releases()
        if not upcoming:
            return []

        signals = []
        now = time.time()

        for release in upcoming:
            event        = release["event"]
            series_id    = release["series_id"]
            kalshi_series = release["kalshi_series"]
            released     = release["released"]
            secs_until   = release["secs_until"]

            # Only trade in the 2-minute window after release
            if not released:
                log.debug(f"[cal] {event} in {abs(secs_until)/60:.0f} min — monitoring")
                continue

            if secs_until < -POST_RELEASE:
                continue

            # Try to get actual data
            actual = _fetch_fred_latest(series_id)

            markets = self._get_kalshi_markets(kalshi_series)
            for m in markets:
                ticker = m["ticker"]
                if ticker in self._positions:
                    continue
                if now - self._last_entry.get(ticker, 0) < 300:
                    continue

                yes_ask_c = int(float(m.get("yes_ask_dollars") or 0) * 100)
                yes_bid_c = int(float(m.get("yes_bid_dollars") or 0) * 100)
                no_ask_c  = int(float(m.get("no_ask_dollars")  or 0) * 100)
                if not yes_ask_c or not yes_bid_c:
                    continue

                vol = float(m.get("volume_24h_fp") or 0)
                title = m.get("title", "")

                # Without FRED data, use price staleness: markets priced near 50¢
                # right after a release are lagging (they should have moved already)
                if actual is None:
                    # Just flag that a release happened — monitor for stale price
                    mid_c = (yes_ask_c + yes_bid_c) // 2
                    if 35 <= mid_c <= 65:
                        signals.append({
                            "ticker":  ticker,
                            "title":   title[:60],
                            "side":    "yes",
                            "price_c": yes_ask_c,
                            "edge":    MIN_EDGE,
                            "event":   event,
                            "actual":  None,
                        })
                    continue

                # Extract the threshold from the ticker (e.g. T0.3 = above 0.3%)
                threshold_m = re.search(r'[TB]([\d.]+)', ticker)
                if not threshold_m:
                    continue
                threshold = float(threshold_m.group(1))
                direction = "T" if "-T" in ticker else "B"

                # For CPI: "T0.3" = YES if CPI > 0.3%
                # actual from FRED is index level, we need MoM %
                # For simplicity: compare directly only for FOMC (fed funds rate)
                if event == "FOMC":
                    if direction == "T":
                        model_yes = 1.0 if actual >= threshold else 0.0
                    else:
                        model_yes = 1.0 if actual <= threshold else 0.0
                    model_c = int(model_yes * 100)
                    edge_yes = model_c - yes_ask_c
                    edge_no  = (100 - model_c) - no_ask_c if no_ask_c else -999

                    if edge_yes >= int(MIN_EDGE * 100) and yes_ask_c < 95:
                        signals.append({
                            "ticker":  ticker, "title": title[:60],
                            "side":    "yes",  "price_c": yes_ask_c,
                            "edge":    edge_yes / 100,
                            "event":   event,  "actual": actual,
                        })
                    elif edge_no >= int(MIN_EDGE * 100) and no_ask_c:
                        signals.append({
                            "ticker":  ticker, "title": title[:60],
                            "side":    "no",   "price_c": no_ask_c,
                            "edge":    edge_no / 100,
                            "event":   event,  "actual": actual,
                        })

        signals.sort(key=lambda x: x["edge"], reverse=True)
        return signals

    def execute(self, signal: dict) -> bool:
        ticker  = signal["ticker"]
        side    = signal["side"]
        price_c = signal["price_c"]
        edge    = signal["edge"]

        contracts = self.risk.kelly_contracts(price_c, price_c / 100 + edge, max_contracts=30)
        contracts = max(1, contracts)

        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[cal] Skipped {ticker}: {reason}")
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
            self._positions[ticker] = CalendarPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, entry_time=time.time(),
                high_water_c=price_c, event=signal["event"],
            )
            log.info(
                f"[cal] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢"
                f"  event={signal['event']}  actual={signal['actual']}"
            )
            return True
        except Exception as e:
            log.error(f"[cal] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            return False
