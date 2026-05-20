"""
Backtest framework for the Kalshi bot.

Two modes — pick the one that fits the strategy:

  1. REPLAY (historical, immediate results)
     For settlement strategies only. Walks every settled Kalshi market in a
     date range, pulls the underlying data (Coinbase/Yahoo/Open-Meteo) at the
     strategy's natural entry window, simulates the strategy's _evaluate()
     decision, estimates the Kalshi entry price using the underlying-vs-
     threshold proximity, and computes P&L against the known settlement.

     Limitations:
       - Entry-price estimation is a heuristic, not exact orderbook replay
         (Kalshi's historical orderbook isn't accessible). We use:
           if YES is "locked" by our model → assume yes_ask ≈ 0.82
           if NO is "locked"               → assume no_ask  ≈ 0.82
         then apply the MIN_PRICE_C floor and MAX_PRICE_C ceiling as the live
         strategy would. This biases toward optimistic edge but is consistent.
       - Slippage modeled as 1¢ of additional cost.
       - Only the SETTLEMENT strategies can be replayed this way. The 9 OFF
         model-based strategies depend on Kalshi price history which we don't
         have — for those, use --shadow.

  2. SHADOW (forward-paper, real conditions)
     Runs strategies live but intercepts kalshi.place_order() with a journal
     writer. Real prices, real timing, real settlement, no real money. Run
     for ≥7 days then --report to see verdicts on each strategy.

Usage:
  python backtest.py --replay --strategy settle-crypto --days 7
  python backtest.py --replay --all --days 14
  python backtest.py --shadow --strategy momentum     # enable, then run main.py
  python backtest.py --report                          # read paper-trade journal
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import csv
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Lazy imports so --help works without all deps
def _client():
    from clients.kalshi import KalshiClient
    return KalshiClient(
        api_key=os.environ["KALSHI_API_KEY"],
        key_file=str(Path(__file__).parent / "kalshi_private.pem"),
        demo=False,
    )


RESULTS_DIR = Path(__file__).parent / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Heuristic entry pricing ─────────────────────────────────────────────────
# We can't replay the Kalshi orderbook exactly, but we know that when underlying
# data clearly indicates the outcome, market makers tend to price the "locked"
# side around 80-90¢. We use a midpoint and let MAX_PRICE_C cap it.
def _estimate_locked_price_c(underlying_distance_pct: float) -> int:
    """
    Given how far the underlying is past the threshold (as a fraction of the
    safety margin — e.g., 1.0 = right at the margin, 3.0 = 3x past margin),
    estimate the YES (or NO) ask price the market would have shown.

    Beyond 3x margin we cap at 88¢ (close to MAX_PRICE_C of 89).
    Closer to threshold the market is less convinced — lower price.
    """
    if underlying_distance_pct <= 0.5:
        return 60   # borderline — market still uncertain
    if underlying_distance_pct <= 1.0:
        return 72
    if underlying_distance_pct <= 2.0:
        return 80
    return 86


def _kalshi_fee_cents(price_c: int, contracts: int = 1) -> float:
    p = price_c / 100
    return 0.07 * p * (1 - p) * 100 * contracts


# ── Historical data fetchers (cached on disk so we don't hammer APIs) ──────
def _cache_path(name: str) -> Path:
    return RESULTS_DIR / f".cache_{name}.json"

def _cached_get(name: str, fetcher, ttl_seconds: int = 86400):
    p = _cache_path(name)
    if p.exists() and time.time() - p.stat().st_mtime < ttl_seconds:
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    result = fetcher()
    if result is not None:
        try:
            p.write_text(json.dumps(result))
        except Exception:
            pass
    return result


def fetch_coinbase_history(product: str, start_ts: int, end_ts: int,
                            granularity: int = 60) -> list[dict]:
    """1-min Coinbase candles. Returns list of {ts, open, high, low, close}."""
    cache_key = f"coinbase_{product}_{start_ts}_{end_ts}_{granularity}"
    def _fetch():
        try:
            url = f"https://api.exchange.coinbase.com/products/{product}/candles"
            r = httpx.get(url, params={
                "start": datetime.fromtimestamp(start_ts, timezone.utc).isoformat(),
                "end":   datetime.fromtimestamp(end_ts,   timezone.utc).isoformat(),
                "granularity": granularity,
            }, timeout=15)
            r.raise_for_status()
            # Coinbase format: [[ts, low, high, open, close, volume], ...]
            return [{"ts": row[0], "low": row[1], "high": row[2],
                     "open": row[3], "close": row[4], "volume": row[5]}
                    for row in r.json()]
        except Exception as e:
            print(f"  [data] coinbase {product} failed: {e}")
            return []
    return _cached_get(cache_key, _fetch) or []


def fetch_yahoo_history(symbol: str, days: int = 7) -> list[dict]:
    """1-min Yahoo chart bars. Returns list of {ts, close, high, low}.
    Yahoo's /v8/chart rejects range > 7d when interval=1m — clamp accordingly."""
    days = min(days, 7)
    cache_key = f"yahoo_{symbol.replace('=','_').replace('^','_')}_{days}"
    def _fetch():
        try:
            r = httpx.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"interval": "1m", "range": f"{days}d"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            res = data.get("chart", {}).get("result", [])
            if not res:
                return []
            ts_list = res[0].get("timestamp", []) or []
            ind = res[0].get("indicators", {}).get("quote", [{}])[0]
            closes = ind.get("close") or []
            highs  = ind.get("high") or []
            lows   = ind.get("low") or []
            return [
                {"ts": ts, "close": c, "high": h, "low": l}
                for ts, c, h, l in zip(ts_list, closes, highs, lows)
                if c is not None
            ]
        except Exception as e:
            print(f"  [data] yahoo {symbol} failed: {e}")
            return []
    return _cached_get(cache_key, _fetch) or []


def fetch_openmeteo_history(lat: float, lon: float, date: str, tz: str) -> dict:
    """Hourly observed temperatures for a single day in city local TZ."""
    cache_key = f"openmeteo_{round(lat,2)}_{round(lon,2)}_{date}"
    def _fetch():
        try:
            r = httpx.get("https://archive-api.open-meteo.com/v1/archive", params={
                "latitude": lat, "longitude": lon,
                "start_date": date, "end_date": date,
                "hourly": "temperature_2m",
                "temperature_unit": "fahrenheit",
                "timezone": tz,
            }, timeout=15)
            r.raise_for_status()
            return r.json().get("hourly", {})
        except Exception as e:
            print(f"  [data] open-meteo {lat},{lon} {date} failed: {e}")
            return {}
    return _cached_get(cache_key, _fetch) or {}


# ── Settled-market fetcher from Kalshi ──────────────────────────────────────
def fetch_settled_markets(series_prefixes: list[str], days: int) -> list[dict]:
    """Pull all settled Kalshi markets in the last `days` days, filtered by prefix."""
    k = _client()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    all_markets = []
    for prefix in series_prefixes:
        cursor = None
        for _ in range(10):
            params = {"limit": 100, "series_ticker": prefix, "status": "settled"}
            if cursor:
                params["cursor"] = cursor
            try:
                r = k._get("/markets", params)
            except Exception as e:
                print(f"  [kalshi] settled fetch failed for {prefix}: {e}")
                break
            batch = r.get("markets", [])
            for m in batch:
                ct = m.get("close_time", "")
                try:
                    close_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    if close_dt >= cutoff:
                        all_markets.append(m)
                except Exception:
                    pass
            cursor = r.get("cursor")
            if not cursor or not batch:
                break
    return all_markets


# ── Replay results ──────────────────────────────────────────────────────────
@dataclass
class Trade:
    strategy:    str
    ticker:      str
    side:        str
    entry_c:     int
    contracts:   int
    settled_to:  str        # "yes" or "no"
    profit_c:    float      # net cents per contract (incl. fee)
    profit_usd:  float      # total
    reason:      str


# ── Per-strategy replay implementations ─────────────────────────────────────

def replay_crypto(markets: list[dict], verbose: bool = False) -> list[Trade]:
    """Replay settlement_crypto against settled BTC/ETH/DOGE/XRP markets."""
    from strategies.settlement_crypto import (
        SUPPORTED_SERIES, MIN_EDGE_C, MIN_PRICE_C, MAX_PRICE_C,
        ABS_MARGIN_FLOOR_USD, MIN_TRADE_DOLLARS, MAX_TRADE_DOLLARS,
        RESOLVE_MIN_S, RESOLVE_MAX_S, _parse_crypto_ticker,
        _safety_margin_pct,
    )

    trades: list[Trade] = []
    for m in markets:
        ticker = m["ticker"]
        parsed = _parse_crypto_ticker(ticker)
        if not parsed:
            continue

        product = parsed["pair"]
        threshold = parsed["threshold"]
        result = (m.get("result") or "").lower()
        if result not in ("yes", "no"):
            continue

        # The decision window: RESOLVE_MIN_S to RESOLVE_MAX_S before settle.
        # Walk forward through it minute-by-minute and take the FIRST trigger.
        settle_dt = parsed["resolve_dt"]
        window_start_ts = int((settle_dt - timedelta(seconds=RESOLVE_MAX_S)).timestamp())
        window_end_ts   = int((settle_dt - timedelta(seconds=RESOLVE_MIN_S)).timestamp())
        if window_end_ts <= window_start_ts:
            continue

        candles = fetch_coinbase_history(product, window_start_ts - 60, window_end_ts + 60)
        if not candles:
            continue
        candles.sort(key=lambda c: c["ts"])

        triggered = False
        for c in candles:
            ts = c["ts"]
            if not (window_start_ts <= ts <= window_end_ts):
                continue
            spot = c["close"]
            seconds_left = (settle_dt - datetime.fromtimestamp(ts, timezone.utc)).total_seconds()
            margin = max(threshold * _safety_margin_pct(seconds_left), ABS_MARGIN_FLOOR_USD)

            side = None
            distance = 0.0
            if spot >= threshold + margin:
                side = "yes"
                distance = (spot - threshold) / margin
            elif spot <= threshold - margin:
                side = "no"
                distance = (threshold - spot) / margin
            if side is None:
                continue

            entry_c = _estimate_locked_price_c(distance)
            entry_c = max(MIN_PRICE_C, min(MAX_PRICE_C, entry_c)) + 1   # +1 slippage
            fee_c = _kalshi_fee_cents(entry_c, 1)
            edge_c = (100 - entry_c) - fee_c
            if edge_c < MIN_EDGE_C:
                continue

            # Sized for $25 (typical tier-0 capped settle trade)
            contracts = max(1, int(25.0 / (entry_c / 100)))

            won = (side == result)
            profit_per_contract = ((100 - entry_c) - fee_c) if won else -entry_c - fee_c
            profit_usd = profit_per_contract * contracts / 100

            trades.append(Trade(
                strategy="settle-crypto",
                ticker=ticker, side=side,
                entry_c=entry_c, contracts=contracts,
                settled_to=result,
                profit_c=profit_per_contract,
                profit_usd=round(profit_usd, 2),
                reason=f"spot=${spot:,.2f} vs T=${threshold:,.2f} margin=${margin:,.2f} ({seconds_left/60:.1f}min)",
            ))
            triggered = True
            if verbose:
                print(f"    {ticker:35s}  {side.upper():3s} @ {entry_c}¢ x{contracts}  "
                      f"→ settled={result.upper()}  P&L = ${profit_usd:+.2f}")
            break

    return trades


def replay_stocks(markets: list[dict], verbose: bool = False) -> list[Trade]:
    """Replay settlement_stocks against settled SPX/NDX/DJI markets."""
    from strategies.settlement_stocks import (
        SUPPORTED_SERIES, MIN_EDGE_C, MIN_PRICE_C, MAX_PRICE_C,
        MIN_TRADE_DOLLARS, MAX_TRADE_DOLLARS,
        RESOLVE_MIN_S, RESOLVE_MAX_S, _parse_stock_ticker,
        _safety_margin_points,
    )

    trades: list[Trade] = []
    # Pull 7 days of 1-min bars for each unique symbol once
    symbol_cache: dict[str, list[dict]] = {}

    for m in markets:
        ticker = m["ticker"]
        parsed = _parse_stock_ticker(ticker)
        if not parsed:
            continue
        symbol = parsed["symbol"]
        threshold = parsed["threshold"]
        result = (m.get("result") or "").lower()
        if result not in ("yes", "no"):
            continue

        if symbol not in symbol_cache:
            symbol_cache[symbol] = fetch_yahoo_history(symbol, days=14)
        bars = symbol_cache[symbol]
        if not bars:
            continue

        settle_dt = parsed["resolve_dt"]
        window_start = int((settle_dt - timedelta(seconds=RESOLVE_MAX_S)).timestamp())
        window_end   = int((settle_dt - timedelta(seconds=RESOLVE_MIN_S)).timestamp())

        for b in bars:
            ts = b["ts"]
            if not (window_start <= ts <= window_end):
                continue
            quote = b["close"]
            seconds_left = (settle_dt - datetime.fromtimestamp(ts, timezone.utc)).total_seconds()
            margin = _safety_margin_points(seconds_left, symbol)

            side = None
            distance = 0.0
            if quote >= threshold + margin:
                side = "yes"
                distance = (quote - threshold) / margin
            elif quote <= threshold - margin:
                side = "no"
                distance = (threshold - quote) / margin
            if side is None:
                continue

            entry_c = _estimate_locked_price_c(distance)
            entry_c = max(MIN_PRICE_C, min(MAX_PRICE_C, entry_c)) + 1
            fee_c = _kalshi_fee_cents(entry_c, 1)
            edge_c = (100 - entry_c) - fee_c
            if edge_c < MIN_EDGE_C:
                continue

            contracts = max(1, int(25.0 / (entry_c / 100)))
            won = (side == result)
            profit_per_contract = ((100 - entry_c) - fee_c) if won else -entry_c - fee_c
            profit_usd = profit_per_contract * contracts / 100

            trades.append(Trade(
                strategy="settle-stocks",
                ticker=ticker, side=side,
                entry_c=entry_c, contracts=contracts,
                settled_to=result,
                profit_c=profit_per_contract,
                profit_usd=round(profit_usd, 2),
                reason=f"{symbol}={quote:.2f} vs T={threshold:.2f} margin={margin:.1f} ({seconds_left/60:.1f}min)",
            ))
            if verbose:
                print(f"    {ticker:40s}  {side.upper():3s} @ {entry_c}¢ x{contracts}  "
                      f"→ {result.upper()}  ${profit_usd:+.2f}")
            break

    return trades


def replay_weather(markets: list[dict], verbose: bool = False) -> list[Trade]:
    """Replay settlement (weather) against settled KXHIGH*/KXLOW* markets."""
    from strategies.settlement import (
        MIN_EDGE_C, MIN_PRICE_C, MAX_PRICE_C, TEMP_MARGIN_F,
        RESOLVE_WINDOW_MIN_H, RESOLVE_WINDOW_MAX_H,
    )
    from strategies.weather import SERIES, _parse_ticker

    trades: list[Trade] = []
    for m in markets:
        ticker = m["ticker"]
        series, target_date, threshold_f = _parse_ticker(ticker)
        if not series or not target_date or threshold_f is None:
            continue
        cfg = SERIES.get(series)
        if not cfg:
            continue
        result = (m.get("result") or "").lower()
        if result not in ("yes", "no"):
            continue

        obs = fetch_openmeteo_history(cfg["lat"], cfg["lon"], target_date, cfg["tz"])
        times = obs.get("time", []) or []
        temps = obs.get("temperature_2m", []) or []
        if not times or not temps:
            continue

        # Settle is end-of-day. Entry window starts at RESOLVE_WINDOW_MAX_H before.
        # By the late-afternoon (e.g. 4-8pm local for highs), max is usually settled.
        # Look at the observed max/min so far at hour 18 (6pm local) — a typical
        # late-day check.
        kind = cfg["kind"]
        # Use values up to the 18th hour
        snapshot_temps = [t for tt, t in zip(times, temps) if tt and "T18:" in tt or "T17:" in tt or "T16:" in tt]
        # Fallback: use all available
        if not snapshot_temps:
            snapshot_temps = [t for t in temps if t is not None]
        if not snapshot_temps:
            continue
        # For HIGH: did observed_max exceed threshold + margin?
        # For LOW:  did observed_min fall below threshold - margin?
        all_temps = [t for t in temps if t is not None]
        if not all_temps:
            continue
        full_max = max(all_temps)
        full_min = min(all_temps)

        side = None
        if kind == "high":
            if full_max >= threshold_f + TEMP_MARGIN_F:
                side = "yes"; distance = (full_max - threshold_f) / TEMP_MARGIN_F
            elif full_max < threshold_f - TEMP_MARGIN_F:
                side = "no";  distance = (threshold_f - full_max) / TEMP_MARGIN_F
        else:   # low
            if full_min <= threshold_f - TEMP_MARGIN_F:
                side = "yes"; distance = (threshold_f - full_min) / TEMP_MARGIN_F
            elif full_min > threshold_f + TEMP_MARGIN_F:
                side = "no";  distance = (full_min - threshold_f) / TEMP_MARGIN_F
            else:
                continue
        if side is None:
            continue

        entry_c = _estimate_locked_price_c(distance)
        entry_c = max(MIN_PRICE_C, min(MAX_PRICE_C, entry_c)) + 1
        fee_c = _kalshi_fee_cents(entry_c, 1)
        edge_c = (100 - entry_c) - fee_c
        if edge_c < MIN_EDGE_C:
            continue

        contracts = max(1, int(25.0 / (entry_c / 100)))
        won = (side == result)
        profit_per_contract = ((100 - entry_c) - fee_c) if won else -entry_c - fee_c
        profit_usd = profit_per_contract * contracts / 100

        trades.append(Trade(
            strategy="settle-weather",
            ticker=ticker, side=side,
            entry_c=entry_c, contracts=contracts,
            settled_to=result,
            profit_c=profit_per_contract,
            profit_usd=round(profit_usd, 2),
            reason=f"{kind}=({full_min:.1f}/{full_max:.1f})°F vs T={threshold_f}°F ±{TEMP_MARGIN_F}",
        ))
        if verbose:
            print(f"    {ticker:35s}  {side.upper():3s} @ {entry_c}¢ x{contracts}  "
                  f"→ {result.upper()}  ${profit_usd:+.2f}")

    return trades


REPLAY_HANDLERS = {
    "settle-crypto":  ("KXBTC KXETH KXDOGE KXXRP".split(),                              replay_crypto),
    "settle-stocks":  ("KXINX KXSPXCLOSE KXNDAQ KXDJI".split(),                         replay_stocks),
    "settle-weather": (["KXHIGHNY","KXHIGHCHI","KXHIGHMIA","KXHIGHLAX","KXHIGHDEN",
                        "KXHIGHTATL","KXHIGHTBOS","KXHIGHTDC","KXHIGHTHOU","KXHIGHTDAL",
                        "KXHIGHTPHX","KXHIGHTSEA","KXHIGHTSFO","KXHIGHPHIL","KXHIGHAUS",
                        "KXHIGHTNOLA","KXHIGHTMIN","KXHIGHTOKC","KXHIGHTLV","KXHIGHTSATX",
                        "KXLOWTNYC","KXLOWTCHI","KXLOWTMIA","KXLOWTLAX","KXLOWTDEN",
                        "KXLOWTATL","KXLOWTBOS","KXLOWTDC","KXLOWTHOU","KXLOWTDAL",
                        "KXLOWTPHX","KXLOWTSEA","KXLOWTSFO","KXLOWTPHIL","KXLOWTAUS",
                        "KXLOWTNOLA","KXLOWTMIN","KXLOWTOKC","KXLOWTLV","KXLOWTSATX"],
                       replay_weather),
}


# ── Stats ───────────────────────────────────────────────────────────────────
def summarize(trades: list[Trade], label: str = "OVERALL") -> dict:
    if not trades:
        return {"label": label, "n": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t.profit_usd > 0)
    total_pnl = sum(t.profit_usd for t in trades)
    avg_edge_c = sum(t.profit_c for t in trades) / n
    # Running PnL for drawdown
    running, peak, max_dd = 0.0, 0.0, 0.0
    for t in trades:
        running += t.profit_usd
        peak = max(peak, running)
        max_dd = min(max_dd, running - peak)
    avg_return = total_pnl / n
    # Crude Sharpe (per-trade)
    if n > 1:
        mean = avg_return
        var  = sum((t.profit_usd - mean) ** 2 for t in trades) / (n - 1)
        sd   = var ** 0.5 if var > 0 else 0
        sharpe = (mean / sd) * (n ** 0.5) if sd > 0 else 0
    else:
        sharpe = 0
    return {
        "label":         label,
        "n":             n,
        "wins":          wins,
        "win_rate":      round(wins / n * 100, 1),
        "total_pnl":     round(total_pnl, 2),
        "avg_pnl":       round(avg_return, 2),
        "avg_edge_c":    round(avg_edge_c, 2),
        "max_drawdown":  round(max_dd, 2),
        "sharpe":        round(sharpe, 2),
    }


def print_summary(stats: dict) -> None:
    print(f"\n{'='*72}")
    print(f"  {stats['label']}  (n={stats['n']})")
    if stats['n'] == 0:
        print(f"  no trades replayed")
        return
    sign = "+" if stats['total_pnl'] >= 0 else ""
    print(f"  Total P&L:      {sign}${stats['total_pnl']:.2f}")
    print(f"  Win rate:       {stats['win_rate']:.1f}%  ({stats['wins']}/{stats['n']})")
    print(f"  Avg P&L/trade:  ${stats['avg_pnl']:.2f}")
    print(f"  Avg edge:       {stats['avg_edge_c']:.2f}¢ per contract")
    print(f"  Max drawdown:   ${stats['max_drawdown']:.2f}")
    print(f"  Sharpe (rough): {stats['sharpe']:.2f}")


# ── Shadow paper-trade mode ─────────────────────────────────────────────────
SHADOW_JOURNAL = RESULTS_DIR / "paper_trades.jsonl"

def shadow_enable(strategies_csv: str) -> None:
    """Toggle shadow mode for the listed strategies in config.yaml."""
    flag_path = RESULTS_DIR / "shadow_strategies.txt"
    requested = [s.strip() for s in strategies_csv.split(",") if s.strip()]
    flag_path.write_text("\n".join(requested))
    print(f"✅ Shadow mode armed for: {', '.join(requested)}")
    print(f"   Journal: {SHADOW_JOURNAL}")
    print(f"   Now (re)start the bot. Strategies in this list will log paper trades")
    print(f"   instead of placing real orders. Run --report any time to see results.")


def shadow_disable() -> None:
    flag_path = RESULTS_DIR / "shadow_strategies.txt"
    if flag_path.exists():
        flag_path.unlink()
    print("Shadow mode disabled. Restart the bot for changes to take effect.")


def report_shadow() -> None:
    if not SHADOW_JOURNAL.exists():
        print("No paper trades logged yet.")
        print(f"  Expected: {SHADOW_JOURNAL}")
        print("  Run: python backtest.py --shadow <strategy>, then restart the bot.")
        return

    print(f"Reading journal: {SHADOW_JOURNAL}")
    trades_by_strategy: dict[str, list[dict]] = defaultdict(list)
    with open(SHADOW_JOURNAL) as f:
        for line in f:
            try:
                rec = json.loads(line)
                trades_by_strategy[rec.get("strategy", "?")].append(rec)
            except Exception:
                continue
    if not trades_by_strategy:
        print("Journal is empty.")
        return

    # Look up settlement results for the recorded paper trades
    k = _client()
    by_strategy_trades: dict[str, list[Trade]] = defaultdict(list)
    for strat, recs in trades_by_strategy.items():
        for rec in recs:
            ticker = rec.get("ticker", "")
            try:
                mkt = k.get_market(ticker).get("market", {})
            except Exception:
                continue
            status = mkt.get("status", "")
            if status != "settled":
                continue
            result = (mkt.get("result") or "").lower()
            if result not in ("yes", "no"):
                continue
            entry_c   = int(rec.get("price_c", 0))
            contracts = int(rec.get("contracts", 0))
            side      = rec.get("side", "")
            won = (side == result)
            fee_c = _kalshi_fee_cents(entry_c, 1)
            profit_pc  = ((100 - entry_c) - fee_c) if won else -entry_c - fee_c
            profit_usd = profit_pc * contracts / 100
            by_strategy_trades[strat].append(Trade(
                strategy=strat, ticker=ticker, side=side,
                entry_c=entry_c, contracts=contracts,
                settled_to=result, profit_c=profit_pc,
                profit_usd=round(profit_usd, 2),
                reason=rec.get("reason", ""),
            ))

    all_trades: list[Trade] = []
    for strat, ts in by_strategy_trades.items():
        all_trades.extend(ts)
        print_summary(summarize(ts, label=f"[shadow] {strat}"))
    if len(by_strategy_trades) > 1:
        print_summary(summarize(all_trades, label="[shadow] OVERALL"))


# ── CSV / journal writers ───────────────────────────────────────────────────
def write_trades_csv(trades: list[Trade], path: Path) -> None:
    if not trades:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(trades[0]).keys()))
        w.writeheader()
        for t in trades:
            w.writerow(asdict(t))


# ── CLI ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Kalshi bot backtest framework")
    sub = p.add_subparsers(dest="mode", required=True)

    rp = sub.add_parser("replay", help="historical replay of settlement strategies")
    rp.add_argument("--strategy", choices=list(REPLAY_HANDLERS.keys()) + ["all"],
                    default="all")
    rp.add_argument("--days", type=int, default=7, help="settled-market lookback window")
    rp.add_argument("--verbose", action="store_true")

    sh = sub.add_parser("shadow", help="enable shadow paper-trade mode")
    sh.add_argument("strategies", help="comma-separated strategy tags (e.g. momentum,news)")

    sub.add_parser("shadow-off", help="disable shadow paper-trade mode")
    sub.add_parser("report", help="report on shadow paper-trade journal")

    args = p.parse_args()

    if args.mode == "shadow":
        shadow_enable(args.strategies)
        return
    if args.mode == "shadow-off":
        shadow_disable()
        return
    if args.mode == "report":
        report_shadow()
        return

    # replay
    targets = list(REPLAY_HANDLERS.keys()) if args.strategy == "all" else [args.strategy]
    all_trades: list[Trade] = []
    for tag in targets:
        prefixes, handler = REPLAY_HANDLERS[tag]
        print(f"\n[{tag}]  Pulling settled markets for {len(prefixes)} series, {args.days}d back…")
        markets = fetch_settled_markets(prefixes, args.days)
        print(f"  {len(markets)} settled markets found")
        if not markets:
            continue
        trades = handler(markets, verbose=args.verbose)
        print(f"  {len(trades)} trade(s) replayed")
        all_trades.extend(trades)
        print_summary(summarize(trades, label=f"[{tag}]"))
        write_trades_csv(trades, RESULTS_DIR / f"replay_{tag}_{args.days}d.csv")

    if len(targets) > 1:
        print_summary(summarize(all_trades, label="ALL STRATEGIES COMBINED"))
        write_trades_csv(all_trades, RESULTS_DIR / f"replay_all_{args.days}d.csv")


if __name__ == "__main__":
    main()
