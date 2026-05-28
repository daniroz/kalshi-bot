"""
ML calibration strategy — find markets where the order book disagrees with
a trained probability model.

Two pieces:
  1. **FeatureRecorder** (always running when enabled): every hour, snapshots
     features for every open Kalshi market we can see and writes them to
     ml_data/snapshots.jsonl. Each line is one (ticker, timestamp, features)
     record. After markets resolve, train_calibration.py joins these with
     outcomes to produce a training dataset.
  2. **MLCalibrationStrategy**: at inference time, computes the same features
     for currently-open markets and asks the model for P(YES wins). If the
     model's prediction is > yes_ask + threshold OR < yes_bid - threshold,
     trade with size = kelly_fraction × max_position.

Honest scope: this strategy STARTS OFF and does nothing until a trained model
exists at models/ml_calibration.pkl. The feature recorder runs regardless so
data starts accumulating immediately.

Why not use settled-market prices directly to train: those prices reflect the
post-settlement state (yes=99¢ or 0¢). Training on them produces a circular
model. Forward-collected hourly snapshots paired with eventual outcomes are
the only path to real training data here.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


SNAPSHOT_PATH = Path(__file__).parent.parent / "ml_data" / "snapshots.jsonl"
MODEL_PATH    = Path(__file__).parent.parent / "models" / "ml_calibration.pkl"

# ── Tunables ───────────────────────────────────────────────────────────────
SNAPSHOT_EVERY_S    = 3600       # 1 hour between snapshots
MAX_OPEN_MARKETS    = 500        # cap snapshot volume per cycle
SCAN_INTERVAL_S     = 300        # 5 min between inference passes

MIN_EDGE            = 0.05       # require ≥5¢ divergence between model & market
MIN_PRICE_C         = 15         # don't bet extreme prices
MAX_PRICE_C         = 85
MIN_TRADE_DOLLARS   = 5.0
MAX_TRADE_DOLLARS   = 30.0       # was 15.0 (May 28) — sized up 2x; still smaller than other strategies until model is trusted
KELLY_FRACTION      = 0.25       # quarter Kelly — conservative

# Series we consider "in scope" for the model — exclude one-off / weird stuff
ELIGIBLE_PREFIXES = (
    "KXBTC","KXETH","KXDOGE","KXXRP","KXSOL",
    "KXINX","KXNDAQ","KXSPXCLOSE","KXDJI",
    "KXWTI","KXGOLD","KXNATGAS",
    "KXEURUSD","KXUSDJPY","KXGBPUSD",
    "KXHIGHNY","KXHIGHCHI","KXHIGHMIA","KXHIGHLAX",
    "KXLOWTNYC","KXLOWTCHI","KXLOWTMIA","KXLOWTLAX",
    "KXNBAGAME","KXNHLGAME","KXMLBGAME",
    "KXCPI","KXFED","KXGDP","KXNFP","KXUNEMPLOYMENT",
)


# ── Feature extraction ─────────────────────────────────────────────────────
def extract_features(m: dict) -> Optional[dict]:
    """Pull features we'll use for both training and inference."""
    ticker = m.get("ticker", "")
    prefix = ticker.split("-")[0] if "-" in ticker else ticker
    if not any(prefix.startswith(p) or prefix == p for p in ELIGIBLE_PREFIXES):
        return None

    yes_bid = float(m.get("yes_bid_dollars") or 0)
    yes_ask = float(m.get("yes_ask_dollars") or 0)
    if yes_ask <= 0 or yes_bid <= 0:
        return None

    volume_24h = float(m.get("volume_24h_fp") or 0)
    open_interest = float(m.get("open_interest") or 0)
    open_ts_str = m.get("open_time") or ""
    close_ts_str = m.get("close_time") or ""
    try:
        open_dt  = datetime.fromisoformat(open_ts_str.replace("Z","+00:00"))
        close_dt = datetime.fromisoformat(close_ts_str.replace("Z","+00:00"))
        now      = datetime.now(timezone.utc)
        market_age_h     = (now - open_dt).total_seconds() / 3600
        time_to_close_h  = (close_dt - now).total_seconds() / 3600
        market_life_h    = (close_dt - open_dt).total_seconds() / 3600
        if market_life_h <= 0:
            return None
        life_fraction = market_age_h / market_life_h   # 0=just opened, 1=closing now
    except Exception:
        return None

    if time_to_close_h <= 0:    # already past close
        return None

    return {
        "ticker":            ticker,
        "prefix":            prefix,
        "yes_bid":           yes_bid,
        "yes_ask":           yes_ask,
        "midpoint":          (yes_bid + yes_ask) / 2,
        "spread":            yes_ask - yes_bid,
        "volume_24h":        volume_24h,
        "open_interest":     open_interest,
        "log_volume":        math.log1p(volume_24h),
        "log_oi":            math.log1p(open_interest),
        "time_to_close_h":   time_to_close_h,
        "market_age_h":      market_age_h,
        "life_fraction":     life_fraction,
        "snapshot_ts":       int(time.time()),
    }


# ── Feature recorder ───────────────────────────────────────────────────────
class FeatureRecorder:
    """Periodically writes feature snapshots for all open markets to JSONL."""

    def __init__(self, kalshi: KalshiClient):
        self.kalshi = kalshi
        self._last_snapshot: float = 0.0
        self._lock = threading.Lock()

    def maybe_snapshot(self) -> int:
        """If due, write snapshots. Returns count written."""
        now = time.time()
        with self._lock:
            if now - self._last_snapshot < SNAPSHOT_EVERY_S:
                return 0
            self._last_snapshot = now

        try:
            markets = get_liquid_markets(self.kalshi, min_volume=0)
        except Exception as e:
            log.warning(f"[ml-collect] market fetch failed: {e}")
            return 0

        n_written = 0
        SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
        with open(SNAPSHOT_PATH, "a") as f:
            for m in markets[:MAX_OPEN_MARKETS]:
                feats = extract_features(m)
                if not feats:
                    continue
                f.write(json.dumps(feats) + "\n")
                n_written += 1
        if n_written:
            log.info(f"[ml-collect] wrote {n_written} feature snapshots → {SNAPSHOT_PATH.name}")
        return n_written


# ── Strategy ───────────────────────────────────────────────────────────────
@dataclass
class MLPosition:
    ticker: str
    side:   str          # "yes" or "no"
    contracts: int
    entry_price_c: int
    model_prob: float


class MLCalibrationStrategy:
    """Trades when the trained model disagrees with the order book by >MIN_EDGE."""

    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk = risk
        self._last_scan: float = 0.0
        self._positions: dict[str, MLPosition] = {}
        self._lock = threading.Lock()
        self.recorder = FeatureRecorder(kalshi)

        # Load model if present
        self.model = None
        self.feature_order: list[str] = []
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    bundle = pickle.load(f)
                self.model = bundle["model"]
                self.feature_order = bundle["feature_order"]
                log.info(f"[ml-cal] loaded model from {MODEL_PATH.name} "
                         f"({len(self.feature_order)} features)")
            except Exception as e:
                log.warning(f"[ml-cal] failed to load model: {e}")

    def _predict(self, feats: dict) -> Optional[float]:
        """Return predicted P(YES wins) or None if no model."""
        if not self.model or not self.feature_order:
            return None
        try:
            x = [float(feats.get(k, 0)) for k in self.feature_order]
            p = self.model.predict_proba([x])[0][1]   # index 1 = positive class
            return float(p)
        except Exception as e:
            log.debug(f"[ml-cal] predict failed for {feats.get('ticker')}: {e}")
            return None

    def scan(self) -> list[dict]:
        # Always run the recorder regardless of whether the model exists
        self.recorder.maybe_snapshot()

        now = time.time()
        with self._lock:
            if now - self._last_scan < SCAN_INTERVAL_S:
                return []
            self._last_scan = now

        if not self.model:
            # Model not trained yet — do nothing
            return []

        try:
            markets = get_liquid_markets(self.kalshi, min_volume=100)
        except Exception:
            return []

        opps = []
        for m in markets:
            ticker = m["ticker"]
            if ticker in self._positions:
                continue
            if self.risk.open_positions.get(ticker, 0) > 0:
                continue
            feats = extract_features(m)
            if not feats:
                continue
            p_yes = self._predict(feats)
            if p_yes is None:
                continue

            yes_ask_c = int(round(feats["yes_ask"] * 100))
            no_ask_c  = int(round(float(m.get("no_ask_dollars") or 0) * 100))
            if not (MIN_PRICE_C <= yes_ask_c <= MAX_PRICE_C):
                continue

            # Trade when model & market disagree
            market_p = yes_ask_c / 100
            edge = p_yes - market_p
            if abs(edge) < MIN_EDGE:
                continue

            if edge > 0 and yes_ask_c <= MAX_PRICE_C:
                # Model: YES more likely than market thinks. Buy YES.
                side, price_c = "yes", yes_ask_c
            elif edge < 0 and no_ask_c > 0 and no_ask_c <= MAX_PRICE_C:
                # Model: NO more likely. Buy NO.
                side, price_c = "no", no_ask_c
                edge = abs(edge)
            else:
                continue

            opps.append({
                "ticker": ticker, "side": side, "price_c": price_c,
                "model_prob": p_yes, "edge": edge, "feats": feats,
            })
        opps.sort(key=lambda x: -x["edge"])
        return opps[:5]

    def execute(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_c = opp["price_c"]
        edge    = opp["edge"]

        # Quarter-Kelly position size — fractional Kelly for safety
        p = opp["model_prob"] if side == "yes" else 1 - opp["model_prob"]
        b = (100 - price_c) / price_c    # net odds
        kelly_full = max(0, (p * (1 + b) - 1) / b) if b > 0 else 0
        kelly = kelly_full * KELLY_FRACTION
        tier_cap = self.risk.balance * self.risk.effective_max_position_pct()
        target_dollars = min(
            MAX_TRADE_DOLLARS,
            tier_cap,
            self.risk.balance * kelly,
        )
        if target_dollars < MIN_TRADE_DOLLARS:
            return False
        contracts = max(1, int(target_dollars / (price_c / 100)))

        with self._lock:
            if ticker in self._positions:
                return False
            self._positions[ticker] = None

        ok, reason = self.risk.approve_trade(
            ticker, price_c, contracts, edge * contracts
        )
        if not ok:
            log.info(f"[ml-cal] Skip {ticker}: {reason}")
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
            self._positions[ticker] = MLPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, model_prob=opp["model_prob"],
            )
            log.info(f"[ml-cal] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢  "
                     f"model_p={opp['model_prob']*100:.1f}%  edge={edge*100:.1f}¢")
            return True
        except Exception as e:
            log.error(f"[ml-cal] order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            self._positions.pop(ticker, None)
            return False
