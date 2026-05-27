"""
Train the ML calibration model from accumulated feature snapshots + Kalshi outcomes.

The bot's FeatureRecorder writes feature snapshots to ml_data/snapshots.jsonl every
hour for every open market. This script:

  1. Reads all snapshot records
  2. For each unique ticker, queries Kalshi for current settlement status
  3. Pairs feature rows with eventual yes/no outcomes (drops still-open markets)
  4. Trains a logistic regression on (features) → P(yes wins)
  5. Saves the model to models/ml_calibration.pkl

Run periodically (cron / launchctl) — the more snapshots, the better the model.
Realistically: need ≥1,000 settled samples to get a model with any predictive value.

Usage:
    python train_calibration.py                 # train on all available data
    python train_calibration.py --report-only   # show data + skip training
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

SNAPSHOT_PATH = Path(__file__).parent / "ml_data" / "snapshots.jsonl"
MODEL_PATH    = Path(__file__).parent / "models" / "ml_calibration.pkl"

# Features used (must match strategy at inference time)
FEATURE_KEYS = [
    "yes_bid", "yes_ask", "midpoint", "spread",
    "log_volume", "log_oi",
    "time_to_close_h", "market_age_h", "life_fraction",
]


def load_snapshots() -> list[dict]:
    if not SNAPSHOT_PATH.exists():
        return []
    rows = []
    with open(SNAPSHOT_PATH) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def fetch_outcomes(tickers: list[str]) -> dict[str, str]:
    """Return {ticker: 'yes'|'no'|None} for each."""
    from clients.kalshi import KalshiClient
    k = KalshiClient(
        api_key=os.environ["KALSHI_API_KEY"],
        key_file=str(Path(__file__).parent / "kalshi_private.pem"),
        demo=False,
    )
    out = {}
    for t in tickers:
        try:
            m = k.get_market(t).get("market", {})
            if m.get("status") == "settled":
                out[t] = (m.get("result") or "").lower()
            else:
                out[t] = None
        except Exception:
            out[t] = None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    snaps = load_snapshots()
    print(f"Loaded {len(snaps)} feature snapshots from {SNAPSHOT_PATH}")
    if not snaps:
        print("\nNo snapshots yet. The FeatureRecorder writes one per hour while the")
        print("bot is running. Come back in a few days.")
        return

    # Group by ticker — for training, use the LAST snapshot per ticker (closest to resolution)
    by_ticker: dict[str, dict] = {}
    for r in snaps:
        t = r.get("ticker", "")
        if not t: continue
        if t not in by_ticker or r.get("snapshot_ts", 0) > by_ticker[t].get("snapshot_ts", 0):
            by_ticker[t] = r
    print(f"  {len(by_ticker)} unique tickers")

    print(f"\nFetching settlement status for {len(by_ticker)} tickers (this may take a few minutes)…")
    outcomes = fetch_outcomes(list(by_ticker.keys()))
    settled_count = sum(1 for v in outcomes.values() if v in ("yes","no"))
    print(f"  {settled_count} are settled with yes/no outcome")

    if args.report_only or settled_count < 50:
        if settled_count < 50:
            print(f"\nNeed ≥50 settled samples to train a meaningful model. Have {settled_count}.")
            print("Keep the bot running (FeatureRecorder snapshots every hour) and re-run later.")
        return

    # Build training matrix
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, brier_score_loss
    except ImportError:
        print("Install scikit-learn: pip install scikit-learn")
        return

    X, y = [], []
    for t, snap in by_ticker.items():
        outcome = outcomes.get(t)
        if outcome not in ("yes","no"): continue
        row = [float(snap.get(k, 0)) for k in FEATURE_KEYS]
        X.append(row); y.append(1 if outcome == "yes" else 0)
    X = np.array(X); y = np.array(y)
    print(f"\nTraining matrix: {X.shape}  (label rate yes={y.mean()*100:.1f}%)")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(Xtr, ytr)

    p_train = model.predict_proba(Xtr)[:, 1]
    p_test  = model.predict_proba(Xte)[:, 1]
    print(f"\nTrain: AUC={roc_auc_score(ytr, p_train):.3f}  Brier={brier_score_loss(ytr, p_train):.3f}")
    print(f"Test:  AUC={roc_auc_score(yte, p_test):.3f}  Brier={brier_score_loss(yte, p_test):.3f}")
    print(f"\nFeature weights:")
    for fk, w in sorted(zip(FEATURE_KEYS, model.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {fk:18s}  {w:+.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feature_order": FEATURE_KEYS}, f)
    print(f"\nSaved → {MODEL_PATH}")
    print("Strategy will auto-load this on next bot restart.")


if __name__ == "__main__":
    main()
