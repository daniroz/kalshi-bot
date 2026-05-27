"""
Per-strategy P&L from the dashboard's already-correct per-position numbers.

The dashboard's /api/snapshot endpoint already does the hard work:
  - Pulls open positions with mark-to-market values
  - Computes unrealized_pnl per position using avg cost basis from fills

We just call it, bucket by ticker pattern, and aggregate. No reinventing.

Ticker → strategy mapping:
  KXINX, KXNDAQ, KXSPXCLOSE, KXDJI               → settle-stocks
  KXBTC, KXETH, KXDOGE, KXXRP                    → settle-crypto
  KXHIGH*, KXLOW*                                → settle-weather
  KXWTI, KXGOLD, KXNATGAS                        → settle-comm
  KXEURUSD, KXUSDJPY                             → settle-fx
  KXCPI, KXFED, KXGDP, KXUNEMPLOYMENT, KXNFP     → mm-macro
  KXNBAGAME, KXNHLGAME, KXMLBGAME, KXNBASERIES   → sports
  *                                              → other

Usage:
    python measure_pnl.py
    python measure_pnl.py --csv pnl.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import httpx


def bucket_strategy(ticker: str) -> str:
    """Map a Kalshi ticker → likely strategy bucket."""
    prefix = ticker.split("-")[0] if "-" in ticker else ticker
    if prefix.startswith("KXHIGH") or prefix.startswith("KXLOW"):
        return "settle-weather"
    BUCKETS = {
        "settle-stocks":  ("KXINX","KXNDAQ","KXSPXCLOSE","KXDJI"),
        "settle-crypto":  ("KXBTC","KXETH","KXDOGE","KXXRP","KXSOL","KXLTC"),
        "settle-comm":    ("KXWTI","KXGOLD","KXNATGAS","KXSILVER"),
        "settle-fx":      ("KXEURUSD","KXUSDJPY","KXGBPUSD"),
        "mm-macro":       ("KXCPI","KXFED","KXGDP","KXUNEMPLOYMENT","KXNFP","KXFEDDECISION"),
        "sports":         ("KXNBAGAME","KXNHLGAME","KXMLBGAME","KXNBASERIES","KXNHLSERIES",
                          "KXNBASERIESSCORE","KXNHLSERIESSCORE","KXMLB"),
    }
    for strat, prefixes in BUCKETS.items():
        if prefix in prefixes:
            return strat
    return "other"


def main():
    p = argparse.ArgumentParser(description="Per-strategy P&L from the dashboard")
    p.add_argument("--url", default="http://localhost:5555/api/snapshot")
    p.add_argument("--csv", default=None, help="write detailed CSV here")
    args = p.parse_args()

    try:
        r = httpx.get(args.url, timeout=10)
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        print(f"Failed to read dashboard at {args.url}: {e}")
        print("Make sure dashboard.py is running.")
        return

    open_positions = d.get("open_positions", [])
    fills          = d.get("fills", [])

    # ── Unrealized P&L by strategy (from open positions) ────────────────────
    by_strat = defaultdict(lambda: {
        "positions": 0, "cost": 0.0, "value": 0.0, "unrealized": 0.0,
        "with_basis": 0,
    })
    rows = []
    for p in open_positions:
        ticker = p["ticker"]
        strat  = bucket_strategy(ticker)
        qty    = p.get("qty", 0)
        entry  = p.get("entry_c") or 0          # may be None
        mark   = p.get("mark_c") or 0
        val    = p.get("val") or 0
        upnl   = p.get("unrealized_pnl")

        cost_basis = (entry / 100) * qty if entry else 0
        s = by_strat[strat]
        s["positions"]  += 1
        s["cost"]       += cost_basis
        s["value"]      += val
        if upnl is not None:
            s["unrealized"]  += upnl
            s["with_basis"]  += 1
        rows.append({
            "ticker": ticker, "strategy": strat, "side": p.get("side",""),
            "qty": qty, "entry_c": entry, "mark_c": mark, "value": val,
            "unrealized_pnl": upnl,
        })

    # ── Realized P&L from recent fills (SELL legs with computed pnl) ────────
    realized_by_strat = defaultdict(lambda: {"sells": 0, "realized": 0.0})
    for f in fills:
        if f.get("action") != "SELL": continue
        pnl = f.get("pnl")
        if pnl is None: continue
        strat = bucket_strategy(f.get("ticker",""))
        realized_by_strat[strat]["sells"]    += 1
        realized_by_strat[strat]["realized"] += pnl

    # ── Print ───────────────────────────────────────────────────────────────
    equity   = d.get("equity", 0)
    cash     = d.get("cash", 0)
    pos_val  = d.get("pos_val", 0)
    today_p  = d.get("today_pnl", 0)
    alltime  = d.get("alltime_pnl", 0)

    print(f"\nDashboard snapshot:  Equity=${equity:.2f}  Cash=${cash:.2f}  Open=${pos_val:.2f}")
    print(f"  Today P&L:    ${today_p:+.2f}    All-time: ${alltime:+.2f}")
    print(f"\n{'='*94}")
    print(f"{'strategy':17s} {'positions':>9} {'with basis':>11} {'cost':>10} "
          f"{'mark val':>10} {'unrealized':>12} {'sells':>6} {'realized':>11}")
    print("-" * 94)
    totals = defaultdict(float)
    all_strats = set(by_strat) | set(realized_by_strat)
    sortkey = lambda x: -(by_strat[x]["unrealized"] + realized_by_strat[x]["realized"])
    for strat in sorted(all_strats, key=sortkey):
        s  = by_strat.get(strat, {"positions":0,"cost":0,"value":0,"unrealized":0,"with_basis":0})
        rs = realized_by_strat.get(strat, {"sells":0,"realized":0})
        print(f"{strat:17s} {s['positions']:>9} {s['with_basis']:>11} ${s['cost']:>9.2f} "
              f"${s['value']:>9.2f} ${s['unrealized']:>+11.2f} {rs['sells']:>6} ${rs['realized']:>+10.2f}")
        totals["pos"]   += s["positions"]
        totals["cost"]  += s["cost"]
        totals["val"]   += s["value"]
        totals["upnl"]  += s["unrealized"]
        totals["sells"] += rs["sells"]
        totals["real"]  += rs["realized"]
    print("-" * 94)
    print(f"{'TOTAL':17s} {int(totals['pos']):>9} {'':>11} ${totals['cost']:>9.2f} "
          f"${totals['val']:>9.2f} ${totals['upnl']:>+11.2f} {int(totals['sells']):>6} "
          f"${totals['real']:>+10.2f}")

    if args.csv:
        out = Path(__file__).parent / "backtest_results" / args.csv
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nDetailed CSV: {out}")


if __name__ == "__main__":
    main()
