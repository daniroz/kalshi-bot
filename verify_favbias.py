"""
Favorite-Longshot Bias verdict tool.

The whole reason favbias logs `implied_p=X` on every entry is so we can later
ask the only question that matters: did the favorites actually win as often
as their price implied?

This script:
  1. Scans bot.log for every `[favbias] ENTER` line
  2. Looks up each ticker's settlement on Kalshi
  3. For settled ones, compares predicted P(win) (= entry price as decimal)
     against actual outcome (won YES = bet correct)
  4. Computes implied vs actual win rate, EV per contract, total P&L
  5. Gives a verdict:
        KEEP if actual win rate >= implied (bias real on Kalshi)
        KILL if actual win rate < implied by >5% over ≥30 settled trades

Usage:
    python verify_favbias.py
    python verify_favbias.py --csv favbias_results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


# `[favbias] ENTER KXNBA-... YES x40 @ 89¢  implied_p=0.89 edge=2.4¢  Title`
ENTRY_RE = re.compile(
    r"\[favbias\]\s+ENTER\s+(?P<ticker>[A-Z0-9\-\.]+)\s+"
    r"(?P<side>YES|NO)\s+"
    r"x\s*(?P<qty>\d+)\s+@\s+(?P<price>\d+)¢"
    r".*?implied_p=(?P<implied>[\d\.]+)"
)


@dataclass
class FavBet:
    ticker:  str
    side:    str           # "YES" or "NO"
    qty:     int
    price_c: int           # what we paid in cents
    implied: float         # entry price as decimal — the model's predicted P(win)
    # Filled by Kalshi lookup
    status:  str = "unknown"
    result:  str = ""      # "yes" / "no" / ""
    won:     bool = False
    pnl_c:   float = 0.0   # per contract


def _fee_c(price_c: int) -> float:
    p = price_c / 100
    return 0.07 * p * (1 - p) * 100


def parse_log(log_path: Path) -> list[FavBet]:
    bets: list[FavBet] = []
    if not log_path.exists():
        return bets
    with open(log_path, errors="replace") as f:
        for line in f:
            m = ENTRY_RE.search(line)
            if not m:
                continue
            try:
                bets.append(FavBet(
                    ticker  = m.group("ticker"),
                    side    = m.group("side"),
                    qty     = int(m.group("qty")),
                    price_c = int(m.group("price")),
                    implied = float(m.group("implied")),
                ))
            except (ValueError, TypeError):
                continue
    return bets


def lookup_settlements(bets: list[FavBet]) -> None:
    from clients.kalshi import KalshiClient
    k = KalshiClient(
        api_key=os.environ["KALSHI_API_KEY"],
        key_file=str(Path(__file__).parent / "kalshi_private.pem"),
        demo=False,
    )
    seen: dict[str, dict] = {}
    for b in bets:
        info = seen.get(b.ticker)
        if info is None:
            try:
                r = k.get_market(b.ticker)
                m = r.get("market", {})
                info = {"status": m.get("status",""), "result": (m.get("result") or "").lower()}
            except Exception:
                info = {"status": "lookup_failed", "result": ""}
            seen[b.ticker] = info
        b.status = info["status"]
        b.result = info["result"]
        if b.status == "settled" and b.result in ("yes","no"):
            b.won = (b.side.lower() == b.result)
            fee  = _fee_c(b.price_c)
            b.pnl_c = ((100 - b.price_c) - fee) if b.won else -(b.price_c + fee)


def main():
    p = argparse.ArgumentParser(description="Favorite-Longshot Bias verdict")
    p.add_argument("--log", default=str(Path(__file__).parent / "bot.log"))
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    log_path = Path(args.log)
    print(f"Scanning {log_path}…")
    bets = parse_log(log_path)
    if not bets:
        print("No [favbias] ENTER lines found yet. Bot hasn't entered any favorites.")
        print("(This is normal — the strategy needs liquid markets with 85-94¢ asks)")
        return
    print(f"  Found {len(bets)} total favbias entries")

    # De-dup by (ticker, qty, price) — log entries can repeat across sessions
    uniq = list({(b.ticker, b.qty, b.price_c, b.side): b for b in bets}.values())
    print(f"  {len(uniq)} unique trades after dedup")

    print(f"\nLooking up settlement status for {len(set(b.ticker for b in uniq))} tickers…")
    lookup_settlements(uniq)

    settled = [b for b in uniq if b.status == "settled" and b.result]
    open_   = [b for b in uniq if b.status not in ("settled",)]
    print(f"  Settled: {len(settled)}    Still open / unknown: {len(open_)}")

    if not settled:
        print("\nNo settled favbias trades yet. Run again in a few days.")
        return

    # Implied vs actual win rate
    sum_implied = sum(b.implied for b in settled)
    actual_wins = sum(1 for b in settled if b.won)
    implied_win_rate = sum_implied / len(settled)
    actual_win_rate  = actual_wins / len(settled)

    # P&L
    total_pnl_usd = sum(b.pnl_c * b.qty / 100 for b in settled)
    avg_pnl_per_contract = sum(b.pnl_c for b in settled) / len(settled)

    # By price bucket
    buckets: dict[str, list[FavBet]] = defaultdict(list)
    for b in settled:
        if b.price_c <= 87:  buckets["85-87¢"].append(b)
        elif b.price_c <= 90: buckets["88-90¢"].append(b)
        else:                 buckets["91-94¢"].append(b)

    print(f"\n{'='*72}")
    print(f"FAVBIAS REPORT — {len(settled)} settled trades")
    print(f"{'='*72}")
    print(f"  Implied win rate (avg price): {implied_win_rate*100:>5.1f}%")
    print(f"  Actual win rate:              {actual_win_rate*100:>5.1f}%  ({actual_wins}/{len(settled)})")
    gap = (actual_win_rate - implied_win_rate) * 100
    print(f"  Gap:                          {gap:>+5.1f}%   "
          f"{'(actual ≥ implied → bias confirmed)' if gap >= 0 else '(actual < implied → bias not present)'}")
    print()
    print(f"  Total P&L (settled):          ${total_pnl_usd:>+7.2f}")
    print(f"  Avg P&L per contract:         {avg_pnl_per_contract:>+5.2f}¢")
    print()
    print(f"  By price bucket:")
    print(f"    {'bucket':10s} {'n':>4} {'implied':>8} {'actual':>8} {'pnl/ct':>8}")
    for label in ("85-87¢","88-90¢","91-94¢"):
        if label not in buckets: continue
        bs = buckets[label]
        imp = sum(b.implied for b in bs)/len(bs)
        act = sum(1 for b in bs if b.won)/len(bs)
        pnl = sum(b.pnl_c for b in bs)/len(bs)
        print(f"    {label:10s} {len(bs):>4} {imp*100:>7.1f}% {act*100:>7.1f}% {pnl:>+7.2f}¢")

    # ── Verdict ──
    print(f"\n{'─'*72}")
    if len(settled) < 30:
        print(f"VERDICT: not enough data yet ({len(settled)}/30 settled trades).")
        print(f"          Run again after more favbias entries settle.")
    elif actual_win_rate >= implied_win_rate - 0.02:
        print(f"VERDICT: ✅ KEEP — bias confirmed on Kalshi.")
        print(f"          Actual win rate is within 2pp of implied (or above).")
        if total_pnl_usd > 0:
            print(f"          Strategy is also profitable: ${total_pnl_usd:+.2f}")
    elif actual_win_rate < implied_win_rate - 0.05:
        print(f"VERDICT: ❌ KILL — favorites are not winning as often as priced.")
        print(f"          Actual win rate is >5pp below implied. Bias not present on Kalshi.")
        print(f"          → Set favorite_bias.enabled: false in config.yaml")
    else:
        print(f"VERDICT: ⚠️  AMBIGUOUS — actual win rate is 2-5pp below implied.")
        print(f"          Wait for more settled trades or check by bucket above.")

    if args.csv:
        out = Path(__file__).parent / "backtest_results" / args.csv
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ticker","side","qty","price_c","implied",
                                              "status","result","won","pnl_c"])
            w.writeheader()
            for b in uniq:
                w.writerow({"ticker":b.ticker,"side":b.side,"qty":b.qty,
                            "price_c":b.price_c,"implied":b.implied,
                            "status":b.status,"result":b.result,
                            "won":b.won,"pnl_c":round(b.pnl_c,2)})
        print(f"\nDetailed CSV: {out}")


if __name__ == "__main__":
    main()
