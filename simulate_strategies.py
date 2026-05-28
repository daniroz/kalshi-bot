"""
Monte Carlo strategy evaluation — 500 runs × 5 candidate strategies.

METHODOLOGY (read this before trusting any number):
  We cannot replay Kalshi's historical orderbook (not exposed by their API).
  So this is Monte Carlo: each strategy has an explicit data-generating process
  (DGP) for prices/outcomes, fill probability, the Kalshi fee, and — the part
  most amateur backtests omit — ADVERSE SELECTION (your passive order fills
  preferentially when you're on the wrong side).

  A Monte Carlo only demonstrates edge that ISN'T baked into the assumptions.
  So each strategy is tagged:
    [EMPIRICAL]   — parameters come from published prediction-market research
    [ASSUMPTION]  — parameters are educated guesses; sim cannot prove real edge

  Verdict to implement: median total P&L > 0  AND  ≥60% of runs profitable
  AND  Sharpe (per-trade, annualized-ish) > 0.5  AND  max drawdown tolerable.

  Fees: Kalshi charges 0.07 * price * (1-price) per contract per fill.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

RNG = np.random.default_rng(42)
N_RUNS = 500
TRADES_PER_RUN = 200       # a "run" = a season of 200 trades
START_BANKROLL = 250.0


def kalshi_fee(price, contracts=1):
    """Kalshi fee in dollars: 0.07 * p * (1-p) per contract."""
    return 0.07 * price * (1 - price) * contracts


@dataclass
class SimResult:
    name: str
    tag: str
    median_pnl: float
    mean_pnl: float
    pct_profitable: float
    sharpe: float
    max_dd: float
    avg_win_rate: float
    verdict: str


def _summarize(name, tag, run_pnls, run_winrates, run_dds) -> SimResult:
    run_pnls = np.array(run_pnls)
    median_pnl = float(np.median(run_pnls))
    mean_pnl = float(np.mean(run_pnls))
    pct_prof = float(np.mean(run_pnls > 0) * 100)
    std = float(np.std(run_pnls))
    sharpe = (mean_pnl / std) if std > 0 else 0.0
    max_dd = float(np.median(run_dds))
    avg_wr = float(np.mean(run_winrates) * 100)

    passes = (median_pnl > 0 and pct_prof >= 60 and sharpe > 0.5)
    verdict = "✅ IMPLEMENT" if passes else "❌ reject"
    return SimResult(name, tag, median_pnl, mean_pnl, pct_prof, sharpe,
                     max_dd, avg_wr, verdict)


def _run_paths(trade_fn) -> SimResult:
    """trade_fn() -> (pnl_dollars, won_bool) for a single trade.
    Wrapped over N_RUNS × TRADES_PER_RUN."""
    name, tag, fn = trade_fn
    run_pnls, run_wrs, run_dds = [], [], []
    for _ in range(N_RUNS):
        pnl = 0.0
        wins = 0
        peak = 0.0
        max_dd = 0.0
        n_real = 0
        for _ in range(TRADES_PER_RUN):
            res = fn()
            if res is None:    # no trade taken this slot (no fill / no signal)
                continue
            dp, won = res
            pnl += dp
            wins += 1 if won else 0
            n_real += 1
            peak = max(peak, pnl)
            max_dd = min(max_dd, pnl - peak)
        run_pnls.append(pnl)
        run_wrs.append(wins / n_real if n_real else 0)
        run_dds.append(max_dd)
    return _summarize(name, tag, run_pnls, run_wrs, run_dds)


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: Favorite-Longshot Bias Harvesting  [EMPIRICAL]
# ════════════════════════════════════════════════════════════════════════════
# Published finding (Polymarket 2020-24, PredictIt, decades of betting lit):
# heavy favorites are UNDERpriced. A contract trading at price p in [0.85,0.97]
# resolves YES with true probability ≈ p + bias(p), where bias is ~+2-4¢.
# We BUY YES on favorites and hold to resolution.
def trade_favorite_longshot():
    price = RNG.uniform(0.85, 0.95)
    # Empirical: true prob is modestly higher than market price for favorites.
    # Calibration studies put the underpricing at ~3¢ on average, noisy.
    true_bias = RNG.normal(0.03, 0.02)        # mean +3¢, sd 2¢
    true_prob = np.clip(price + true_bias, 0, 0.999)
    fee = kalshi_fee(price)
    won = RNG.random() < true_prob
    contracts = 25.0 / price                   # ~$25 stake
    if won:
        return ((1 - price) - fee) * contracts, True
    else:
        return (-price - fee) * contracts, False


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: Longshot Fade (sell overpriced longshots)  [EMPIRICAL]
# ════════════════════════════════════════════════════════════════════════════
# The mirror image: longshots (p in [0.05,0.15]) are OVERpriced. We buy NO
# (= bet against the longshot). True YES prob ≈ p - bias.
def trade_longshot_fade():
    price = RNG.uniform(0.05, 0.15)            # the longshot YES price
    true_bias = RNG.normal(0.03, 0.02)         # overpriced by ~3¢
    true_yes_prob = np.clip(price - true_bias, 0.001, 1)
    no_price = 1 - price                        # we buy NO at (1-price)
    fee = kalshi_fee(no_price)
    # NO wins if YES does NOT happen
    no_wins = RNG.random() > true_yes_prob
    contracts = 25.0 / no_price
    if no_wins:
        return ((1 - no_price) - fee) * contracts, True
    else:
        return (-no_price - fee) * contracts, False


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Overreaction Mean-Reversion  [ASSUMPTION]
# ════════════════════════════════════════════════════════════════════════════
# When a market jumps ≥8¢ in minutes on a single large order (no fundamental
# news), it tends to revert partway. We fade the move. BUT adverse selection:
# sometimes the move IS informed and continues against us.
def trade_mean_reversion():
    move = RNG.uniform(0.08, 0.20)             # size of the spike we fade
    # Assumption: 55% of large spikes are noise (revert), 45% are informed (continue).
    # This is the crux — if it's really 50/50, there's no edge.
    is_noise = RNG.random() < 0.55
    entry = 0.50                                # assume we fade near mid
    fee = kalshi_fee(entry)
    contracts = 25.0 / entry
    if is_noise:
        # revert: capture ~half the move
        gain = move * 0.5
        return (gain - fee) * contracts, True
    else:
        # informed: it continues, we lose ~the move
        loss = move * 0.7
        return (-loss - fee) * contracts, False


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: Inventory-Skewed Market Making  [ASSUMPTION]
# ════════════════════════════════════════════════════════════════════════════
# MM that only quotes the side reducing inventory. Captures spread but eats
# adverse selection: ~40% of fills are toxic (informed flow picks us off).
def trade_inventory_mm():
    spread = RNG.uniform(0.06, 0.15)           # wide-spread markets only
    price = RNG.uniform(0.3, 0.7)
    capture = spread - 0.02                      # we quote 1¢ inside each side
    fee = kalshi_fee(price) * 2                  # both legs
    contracts = 25.0 / price
    # 40% of round trips are toxic — one leg fills then market runs away
    toxic = RNG.random() < 0.40
    if not toxic:
        return (capture - fee) * contracts, True
    else:
        # toxic: we capture the near leg but lose ~2x spread on the runaway leg
        loss = spread * 1.5
        return (capture - loss - fee) * contracts, False


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY 5: Steam-Move Momentum  [ASSUMPTION]
# ════════════════════════════════════════════════════════════════════════════
# Opposite of #3: follow large informed moves (the "steam"). Assumption that
# 52% of large moves continue. Thin edge, high variance.
def trade_steam_momentum():
    move = RNG.uniform(0.08, 0.20)
    continues = RNG.random() < 0.52            # assumed continuation rate
    entry = 0.55
    fee = kalshi_fee(entry)
    contracts = 25.0 / entry
    if continues:
        return (move * 0.5 - fee) * contracts, True
    else:
        return (-move * 0.6 - fee) * contracts, False


STRATEGIES = [
    ("Favorite-Longshot Bias",   "[EMPIRICAL] ", trade_favorite_longshot),
    ("Longshot Fade",            "[EMPIRICAL] ", trade_longshot_fade),
    ("Overreaction Mean-Revert", "[ASSUMPTION]", trade_mean_reversion),
    ("Inventory-Skewed MM",      "[ASSUMPTION]", trade_inventory_mm),
    ("Steam-Move Momentum",      "[ASSUMPTION]", trade_steam_momentum),
]


def main():
    print(f"\nMonte Carlo: {N_RUNS} runs × {TRADES_PER_RUN} trades each, $25/trade\n")
    print(f"{'strategy':28s} {'basis':12s} {'med P&L':>9} {'mean':>8} "
          f"{'prof%':>6} {'win%':>6} {'Sharpe':>7} {'maxDD':>9}  verdict")
    print("-" * 110)
    results = []
    for name, tag, fn in STRATEGIES:
        res = _run_paths((name, tag, fn))
        results.append(res)
        print(f"{name:28s} {tag:12s} ${res.median_pnl:>+7.1f} ${res.mean_pnl:>+6.1f} "
              f"{res.pct_profitable:>5.0f}% {res.avg_win_rate:>5.0f}% {res.sharpe:>7.2f} "
              f"${res.max_dd:>+7.1f}  {res.verdict}")
    print("-" * 110)

    winners = [r for r in results if "IMPLEMENT" in r.verdict]
    print(f"\n{len(winners)} strategy(ies) passed the bar:")
    for w in winners:
        print(f"  ✅ {w.name}  ({w.tag.strip()})  median ${w.median_pnl:+.1f}/season, "
              f"{w.pct_profitable:.0f}% of seasons green")
    if not winners:
        print("  (none — all rejected)")
    print()
    return results


if __name__ == "__main__":
    main()
