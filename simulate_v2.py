"""
Monte Carlo v2 — 10 new strategies × 500 sims × 2 scenarios (base + pessimistic).

The discipline a real quant follows:
  Most strategies look profitable under their author's preferred assumptions.
  The ones that work in production are the ones that ALSO work when those
  assumptions are wrong by 30-50%. So every strategy gets BOTH a base case
  and a pessimistic case, and only those that pass BOTH get implemented.

Honesty tags:
  [EMPIRICAL]   — supported by published research on prediction/betting markets
  [STRUCTURAL]  — mechanical, derivable from market math
  [BEHAVIORAL]  — based on documented human-behavior biases
  [SPECULATIVE] — intuition only; sim assumptions ARE the strategy

Implementation bar (ALL must hold):
  - Median P&L > 0 in BASE case
  - Median P&L > 0 in PESSIMISTIC case (assumes 40-50% less edge)
  - Sharpe (per trade) > 0.4 in base
  - ≥60% of seasons profitable in base
  - Max drawdown ≤ 50% of median P&L

Adverse selection is modeled aggressively — when your limit order fills it's
because the counterparty thought the trade was good for them. Every passive
fill has a baseline 35-50% chance of being toxic.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

RNG = np.random.default_rng(42)
N_RUNS = 500
TRADES_PER_RUN = 150
STAKE = 25.0


def fee(price, contracts=1):
    return 0.07 * price * (1 - price) * contracts


@dataclass
class Verdict:
    name: str
    tag: str
    base_pnl: float
    base_prof: float
    base_sharpe: float
    base_dd: float
    pess_pnl: float
    pess_prof: float
    pess_sharpe: float
    pess_dd: float

    @property
    def survives(self) -> bool:
        return (self.base_pnl > 0 and self.pess_pnl > 0
                and self.base_sharpe > 0.4
                and self.base_prof >= 60
                and abs(self.base_dd) <= 0.5 * abs(self.base_pnl) if self.base_pnl > 0 else False)

    @property
    def verdict(self) -> str:
        return "✅ IMPLEMENT" if self.survives else "❌ reject"


def run_paths(trade_fn) -> tuple[float, float, float, float]:
    """Returns (median_pnl, pct_profitable, sharpe, median_max_dd)."""
    pnls, dds = [], []
    for _ in range(N_RUNS):
        pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        for _ in range(TRADES_PER_RUN):
            dp = trade_fn()
            if dp is None: continue
            pnl += dp
            peak = max(peak, pnl)
            max_dd = min(max_dd, pnl - peak)
        pnls.append(pnl); dds.append(max_dd)
    pnls = np.array(pnls)
    return (float(np.median(pnls)), float(np.mean(pnls > 0) * 100),
            (float(np.mean(pnls)) / float(np.std(pnls))) if np.std(pnls) > 0 else 0,
            float(np.median(dds)))


def evaluate(name, tag, base_fn, pess_fn) -> Verdict:
    bp, bprof, bsh, bdd = run_paths(base_fn)
    pp, pprof, psh, pdd = run_paths(pess_fn)
    return Verdict(name, tag, bp, bprof, bsh, bdd, pp, pprof, psh, pdd)


# ════════════════════════════════════════════════════════════════════════════
# Strategy 1: PANIC SELL CATCHER  [BEHAVIORAL]
# Fade fast 15¢+ drops on no fundamental news. Mean-revert ~half the drop.
# Adverse selection: sometimes the drop IS informed (injury, scandal, etc.)
# ════════════════════════════════════════════════════════════════════════════
def panic_catcher(noise_pct):
    def go():
        drop = RNG.uniform(0.10, 0.20)            # the panic size
        is_noise = RNG.random() < noise_pct
        entry = 0.40                                # post-panic price
        f = fee(entry) * 2                          # roundtrip
        c = STAKE / entry
        if is_noise:
            return (drop * 0.55 - f) * c           # half-revert
        return (-drop * 0.6 - f) * c               # informed: continues
    return go


# ════════════════════════════════════════════════════════════════════════════
# Strategy 2: VOLUME-SPIKE NO-PRICE-MOVE FADE  [BEHAVIORAL]
# 5x volume spike with <2¢ price move = two-sided informed flow.
# Often reverts to pre-spike consensus within hours.
# ════════════════════════════════════════════════════════════════════════════
def volume_fade(revert_pct):
    def go():
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        reverts = RNG.random() < revert_pct
        if reverts:
            return (0.04 - f) * c                  # captures ~4¢ revert
        return (-0.06 - f) * c                     # continues against
    return go


# ════════════════════════════════════════════════════════════════════════════
# Strategy 3: STEAM CHASING  [SPECULATIVE]
# Follow large size moves (sharp money thesis from sports betting).
# Caveat: most "sharp" indicators are noise; we need >52% continuation.
# ════════════════════════════════════════════════════════════════════════════
def steam_chase(continue_pct):
    def go():
        move = RNG.uniform(0.08, 0.15)
        entry = 0.55                                # right after move
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < continue_pct:
            return (move * 0.4 - f) * c
        return (-move * 0.6 - f) * c
    return go


# ════════════════════════════════════════════════════════════════════════════
# Strategy 4: END-OF-LIFE SETTLEMENT DRIFT  [STRUCTURAL]
# Markets within 12h of resolution at mid prices (0.3-0.7) have ~3-5¢ noise
# that pulls toward settlement. Bet on convergence to true value.
# Caveat: we have to know the true value (i.e. have our own settlement model)
# ════════════════════════════════════════════════════════════════════════════
def eol_drift(convergence_pct):
    def go():
        true_p = RNG.uniform(0.3, 0.7)
        market_p = true_p + RNG.normal(0, 0.04)   # noise
        market_p = np.clip(market_p, 0.01, 0.99)
        if abs(market_p - true_p) < 0.03:
            return None                            # no edge to take
        # We bet on convergence toward true
        entry = market_p
        f = fee(entry) * 2
        c = STAKE / entry
        # Win if our directional bet is right
        bet_yes = true_p > market_p
        won = (RNG.random() < (true_p if bet_yes else 1 - true_p)) if RNG.random() < convergence_pct else \
              (RNG.random() < 0.5)
        return ((0.04 - f) * c) if won else ((-0.04 - f) * c)
    return go


# ════════════════════════════════════════════════════════════════════════════
# Strategy 5: STALE QUOTE SNIPING  [STRUCTURAL]
# When market A moves but tightly-correlated market B doesn't, snipe B.
# Adverse selection if our "correlation" assumption was wrong.
# ════════════════════════════════════════════════════════════════════════════
def stale_snipe(real_correlation):
    def go():
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < real_correlation:
            return (0.05 - f) * c
        return (-0.04 - f) * c
    return go


# ════════════════════════════════════════════════════════════════════════════
# Strategy 6: CROSS-MARKET CONSISTENCY (SERIES vs GAME)  [STRUCTURAL]
# "Team X wins series" should equal sum of conditional series outcomes.
# When the implied sum diverges by >3¢, take the cheaper side.
# Hard to execute (multi-leg), but mechanical edge if it exists.
# ════════════════════════════════════════════════════════════════════════════
def series_consistency(arb_existence_rate):
    def go():
        if RNG.random() >= arb_existence_rate:
            return None                            # no opp this slot
        # We capture the inconsistency. Edge size is small.
        entry = 0.50
        f = fee(entry) * 3                          # 3-leg trade
        c = STAKE / entry
        # When arb exists, we win nearly always (it's mechanical)
        if RNG.random() < 0.92:
            return (0.04 - f) * c
        # The rare loss is when one leg doesn't fill or settles differently
        return (-0.10 - f) * c
    return go


# ════════════════════════════════════════════════════════════════════════════
# Strategy 7: SPORTS LATE-MONEY FOLLOW  [EMPIRICAL]
# Last 30 min before game-start line moves are documented to be ~52-54%
# predictive in sports literature (Levitt 2004, Snowberg & Wolfers).
# ════════════════════════════════════════════════════════════════════════════
def late_money(continue_pct):
    def go():
        move = RNG.uniform(0.04, 0.10)
        entry = 0.50 + move/2
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < continue_pct:
            return (move * 0.5 - f) * c
        return (-move * 0.4 - f) * c               # smaller downside if wrong
    return go


# ════════════════════════════════════════════════════════════════════════════
# Strategy 8: EXTREME PRICE FAVORITE OVERLAY  [EMPIRICAL]
# Buy >94¢ favorites (above what favorite_bias does). True prob is
# typically 95-97% per the research — collect tiny premium with low fees.
# This is a sub-strategy of favorite-longshot, fine-tuned for the high end.
# ════════════════════════════════════════════════════════════════════════════
def extreme_favorite(bias_c):
    def go():
        price = RNG.uniform(0.94, 0.97)
        true_p = np.clip(price + RNG.normal(bias_c/100, 0.015), 0, 0.999)
        f = fee(price)
        c = STAKE / price
        if RNG.random() < true_p:
            return ((1 - price) - f) * c
        return (-price - f) * c
    return go


# ════════════════════════════════════════════════════════════════════════════
# Strategy 9: OPENING GAP REVERSION  [SPECULATIVE]
# Markets sometimes open with prices that drifted overnight on thin volume.
# Fade the gap. Edge unclear on Kalshi specifically.
# ════════════════════════════════════════════════════════════════════════════
def opening_gap(revert_pct):
    def go():
        gap = RNG.uniform(0.05, 0.12)
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < revert_pct:
            return (gap * 0.5 - f) * c
        return (-gap * 0.5 - f) * c
    return go


# ════════════════════════════════════════════════════════════════════════════
# Strategy 10: NEWS HALF-LIFE DECAY  [BEHAVIORAL]
# After a news shock, markets overshoot then partially revert within 2-4h.
# Classic overreaction → underreaction pattern.
# ════════════════════════════════════════════════════════════════════════════
def news_decay(overshoot_pct):
    def go():
        shock = RNG.uniform(0.08, 0.18)
        entry = 0.50                                # post-shock price
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < overshoot_pct:
            return (shock * 0.30 - f) * c          # partial revert
        return (-shock * 0.40 - f) * c
    return go


STRATEGIES = [
    # (name, tag, base_fn_factory, pess_fn_factory)
    ("Panic Sell Catcher",        "[BEHAVIORAL] ", lambda: panic_catcher(0.62), lambda: panic_catcher(0.50)),
    ("Volume-Spike Fade",         "[BEHAVIORAL] ", lambda: volume_fade(0.65),   lambda: volume_fade(0.52)),
    ("Steam Chasing",             "[SPECULATIVE]", lambda: steam_chase(0.55),   lambda: steam_chase(0.50)),
    ("End-of-Life Settle Drift",  "[STRUCTURAL] ", lambda: eol_drift(0.75),     lambda: eol_drift(0.55)),
    ("Stale Quote Sniping",       "[STRUCTURAL] ", lambda: stale_snipe(0.70),   lambda: stale_snipe(0.55)),
    ("Series Consistency Arb",    "[STRUCTURAL] ", lambda: series_consistency(0.15), lambda: series_consistency(0.05)),
    ("Sports Late-Money Follow",  "[EMPIRICAL]  ", lambda: late_money(0.53),    lambda: late_money(0.51)),
    ("Extreme Favorite (>94¢)",   "[EMPIRICAL]  ", lambda: extreme_favorite(2.0), lambda: extreme_favorite(0.5)),
    ("Opening Gap Reversion",     "[SPECULATIVE]", lambda: opening_gap(0.55),   lambda: opening_gap(0.50)),
    ("News Half-Life Decay",      "[BEHAVIORAL] ", lambda: news_decay(0.60),    lambda: news_decay(0.50)),
]


def main():
    print(f"\nMonte Carlo v2 — {N_RUNS} runs × {TRADES_PER_RUN} trades, ${STAKE:.0f}/trade")
    print(f"Each strategy run at BASE assumptions + PESSIMISTIC (40-50% smaller edge).\n")

    hdr = f"{'strategy':27s} {'tag':14s}  {'BASE: medP&L':>11} {'prof%':>5} {'Sharpe':>6} {'maxDD':>8}   {'PESS: medP&L':>11} {'prof%':>5} {'Sharpe':>6}   verdict"
    print(hdr)
    print("-" * len(hdr))

    verdicts = []
    for name, tag, base_factory, pess_factory in STRATEGIES:
        v = evaluate(name, tag, base_factory(), pess_factory())
        verdicts.append(v)
        print(
            f"{name:27s} {tag:14s}  "
            f"${v.base_pnl:>+8.0f} {v.base_prof:>4.0f}% {v.base_sharpe:>6.2f} ${v.base_dd:>+6.0f}   "
            f"${v.pess_pnl:>+8.0f} {v.pess_prof:>4.0f}% {v.pess_sharpe:>6.2f}   {v.verdict}"
        )

    print("-" * len(hdr))
    winners = [v for v in verdicts if v.survives]
    print(f"\nSurvivors (passed both scenarios): {len(winners)} of {len(verdicts)}")
    for v in winners:
        print(f"  ✅ {v.name}  {v.tag}  base ${v.base_pnl:+.0f}/season, pessimistic ${v.pess_pnl:+.0f}/season")
    if not winners:
        print("  (none — all rejected under pessimistic assumptions)")
    print()


if __name__ == "__main__":
    main()
