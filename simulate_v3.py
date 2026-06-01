"""
Monte Carlo v3 — 10 MORE strategies × 500 sims × 2 scenarios.

After v2 rejected 10 of 10, we keep fishing. The discipline stays: every idea
gets tested under BASE + PESSIMISTIC assumptions. Only survivors get built.

These 10 are deliberately different from v1/v2 — exploring corners we haven't:
  structural microstructure, basket math, time-of-day patterns, tail premium,
  liquidity provision, pair trading.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

RNG = np.random.default_rng(7)
N_RUNS = 500
TRADES_PER_RUN = 150
STAKE = 25.0


def fee(price, contracts=1):
    return 0.07 * price * (1 - price) * contracts


@dataclass
class V:
    name: str; tag: str
    base_pnl: float; base_prof: float; base_sharpe: float; base_dd: float
    pess_pnl: float; pess_prof: float; pess_sharpe: float; pess_dd: float

    @property
    def survives(self) -> bool:
        if self.base_pnl <= 0 or self.pess_pnl <= 0: return False
        if self.base_sharpe < 0.4: return False
        if self.base_prof < 60: return False
        if abs(self.base_dd) > 0.5 * abs(self.base_pnl): return False
        return True

    @property
    def verdict(self) -> str:
        return "✅ IMPLEMENT" if self.survives else "❌ reject"


def run_paths(trade_fn):
    pnls, dds = [], []
    for _ in range(N_RUNS):
        pnl, peak, dd = 0.0, 0.0, 0.0
        for _ in range(TRADES_PER_RUN):
            dp = trade_fn()
            if dp is None: continue
            pnl += dp; peak = max(peak, pnl); dd = min(dd, pnl - peak)
        pnls.append(pnl); dds.append(dd)
    pnls = np.array(pnls)
    return (float(np.median(pnls)),
            float(np.mean(pnls > 0) * 100),
            (float(np.mean(pnls))/float(np.std(pnls))) if np.std(pnls) > 0 else 0,
            float(np.median(dds)))


def evaluate(name, tag, base_fn, pess_fn) -> V:
    bp, bprof, bsh, bdd = run_paths(base_fn)
    pp, pprof, psh, pdd = run_paths(pess_fn)
    return V(name, tag, bp, bprof, bsh, bdd, pp, pprof, psh, pdd)


# ════════════════════════════════════════════════════════════════════════════
# 1. POWER-LAW RESOLUTION DRIFT  [STRUCTURAL]
# Markets heading to YES typically drift up in the final 6h as info
# accumulates. Buy markets that are >55¢ and trending positive in last 2h.
# ════════════════════════════════════════════════════════════════════════════
def power_drift(continue_pct):
    def go():
        entry = RNG.uniform(0.55, 0.72)
        f = fee(entry)
        c = STAKE / entry
        # Continues to YES (we win full $1)
        if RNG.random() < continue_pct:
            return ((1 - entry) - f) * c
        return (-entry - f) * c
    return go


# ════════════════════════════════════════════════════════════════════════════
# 2. TAIL PREMIUM SELLING  [STRUCTURAL]
# Sell YES at <5¢ (= buy NO at >95¢). The minimum 1¢ tick + fee economics
# distort extreme prices. True P(YES) often lower than market price.
# This is essentially "favorite-bias from the OTHER side."
# ════════════════════════════════════════════════════════════════════════════
def tail_premium(bias_c):
    def go():
        yes_price = RNG.uniform(0.03, 0.05)        # we're SELLING YES (buying NO)
        no_price = 1 - yes_price
        # True P(YES) is slightly LOWER than market — we win as the NO side
        true_yes = max(0, yes_price - bias_c/100 + RNG.normal(0, 0.01))
        f = fee(no_price)
        c = STAKE / no_price
        no_wins = RNG.random() > true_yes
        if no_wins:
            return ((1 - no_price) - f) * c        # we collect ~1-yes_price
        return (-no_price - f) * c
    return go


# ════════════════════════════════════════════════════════════════════════════
# 3. BASKET-SUM IMBALANCE  [STRUCTURAL]
# When N mutually-exclusive contracts sum to <0.95 or >1.05 → arbitrage.
# We saw this didn't really exist on Kalshi when we probed earlier — but
# liquidity-rare cases might. Test what edge it gives IF it exists.
# Only triggers occasionally (sparse opportunities).
# ════════════════════════════════════════════════════════════════════════════
def basket_arb(opp_rate):
    def go():
        if RNG.random() >= opp_rate:
            return None
        # When the basket arb is real, it's mechanically right
        gap = RNG.uniform(0.03, 0.07)
        # N legs, ~5 typical bin market
        N = 5
        f = fee(0.5) * N                            # ~0.5c fee × 2 sides × N legs ≈ N¢
        # The "stake" is the basket cost ≈ $1; profit is the gap minus fees
        contracts_per_leg = STAKE / 1.0             # buy 1 contract each
        profit = (gap - f/100) * contracts_per_leg
        # Tiny risk of one leg not filling
        if RNG.random() < 0.85:
            return profit
        return -0.05 * contracts_per_leg            # leg-failure cost
    return go


# ════════════════════════════════════════════════════════════════════════════
# 4. NEW-MARKET FIRST-HOUR MM  [SPECULATIVE]
# In the first hour a new event is listed, spreads are wide because nobody
# has priced it yet. We provide liquidity aggressively at our model price.
# Edge if our model is half-decent; loss if our model is bad.
# ════════════════════════════════════════════════════════════════════════════
def first_hour_mm(model_accuracy):
    def go():
        # Model is accurate `model_accuracy` of the time; we capture spread
        # 10c spread, we quote 4c inside, capture 2c×2=4c when both fill
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < model_accuracy:
            return (0.04 - f) * c                  # captured spread minus fees
        return (-0.06 - f) * c                     # model was wrong
    return go


# ════════════════════════════════════════════════════════════════════════════
# 5. VWAP REVERSION  [SPECULATIVE]
# When current price diverges from session VWAP by >5%, fade the divergence.
# Volume-weighted "true value" thesis. Adverse selection if VWAP itself was
# uninformed.
# ════════════════════════════════════════════════════════════════════════════
def vwap_reversion(revert_pct):
    def go():
        divergence = RNG.uniform(0.05, 0.10)
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < revert_pct:
            return (divergence * 0.4 - f) * c
        return (-divergence * 0.5 - f) * c
    return go


# ════════════════════════════════════════════════════════════════════════════
# 6. PAIR-TRADE CORRELATED EVENTS  [STRUCTURAL]
# "Team X wins NBA Finals" and "Team X wins Conference" should correlate ~85%.
# When they diverge, buy the cheap one + sell the rich one.
# ════════════════════════════════════════════════════════════════════════════
def pair_trade(correlation_holds_pct):
    def go():
        divergence = RNG.uniform(0.04, 0.10)
        entry = 0.50
        # 4-leg trade (buy + sell on each side)
        f = fee(entry) * 4
        c = STAKE / entry
        if RNG.random() < correlation_holds_pct:
            return (divergence * 0.6 - f) * c
        return (-divergence * 0.7 - f) * c
    return go


# ════════════════════════════════════════════════════════════════════════════
# 7. STATIONARY-MARKET RANGE TRADING  [SPECULATIVE]
# Recurring markets (daily Trump polls, weekly CPI bins) have similar
# distributions. Buy near historical lows, sell near highs.
# Requires the process to actually be stationary.
# ════════════════════════════════════════════════════════════════════════════
def range_trade(stationarity):
    def go():
        # Entry is at "bottom of range," exit toward median
        entry = 0.30
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < stationarity:
            return (0.06 - f) * c                  # range holds
        return (-0.10 - f) * c                     # process drifted
    return go


# ════════════════════════════════════════════════════════════════════════════
# 8. OFF-HOURS STALE QUOTE  [SPECULATIVE]
# 2am-6am ET, fewer eyeballs → MM quotes stale longer. If we can identify a
# stale quote relative to recent moves, hit it.
# Caveat: most "stale" quotes are stale FOR A REASON (no flow expected).
# ════════════════════════════════════════════════════════════════════════════
def off_hours(real_staleness):
    def go():
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < real_staleness:
            return (0.06 - f) * c
        return (-0.04 - f) * c
    return go


# ════════════════════════════════════════════════════════════════════════════
# 9. POST-NEWS LAY-OFF (initial overshoot fade)  [BEHAVIORAL]
# Immediately after a scheduled news release (Fed, CPI), markets often
# overshoot in the first 60-90s then settle. We wait 60s then fade.
# Distinct from v2 News Decay — that was 2-4h horizon.
# ════════════════════════════════════════════════════════════════════════════
def post_news_layoff(overshoot_pct):
    def go():
        overshoot = RNG.uniform(0.08, 0.15)
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < overshoot_pct:
            return (overshoot * 0.35 - f) * c
        return (-overshoot * 0.40 - f) * c
    return go


# ════════════════════════════════════════════════════════════════════════════
# 10. LIQUIDITY CRUNCH PROVIDER  [STRUCTURAL]
# When the order book is empty on one side (no resting bids/asks for ≥30s),
# provide liquidity at a generous price. The market's desperation makes them
# accept our terms; the price reverts to mid soon after.
# ════════════════════════════════════════════════════════════════════════════
def liquidity_crunch(revert_pct):
    def go():
        # We posted at, e.g., 10c below where price was; fills then mean-reverts
        edge = RNG.uniform(0.06, 0.12)
        entry = 0.40
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < revert_pct:
            return (edge * 0.6 - f) * c
        return (-edge * 0.5 - f) * c
    return go


STRATEGIES = [
    ("Power-Law Resolution Drift", "[STRUCTURAL] ", lambda: power_drift(0.66), lambda: power_drift(0.58)),
    ("Tail Premium Selling",       "[STRUCTURAL] ", lambda: tail_premium(0.8), lambda: tail_premium(0.3)),
    ("Basket-Sum Imbalance",       "[STRUCTURAL] ", lambda: basket_arb(0.10),  lambda: basket_arb(0.04)),
    ("New-Market First-Hour MM",   "[SPECULATIVE]", lambda: first_hour_mm(0.58), lambda: first_hour_mm(0.50)),
    ("VWAP Reversion",             "[SPECULATIVE]", lambda: vwap_reversion(0.58), lambda: vwap_reversion(0.50)),
    ("Pair-Trade Correlated",      "[STRUCTURAL] ", lambda: pair_trade(0.68),  lambda: pair_trade(0.55)),
    ("Stationary Range Trade",     "[SPECULATIVE]", lambda: range_trade(0.62), lambda: range_trade(0.52)),
    ("Off-Hours Stale Quote",      "[SPECULATIVE]", lambda: off_hours(0.55),   lambda: off_hours(0.50)),
    ("Post-News 60s Lay-Off",      "[BEHAVIORAL] ", lambda: post_news_layoff(0.58), lambda: post_news_layoff(0.50)),
    ("Liquidity Crunch Provider",  "[STRUCTURAL] ", lambda: liquidity_crunch(0.62), lambda: liquidity_crunch(0.52)),
]


def main():
    print(f"\nMonte Carlo v3 — {N_RUNS} runs × {TRADES_PER_RUN} trades, ${STAKE:.0f}/trade")
    print(f"BASE + PESSIMISTIC scenarios per strategy. Same survival bar as v2.\n")
    hdr = (f"{'strategy':28s} {'tag':14s}  {'BASE: medP&L':>11} {'prof%':>5} {'Sharpe':>6} {'maxDD':>8}   "
           f"{'PESS: medP&L':>11} {'prof%':>5} {'Sharpe':>6}   verdict")
    print(hdr); print("-" * len(hdr))
    out = []
    for name, tag, b, p in STRATEGIES:
        v = evaluate(name, tag, b(), p())
        out.append(v)
        print(f"{name:28s} {tag:14s}  "
              f"${v.base_pnl:>+8.0f} {v.base_prof:>4.0f}% {v.base_sharpe:>6.2f} ${v.base_dd:>+6.0f}   "
              f"${v.pess_pnl:>+8.0f} {v.pess_prof:>4.0f}% {v.pess_sharpe:>6.2f}   {v.verdict}")
    print("-" * len(hdr))
    winners = [v for v in out if v.survives]
    print(f"\nSurvivors: {len(winners)} of {len(out)}")
    for v in winners:
        print(f"  ✅ {v.name}  {v.tag.strip()}  base ${v.base_pnl:+.0f}, pessimistic ${v.pess_pnl:+.0f}")
    if not winners:
        print("  (none — discipline holds)")


if __name__ == "__main__":
    main()
