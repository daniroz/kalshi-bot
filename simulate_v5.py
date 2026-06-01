"""
Monte Carlo v5 — 50 strategies. Exhaustive exploration round.

After v1-v4 (45 candidates, 3 survivors, 2 audit-confirmed implementations),
this round goes to 50 to genuinely cover the strategy space across every
category a vet might consider:
  microstructure / sports-specific / weather model / political / on-chain /
  social / calendar / scheduling / vol-of-vol / cross-market / stat-arb.

The expectation, honestly: 0-3 survivors. This round's value is *confirming
the strategy space has been explored*, not finding alpha. After this we let
running strategies generate signal — verify_favbias.py and basket_arb scanner.

Same survival bar: positive in BOTH base AND pessimistic, Sharpe > 0.4,
≥60% profitable seasons, max DD ≤ 50% of median P&L.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

RNG = np.random.default_rng(17)
N_RUNS = 500
TRADES_PER_RUN = 150
STAKE = 25.0


def fee(price, c=1):
    return 0.07 * price * (1 - price) * c


@dataclass
class V:
    name: str; tag: str; cat: str
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
        return "✅" if self.survives else "❌"


def run(trade_fn):
    pnls, dds = [], []
    for _ in range(N_RUNS):
        pnl, peak, dd = 0.0, 0.0, 0.0
        for _ in range(TRADES_PER_RUN):
            dp = trade_fn()
            if dp is None: continue
            pnl += dp; peak = max(peak, pnl); dd = min(dd, pnl - peak)
        pnls.append(pnl); dds.append(dd)
    pnls = np.array(pnls)
    return (float(np.median(pnls)), float(np.mean(pnls>0)*100),
            (float(np.mean(pnls))/float(np.std(pnls))) if np.std(pnls)>0 else 0,
            float(np.median(dds)))


def evaluate(name, tag, cat, base_fn, pess_fn) -> V:
    bp,bprof,bsh,bdd = run(base_fn)
    pp,pprof,psh,pdd = run(pess_fn)
    return V(name,tag,cat,bp,bprof,bsh,bdd,pp,pprof,psh,pdd)


# ═════════════════════════════════════════════════════════════════════════
# Generic patterns — each strategy just configures parameters
# ═════════════════════════════════════════════════════════════════════════

def taker(win_p, entry, win_gain, lose_loss):
    """Active 1-leg trade: pay fee once, win or lose by specified amounts."""
    def go():
        f = fee(entry)
        c = STAKE / entry
        if RNG.random() < win_p:
            return (win_gain - f) * c
        return (-lose_loss - f) * c
    return go


def mm_2leg(noise_share, capture, adverse_loss, entry=0.5):
    """Passive 2-leg market making — capture spread, eat adverse selection."""
    def go():
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < noise_share:
            return (capture - f) * c
        return (-adverse_loss - f) * c
    return go


def sparse(opp_rate, win_p, win_gain, lose_loss, entry=0.5, n_fees=2):
    """Sparse opportunity (rare events) — skips when no opp present."""
    def go():
        if RNG.random() >= opp_rate:
            return None
        f = fee(entry) * n_fees
        c = STAKE / entry
        if RNG.random() < win_p:
            return (win_gain - f) * c
        return (-lose_loss - f) * c
    return go


def lottery(p_market, p_true, c_size=None):
    """Buy YES at p_market when true prob is p_true. For longshots / favorites."""
    def go():
        price = RNG.uniform(p_market[0], p_market[1])
        true_p = np.clip(p_true(price), 0.001, 0.999)
        f = fee(price)
        c = STAKE / price
        won = RNG.random() < true_p
        return ((1-price)-f)*c if won else (-price-f)*c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 50 STRATEGIES (name, tag, category, base_factory, pess_factory)
# ═════════════════════════════════════════════════════════════════════════

SCHEMA = [
    # ── MICROSTRUCTURE (10) ────────────────────────────────────────────────
    ("Spread Compression Precursor",   "[STR]", "micro",
        lambda: taker(0.55, 0.50, 0.05, 0.05), lambda: taker(0.50, 0.50, 0.05, 0.05)),
    ("Lopsided Fill Ratio Follow",     "[STR]", "micro",
        lambda: taker(0.56, 0.50, 0.04, 0.04), lambda: taker(0.51, 0.50, 0.04, 0.04)),
    ("Block Trade Detection",          "[STR]", "micro",
        lambda: sparse(0.08, 0.62, 0.05, 0.05), lambda: sparse(0.04, 0.55, 0.05, 0.05)),
    ("Order Book Replenish Lag",       "[STR]", "micro",
        lambda: taker(0.58, 0.50, 0.04, 0.04), lambda: taker(0.51, 0.50, 0.04, 0.04)),
    ("Cross-Tier Depth Signal",        "[STR]", "micro",
        lambda: taker(0.56, 0.50, 0.04, 0.04), lambda: taker(0.50, 0.50, 0.04, 0.04)),
    ("Synthetic Straddle Cap",         "[STR]", "micro",
        lambda: mm_2leg(0.60, 0.04, 0.06), lambda: mm_2leg(0.52, 0.04, 0.06)),
    ("Round-Number Magnetism Fade",    "[BEH]", "micro",
        lambda: taker(0.58, 0.50, 0.04, 0.04), lambda: taker(0.52, 0.50, 0.04, 0.04)),
    ("Last-Trade Reversion",           "[BEH]", "micro",
        lambda: taker(0.56, 0.50, 0.03, 0.04), lambda: taker(0.50, 0.50, 0.03, 0.04)),
    ("Queue Position Race",            "[SPEC]","micro",
        lambda: mm_2leg(0.62, 0.03, 0.05), lambda: mm_2leg(0.54, 0.03, 0.05)),
    ("Quote Stuffing Detection Fade",  "[SPEC]","micro",
        lambda: taker(0.54, 0.50, 0.04, 0.05), lambda: taker(0.50, 0.50, 0.04, 0.05)),

    # ── SPORTS MICRO (8) ───────────────────────────────────────────────────
    ("Star Injury News Lag",           "[BEH]", "sports",
        lambda: sparse(0.05, 0.70, 0.08, 0.06), lambda: sparse(0.02, 0.60, 0.08, 0.06)),
    ("Halftime Reversion (NBA)",       "[BEH]", "sports",
        lambda: taker(0.54, 0.50, 0.05, 0.05), lambda: taker(0.50, 0.50, 0.05, 0.05)),
    ("Quarter-End Foul Trouble",       "[STR]", "sports",
        lambda: sparse(0.10, 0.62, 0.06, 0.06), lambda: sparse(0.05, 0.55, 0.06, 0.06)),
    ("MLB Closer Reliability",         "[STR]", "sports",
        lambda: sparse(0.08, 0.60, 0.05, 0.06), lambda: sparse(0.04, 0.53, 0.05, 0.06)),
    ("NHL Empty-Net Last 2min MM",     "[STR]", "sports",
        lambda: mm_2leg(0.65, 0.05, 0.07), lambda: mm_2leg(0.55, 0.05, 0.07)),
    ("Pitcher Fatigue Bullpen Risk",   "[STR]", "sports",
        lambda: sparse(0.06, 0.65, 0.06, 0.05), lambda: sparse(0.03, 0.55, 0.06, 0.05)),
    ("Back-to-Back Game Fatigue",      "[BEH]", "sports",
        lambda: taker(0.55, 0.50, 0.04, 0.05), lambda: taker(0.50, 0.50, 0.04, 0.05)),
    ("Travel-Distance Effect",         "[BEH]", "sports",
        lambda: taker(0.53, 0.50, 0.04, 0.05), lambda: taker(0.50, 0.50, 0.04, 0.05)),

    # ── WEATHER / SCIENCE (5) ──────────────────────────────────────────────
    ("Weather Model Disagreement",     "[STR]", "weather",
        lambda: sparse(0.12, 0.62, 0.10, 0.08), lambda: sparse(0.06, 0.55, 0.10, 0.08)),
    ("Pressure Trend Predictor",       "[SPEC]","weather",
        lambda: taker(0.55, 0.50, 0.06, 0.06), lambda: taker(0.50, 0.50, 0.06, 0.06)),
    ("Sea-Surface Temp Anomaly",       "[STR]", "weather",
        lambda: sparse(0.05, 0.60, 0.08, 0.07), lambda: sparse(0.02, 0.52, 0.08, 0.07)),
    ("Snowpack Spring Effect",         "[STR]", "weather",
        lambda: sparse(0.04, 0.58, 0.06, 0.06), lambda: sparse(0.02, 0.50, 0.06, 0.06)),
    ("Solar Activity Indirect",        "[SPEC]","weather",
        lambda: taker(0.52, 0.50, 0.04, 0.05), lambda: taker(0.50, 0.50, 0.04, 0.05)),

    # ── POLITICAL / REGULATORY (5) ─────────────────────────────────────────
    ("Federal Register Pub Scan",      "[STR]", "political",
        lambda: sparse(0.03, 0.65, 0.10, 0.08), lambda: sparse(0.01, 0.55, 0.10, 0.08)),
    ("Congressional Whip Count",       "[STR]", "political",
        lambda: sparse(0.05, 0.62, 0.08, 0.07), lambda: sparse(0.02, 0.53, 0.08, 0.07)),
    ("SCOTUS Oral Argument NLP",       "[SPEC]","political",
        lambda: sparse(0.03, 0.58, 0.10, 0.10), lambda: sparse(0.01, 0.50, 0.10, 0.10)),
    ("Presidential Approval Rolling",  "[BEH]", "political",
        lambda: taker(0.54, 0.50, 0.04, 0.05), lambda: taker(0.50, 0.50, 0.04, 0.05)),
    ("State Legislature Patterns",     "[SPEC]","political",
        lambda: sparse(0.04, 0.55, 0.06, 0.06), lambda: sparse(0.02, 0.50, 0.06, 0.06)),

    # ── CRYPTO ON-CHAIN (5) ────────────────────────────────────────────────
    ("Stablecoin De-Peg Correlation",  "[STR]", "crypto",
        lambda: sparse(0.03, 0.70, 0.12, 0.10), lambda: sparse(0.01, 0.55, 0.12, 0.10)),
    ("Mining Hashrate Drop",           "[STR]", "crypto",
        lambda: sparse(0.04, 0.60, 0.08, 0.08), lambda: sparse(0.02, 0.52, 0.08, 0.08)),
    ("MVRV On-Chain Signal",           "[SPEC]","crypto",
        lambda: taker(0.53, 0.50, 0.06, 0.06), lambda: taker(0.50, 0.50, 0.06, 0.06)),
    ("Stock-to-Flow Deviation",        "[SPEC]","crypto",
        lambda: taker(0.52, 0.50, 0.05, 0.06), lambda: taker(0.50, 0.50, 0.05, 0.06)),
    ("Exchange Outflow Trend",         "[BEH]", "crypto",
        lambda: taker(0.55, 0.50, 0.05, 0.05), lambda: taker(0.51, 0.50, 0.05, 0.05)),

    # ── SOCIAL / SENTIMENT (5) ─────────────────────────────────────────────
    ("Reddit /r/Kalshi Mention Spike", "[SPEC]","social",
        lambda: sparse(0.05, 0.58, 0.06, 0.06), lambda: sparse(0.02, 0.50, 0.06, 0.06)),
    ("Wikipedia Edit Timing",          "[SPEC]","social",
        lambda: sparse(0.04, 0.56, 0.06, 0.06), lambda: sparse(0.02, 0.50, 0.06, 0.06)),
    ("Google Trends Correlation",      "[SPEC]","social",
        lambda: taker(0.52, 0.50, 0.04, 0.05), lambda: taker(0.50, 0.50, 0.04, 0.05)),
    ("Twitter Verified Signal",        "[SPEC]","social",
        lambda: taker(0.53, 0.50, 0.05, 0.06), lambda: taker(0.50, 0.50, 0.05, 0.06)),
    ("Press Release Wire Scan",        "[STR]", "social",
        lambda: sparse(0.03, 0.65, 0.10, 0.08), lambda: sparse(0.01, 0.55, 0.10, 0.08)),

    # ── CALENDAR / TIME (5) ────────────────────────────────────────────────
    ("Hour-of-Day Mean Revert",        "[BEH]", "calendar",
        lambda: taker(0.55, 0.50, 0.04, 0.04), lambda: taker(0.50, 0.50, 0.04, 0.04)),
    ("Day-of-Week Pattern",            "[BEH]", "calendar",
        lambda: taker(0.53, 0.50, 0.04, 0.04), lambda: taker(0.50, 0.50, 0.04, 0.04)),
    ("Holiday Liquidity Discount",     "[STR]", "calendar",
        lambda: mm_2leg(0.65, 0.05, 0.07), lambda: mm_2leg(0.55, 0.05, 0.07)),
    ("Tax-Loss Season Flow",           "[SPEC]","calendar",
        lambda: sparse(0.05, 0.55, 0.05, 0.06), lambda: sparse(0.02, 0.50, 0.05, 0.06)),
    ("Daylight-Saving Transition",     "[SPEC]","calendar",
        lambda: sparse(0.02, 0.55, 0.05, 0.06), lambda: sparse(0.01, 0.50, 0.05, 0.06)),

    # ── STATISTICAL / MATH (7) ─────────────────────────────────────────────
    ("Variance Ratio Mean Revert",     "[STR]", "stat",
        lambda: taker(0.56, 0.50, 0.04, 0.04), lambda: taker(0.51, 0.50, 0.04, 0.04)),
    ("Hurst Exponent Regime",          "[SPEC]","stat",
        lambda: taker(0.54, 0.50, 0.05, 0.05), lambda: taker(0.50, 0.50, 0.05, 0.05)),
    ("Cointegration Pair",             "[STR]", "stat",
        lambda: sparse(0.10, 0.60, 0.05, 0.05), lambda: sparse(0.05, 0.53, 0.05, 0.05)),
    ("Kalman Filter Signal",           "[SPEC]","stat",
        lambda: taker(0.54, 0.50, 0.04, 0.04), lambda: taker(0.50, 0.50, 0.04, 0.04)),
    ("Implied Vol Surface Arb",        "[STR]", "stat",
        lambda: sparse(0.06, 0.65, 0.05, 0.05), lambda: sparse(0.03, 0.55, 0.05, 0.05)),
    ("Realized vs Implied Vol",        "[STR]", "stat",
        lambda: sparse(0.05, 0.60, 0.05, 0.05), lambda: sparse(0.02, 0.52, 0.05, 0.05)),
    ("Skew Steepness Anomaly",         "[SPEC]","stat",
        lambda: sparse(0.05, 0.58, 0.06, 0.06), lambda: sparse(0.02, 0.50, 0.06, 0.06)),

    # ── EXTREME ZONE EXTENSIONS (5) ────────────────────────────────────────
    # Test variants of favorite-longshot bias at different edge bands
    ("Mild-Favorite Zone (70-85¢)",    "[EMP]", "favbias",
        lambda: lottery((0.70, 0.85), lambda p: np.clip(p + 0.020 + RNG.normal(0,0.015), 0, 0.999)),
        lambda: lottery((0.70, 0.85), lambda p: np.clip(p + 0.010 + RNG.normal(0,0.015), 0, 0.999))),
    ("Middle Favorite (60-70¢)",       "[BEH]", "favbias",
        lambda: lottery((0.60, 0.70), lambda p: np.clip(p + 0.015 + RNG.normal(0,0.015), 0, 0.999)),
        lambda: lottery((0.60, 0.70), lambda p: np.clip(p + 0.005 + RNG.normal(0,0.015), 0, 0.999))),
    ("Coin-Flip Zone (45-55¢)",        "[SPEC]","favbias",
        lambda: lottery((0.45, 0.55), lambda p: np.clip(p + 0.010 + RNG.normal(0,0.02), 0, 0.999)),
        lambda: lottery((0.45, 0.55), lambda p: np.clip(p + 0.000 + RNG.normal(0,0.02), 0, 0.999))),
    ("Slight Underdog (30-45¢)",       "[SPEC]","favbias",
        lambda: lottery((0.30, 0.45), lambda p: np.clip(p + 0.005 + RNG.normal(0,0.02), 0, 0.999)),
        lambda: lottery((0.30, 0.45), lambda p: np.clip(p - 0.005 + RNG.normal(0,0.02), 0, 0.999))),
    ("Underdog Sell-Side (15-30¢)",    "[EMP]", "favbias",
        lambda: lottery((0.15, 0.30), lambda p: np.clip(p - 0.020 + RNG.normal(0,0.015), 0, 0.999)),
        lambda: lottery((0.15, 0.30), lambda p: np.clip(p - 0.010 + RNG.normal(0,0.015), 0, 0.999))),
]


def main():
    print(f"\nMonte Carlo v5 — {len(SCHEMA)} strategies × {N_RUNS} runs × {TRADES_PER_RUN} trades\n")
    hdr = f"{'#':>3} {'category':10s} {'strategy':33s} {'tag':6s}  {'BASE: P&L':>9} {'p%':>4} {'Shrp':>5}   {'PESS: P&L':>9} {'p%':>4}   ✓"
    print(hdr); print("-" * len(hdr))
    out = []
    for i, (name, tag, cat, b, p) in enumerate(SCHEMA, 1):
        v = evaluate(name, tag, cat, b(), p())
        out.append(v)
        print(f"{i:>3} {cat:10s} {name:33s} {tag:6s}  ${v.base_pnl:>+6.0f} {v.base_prof:>3.0f}% {v.base_sharpe:>5.2f}   ${v.pess_pnl:>+6.0f} {v.pess_prof:>3.0f}%   {v.verdict}")
    print("-" * len(hdr))
    winners = [v for v in out if v.survives]
    print(f"\nSurvivors: {len(winners)} of {len(out)}")
    for v in winners:
        print(f"  ✅ [{v.cat}] {v.name}  base ${v.base_pnl:+.0f}, pess ${v.pess_pnl:+.0f}, "
              f"Sharpe {v.base_sharpe:.2f}/{v.pess_sharpe:.2f}")
    print(f"\nCumulative across all 5 rounds: {45+50} strategies tested.")


if __name__ == "__main__":
    main()
