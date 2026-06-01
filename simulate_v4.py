"""
Monte Carlo v4 — 20 MORE strategies, same brutal discipline.

After 25 prior candidates → 3 implementations, we're now exploring weirder
angles: microstructure, time-of-day patterns, sports-specific math, Bayesian
underreaction, calendar effects. Most will fail. That's expected. The vet's
job is to keep generating ideas faster than the bad ones drain capital.

Same survival bar: positive in BOTH base AND pessimistic, Sharpe > 0.4,
≥60% profitable seasons, max DD ≤ 50% of median P&L.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

RNG = np.random.default_rng(11)
N_RUNS = 500
TRADES_PER_RUN = 150
STAKE = 25.0


def fee(price, c=1):
    return 0.07 * price * (1 - price) * c


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
    return (float(np.median(pnls)), float(np.mean(pnls>0)*100),
            (float(np.mean(pnls))/float(np.std(pnls))) if np.std(pnls)>0 else 0,
            float(np.median(dds)))


def evaluate(name, tag, base_fn, pess_fn) -> V:
    bp,bprof,bsh,bdd = run_paths(base_fn)
    pp,pprof,psh,pdd = run_paths(pess_fn)
    return V(name,tag,bp,bprof,bsh,bdd,pp,pprof,psh,pdd)


# ═════════════════════════════════════════════════════════════════════════
# 1. UNDERDOG LATE-COMEBACK SKEW HARVESTING  [BEHAVIORAL]
# Heavy underdogs (5-10¢) in still-close games have right-skewed payoff:
# market underprices the lottery upside. Buy YES on cheap underdog when game
# is within 1 score with time left.
# ═════════════════════════════════════════════════════════════════════════
def underdog_skew(true_prob_uplift):
    def go():
        p = RNG.uniform(0.05, 0.10)
        true_p = min(0.20, p + true_prob_uplift)
        f = fee(p)
        c = STAKE / p
        won = RNG.random() < true_p
        return ((1-p)-f)*c if won else (-p-f)*c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 2. BAYESIAN UNDERREACTION (slow news incorporation)  [BEHAVIORAL]
# Markets only partially update on news within first 5-10 min. Trade the
# remaining gap to true Bayesian posterior.
# ═════════════════════════════════════════════════════════════════════════
def bayesian_lag(continuation_pct):
    def go():
        shock = RNG.uniform(0.05, 0.10)
        entry = 0.55
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < continuation_pct:
            return (shock * 0.4 - f) * c
        return (-shock * 0.5 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 3. PSYCHOLOGICAL LEVEL STOP-HUNTING  [BEHAVIORAL]
# 50¢ and 75¢ are anchor levels where stops cluster. When price spikes
# through, it overshoots → revert. Fade the spike.
# ═════════════════════════════════════════════════════════════════════════
def stop_hunt_fade(revert_pct):
    def go():
        overshoot = RNG.uniform(0.04, 0.08)
        entry = 0.55
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < revert_pct:
            return (overshoot * 0.5 - f) * c
        return (-overshoot * 0.5 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 4. PENNY-JUMP MM RECAPTURE  [STRUCTURAL]
# Post quotes 1¢ better than existing MM. Capture order flow before they
# adjust. Edge if we can stay ahead; loss to adverse selection on each fill.
# ═════════════════════════════════════════════════════════════════════════
def penny_jump(fill_quality):
    def go():
        # We capture 4¢ per round trip (jumped 1¢ × 2 sides + 2¢ width)
        capture = 0.04
        f = fee(0.5) * 2
        c = STAKE / 0.50
        if RNG.random() < fill_quality:
            return (capture - f) * c
        return (-0.08 - f) * c                  # toxic flow when wrong
    return go


# ═════════════════════════════════════════════════════════════════════════
# 5. SCHEDULED-NEWS FADE (Fed/CPI within 30s)  [BEHAVIORAL]
# Within first 30s of a scheduled release, prices wildly overshoot before
# settling. Wait 30s, fade overshoot. (Distinct from v3's 60s lay-off.)
# ═════════════════════════════════════════════════════════════════════════
def scheduled_news_fade(revert_pct):
    def go():
        overshoot = RNG.uniform(0.10, 0.20)
        entry = 0.55
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < revert_pct:
            return (overshoot * 0.45 - f) * c
        return (-overshoot * 0.55 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 6. SERIES SCORELINE BINOMIAL FIT  [STRUCTURAL]
# "Team A wins 4-1" markets should fit binomial distribution given implied
# per-game probability. When market deviates from fit, take the corrected
# side. Multi-leg with execution complexity.
# ═════════════════════════════════════════════════════════════════════════
def binomial_fit(opp_rate):
    def go():
        if RNG.random() >= opp_rate:
            return None
        entry = 0.20
        f = fee(entry) * 2
        c = STAKE / entry
        # When mispricing is real, ~75% chance correction
        if RNG.random() < 0.72:
            return (0.05 - f) * c
        return (-0.06 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 7. SAME-EVENING CROSS-GAME MOMENTUM  [SPECULATIVE]
# First NBA blowout of the evening → second-game underdog markets shift
# (volume migrates). Trade the migration before book repricing.
# ═════════════════════════════════════════════════════════════════════════
def cross_game_momentum(real_effect):
    def go():
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < real_effect:
            return (0.04 - f) * c
        return (-0.05 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 8. RECURRING-MARKET ANCHORING  [BEHAVIORAL]
# Daily Trump polling, weekly CPI bins — markets anchor on prior resolution.
# Fade overcorrection when prior week saw extreme outcome.
# ═════════════════════════════════════════════════════════════════════════
def anchor_fade(revert_pct):
    def go():
        anchor_bias = RNG.uniform(0.05, 0.10)
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < revert_pct:
            return (anchor_bias * 0.4 - f) * c
        return (-anchor_bias * 0.5 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 9. BID-MOVE-ASK-LAG  [STRUCTURAL]
# When bid jumps but ask hasn't moved, the spread compresses → information
# is hitting bid side. Buy through stale ask before it adjusts.
# ═════════════════════════════════════════════════════════════════════════
def bid_ask_lag(info_pct):
    def go():
        edge = RNG.uniform(0.03, 0.07)
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < info_pct:
            return (edge * 0.5 - f) * c
        return (-edge * 0.5 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 10. WEEKEND/LOW-VOL MARKET MAKING  [STRUCTURAL]
# Weekends have lower retail flow → less informed counterparty per fill.
# MM aggressively on weekend sports games where retail noise dominates.
# ═════════════════════════════════════════════════════════════════════════
def weekend_mm(noise_share):
    def go():
        spread = RNG.uniform(0.06, 0.10)
        capture = spread - 0.02
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < noise_share:
            return (capture - f) * c
        return (-spread * 1.0 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 11. END-OF-MONTH FLOW  [SPECULATIVE]
# Some markets see end-of-month rebalance flow → predictable price pressure.
# Highly market-specific; mostly a Kalshi-on-Wall-Street thesis.
# ═════════════════════════════════════════════════════════════════════════
def eom_flow(real_effect):
    def go():
        edge = RNG.uniform(0.04, 0.08)
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < real_effect:
            return (edge * 0.4 - f) * c
        return (-edge * 0.5 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 12. HIDDEN-SIZE DETECTION  [STRUCTURAL]
# When same-size order keeps reappearing at same price after partial fills,
# someone is hiding size. Follow the direction.
# ═════════════════════════════════════════════════════════════════════════
def hidden_size_follow(signal_quality):
    def go():
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < signal_quality:
            return (0.06 - f) * c
        return (-0.05 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 13. CROSS-SPORT EVENING FOCUS  [SPECULATIVE]
# When NHL+NBA both run, NHL gets less attention → wider spreads. MM on
# NHL specifically during NBA primetime.
# ═════════════════════════════════════════════════════════════════════════
def cross_sport_attention(noise_share):
    def go():
        spread = RNG.uniform(0.05, 0.09)
        capture = spread - 0.02
        entry = 0.45
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < noise_share:
            return (capture - f) * c
        return (-spread * 1.0 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 14. RESOLUTION-RULE EDGE CASE  [STRUCTURAL]
# Some markets have unusual resolution rules (e.g., "above 95% measured
# at NYSE 4:01pm"). If we read the rules and market hasn't, edge.
# ═════════════════════════════════════════════════════════════════════════
def rule_edge(opp_rate):
    def go():
        if RNG.random() >= opp_rate:
            return None
        entry = 0.30
        f = fee(entry) * 2
        c = STAKE / entry
        # When we have a real rule edge, we win ~80% of the time
        if RNG.random() < 0.78:
            return (0.40 - f) * c             # huge per-trade if we find it
        return (-0.30 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 15. SUB-SECOND MOMENTUM (last 60s of crypto/stock)  [SPECULATIVE]
# In last 60s of resolution, prices have to converge. Take direction
# of micro-trend. Pure latency game — we lose if anyone is faster.
# ═════════════════════════════════════════════════════════════════════════
def sub_second_mom(latency_win_rate):
    def go():
        entry = 0.50
        f = fee(entry)
        c = STAKE / entry
        if RNG.random() < latency_win_rate:
            return ((1-entry) - f) * c
        return (-entry - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 16. CONDITIONAL PROBABILITY ARB  [STRUCTURAL]
# P(A∩B) should = P(A) × P(B|A). When markets independently price A, B, and
# A&B without enforcing this identity, trade the cheap leg.
# ═════════════════════════════════════════════════════════════════════════
def conditional_arb(opp_rate):
    def go():
        if RNG.random() >= opp_rate:
            return None
        f = fee(0.5) * 3
        c = STAKE / 1.0
        if RNG.random() < 0.80:
            return (0.05 - f/100) * c
        return (-0.06 - f/100) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 17. SOCIAL SENTIMENT BURST  [SPECULATIVE]
# Twitter/Reddit chatter spikes precede price moves by 2-5 min. Catch the
# precursor signal. Hard to operationalize, costly to verify.
# ═════════════════════════════════════════════════════════════════════════
def social_burst(signal_value):
    def go():
        entry = 0.50
        f = fee(entry) * 2
        c = STAKE / entry
        if RNG.random() < signal_value:
            return (0.05 - f) * c
        return (-0.05 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 18. HETEROSCEDASTICITY HARVEST  [SPECULATIVE]
# Periods of high volatility cluster. Bet on continued vol via straddle-like
# basket positions. Complex to implement; modest edge if any.
# ═════════════════════════════════════════════════════════════════════════
def het_harvest(persistence):
    def go():
        f = fee(0.5) * 2
        c = STAKE / 0.50
        if RNG.random() < persistence:
            return (0.04 - f) * c
        return (-0.06 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 19. ICEBERG-ORDER DETECTION  [STRUCTURAL]
# When the same price level keeps refilling after our fills, someone has
# an iceberg. Pull our quote → wait → re-enter once they've cleared.
# ═════════════════════════════════════════════════════════════════════════
def iceberg(detection_quality):
    def go():
        f = fee(0.5) * 2
        c = STAKE / 0.50
        if RNG.random() < detection_quality:
            return (0.03 - f) * c
        return (-0.06 - f) * c
    return go


# ═════════════════════════════════════════════════════════════════════════
# 20. RESOLUTION-HOUR VOLATILITY EXPANSION  [STRUCTURAL]
# In the final 60 min before scheduled resolution, vol expands as final flow
# arrives. Sell vol via straddle: short YES at the top AND short NO at the
# top simultaneously. (Theta harvesting analog.)
# ═════════════════════════════════════════════════════════════════════════
def vol_sell(theta_capture):
    def go():
        # We collect ~3¢ of "premium" on each side that's overpriced
        f = fee(0.5) * 4              # 4 legs
        c = STAKE / 1.0
        if RNG.random() < theta_capture:
            return (0.05 - f/100) * c
        return (-0.08 - f/100) * c
    return go


STRATEGIES = [
    ("Underdog Late-Comeback Skew", "[BEHAVIORAL] ", lambda: underdog_skew(0.035), lambda: underdog_skew(0.020)),
    ("Bayesian Underreaction",      "[BEHAVIORAL] ", lambda: bayesian_lag(0.58),   lambda: bayesian_lag(0.51)),
    ("Stop-Hunt Fade (50¢/75¢)",   "[BEHAVIORAL] ", lambda: stop_hunt_fade(0.58), lambda: stop_hunt_fade(0.50)),
    ("Penny-Jump MM Recapture",     "[STRUCTURAL] ", lambda: penny_jump(0.62),     lambda: penny_jump(0.52)),
    ("Scheduled-News 30s Fade",     "[BEHAVIORAL] ", lambda: scheduled_news_fade(0.60), lambda: scheduled_news_fade(0.50)),
    ("Series Binomial Fit",         "[STRUCTURAL] ", lambda: binomial_fit(0.12),   lambda: binomial_fit(0.05)),
    ("Cross-Game Same-Evening Mo",  "[SPECULATIVE]", lambda: cross_game_momentum(0.55), lambda: cross_game_momentum(0.50)),
    ("Recurring-Market Anchoring",  "[BEHAVIORAL] ", lambda: anchor_fade(0.56),    lambda: anchor_fade(0.50)),
    ("Bid-Moves-Ask-Lags Hit",      "[STRUCTURAL] ", lambda: bid_ask_lag(0.58),    lambda: bid_ask_lag(0.52)),
    ("Weekend Low-Vol MM",          "[STRUCTURAL] ", lambda: weekend_mm(0.68),     lambda: weekend_mm(0.58)),
    ("End-of-Month Flow",           "[SPECULATIVE]", lambda: eom_flow(0.55),       lambda: eom_flow(0.50)),
    ("Hidden-Size Follow",          "[STRUCTURAL] ", lambda: hidden_size_follow(0.60), lambda: hidden_size_follow(0.52)),
    ("Cross-Sport Attention MM",    "[SPECULATIVE]", lambda: cross_sport_attention(0.68), lambda: cross_sport_attention(0.58)),
    ("Resolution-Rule Edge Case",   "[STRUCTURAL] ", lambda: rule_edge(0.03),      lambda: rule_edge(0.01)),
    ("Sub-Second Final Momentum",   "[SPECULATIVE]", lambda: sub_second_mom(0.52), lambda: sub_second_mom(0.50)),
    ("Conditional Prob Arb",        "[STRUCTURAL] ", lambda: conditional_arb(0.08),lambda: conditional_arb(0.03)),
    ("Social Sentiment Burst",      "[SPECULATIVE]", lambda: social_burst(0.55),   lambda: social_burst(0.50)),
    ("Heteroscedasticity Capture",  "[SPECULATIVE]", lambda: het_harvest(0.55),    lambda: het_harvest(0.50)),
    ("Iceberg Order Detect",        "[STRUCTURAL] ", lambda: iceberg(0.62),        lambda: iceberg(0.52)),
    ("Resolution-Hour Vol Sell",    "[STRUCTURAL] ", lambda: vol_sell(0.60),       lambda: vol_sell(0.52)),
]


def main():
    print(f"\nMonte Carlo v4 — 20 strategies × {N_RUNS} runs × {TRADES_PER_RUN} trades each\n")
    hdr = (f"{'strategy':28s} {'tag':14s}  {'BASE: P&L':>9} {'prof%':>5} {'Shrp':>5} {'DD':>7}   "
           f"{'PESS: P&L':>9} {'prof%':>5} {'Shrp':>5}   verdict")
    print(hdr); print("-" * len(hdr))
    out = []
    for name, tag, b, p in STRATEGIES:
        v = evaluate(name, tag, b(), p())
        out.append(v)
        print(f"{name:28s} {tag:14s}  ${v.base_pnl:>+6.0f} {v.base_prof:>4.0f}% {v.base_sharpe:>5.2f} ${v.base_dd:>+5.0f}   "
              f"${v.pess_pnl:>+6.0f} {v.pess_prof:>4.0f}% {v.pess_sharpe:>5.2f}   {v.verdict}")
    print("-" * len(hdr))
    winners = [v for v in out if v.survives]
    print(f"\nSurvivors: {len(winners)} of {len(out)}")
    for v in winners:
        print(f"  ✅ {v.name}  {v.tag.strip()}  base ${v.base_pnl:+.0f}, pessimistic ${v.pess_pnl:+.0f}")
    if not winners:
        print("  (none — bar held)")


if __name__ == "__main__":
    main()
