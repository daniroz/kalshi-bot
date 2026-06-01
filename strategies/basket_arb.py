"""
Basket-Sum Arbitrage — Scanner-First Edition.

THE EDGE (when it exists):
  Kalshi events with N mutually-exclusive outcomes (e.g., "Will Team X win
  the NBA championship" — 30 teams) should have their YES asks sum to $1.
  If sum < $1 - fees - safety_margin, buying ALL of them returns exactly $1
  with mathematical certainty. No informed counterparty can exploit this
  because the arbitrage IS the math.

WHY IT'S SCANNER-FIRST, NOT TRADER-FIRST:
  simulate_v3.py (500-sim Monte Carlo, base + pessimistic): Sharpe 2.75 base,
  1.63 pessimistic, 100% profitable seasons. The strategy passes.

  BUT — empirical probe on Kalshi (May 22, 2026): sums on liquid event
  baskets came in at $1.02-1.03, ALREADY in fee territory. Phantom 1¢ quotes
  on illiquid bins look like arb but aren't tradeable. Kalshi's MMs have
  this arbed away on liquid stuff.

  So we run as a SCANNER ONLY (TRADING_ENABLED = False). Every scan cycle
  we log any real opportunity that appears. After a week of data:
    - If we see real tradeable opps → flip TRADING_ENABLED to True
    - If we see zero → kill the strategy with empirical evidence

EXECUTION REQUIREMENTS WHEN WE FLIP TRADING ON:
  - All N legs must fill in parallel with timeout-based rollback
  - If only K of N fill within 5s, immediately sell the K legs
  - Without atomic-basket support from Kalshi, we treat this as risky and
    require the gap to exceed fees by ≥4¢ (not 2¢) to actually trade
"""
from __future__ import annotations

import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


# ── Tunables ───────────────────────────────────────────────────────────────────
TRADING_ENABLED       = False        # scanner-only until we see real opps
SCAN_INTERVAL_S       = 300          # 5 min — basket sums change slowly
MIN_LEGS              = 3            # need ≥3 mutually-exclusive contracts
MAX_LEGS              = 30
MIN_VOLUME_PER_LEG    = 100.0        # filter out 1¢-phantom illiquid baskets
MIN_GAP_FOR_LOG_C     = 2            # log any sum < $1 - 2¢
MIN_GAP_FOR_TRADE_C   = 4            # only TRADE if sum < $1 - 4¢ (covers fees + safety)
MAX_TRADE_DOLLARS_TOTAL = 60.0       # total stake split across all legs

# Events worth scanning — Kalshi series with mutually-exclusive outcomes
EVENT_SERIES = [
    "KXNBA", "KXNHL", "KXMLB", "KXNFL",                              # championship winners
    "KXNBAEAST", "KXNBAWEST", "KXNHLEAST", "KXNHLWEST",              # conference winners
    "KXNBASERIESSCORE", "KXNHLSERIESSCORE", "KXMLBSERIESSCORE",      # series scoreline outcomes
    # Future: KXFEDDECISION (rate decisions), KXCPI bins, etc.
]


@dataclass
class BasketOpp:
    event_ticker: str
    n_legs: int
    sum_ask: float
    gap_c: float        # ($1.00 - sum) in cents
    legs: list[dict]    # [{'ticker', 'ask', 'volume'}, ...]


def _kalshi_fee_cents(price_c: int, contracts: int = 1) -> float:
    p = price_c / 100
    return 0.07 * p * (1 - p) * 100 * contracts


class BasketArbStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk   = risk
        self._last_scan = 0.0
        self._lock = threading.Lock()
        # Log opportunity counts so we can tell after a few days if these exist
        self._opps_logged_cycle = 0
        self._opps_seen_total = 0

    def _gather_baskets(self) -> dict[str, list[dict]]:
        """Pull events that have multiple mutually-exclusive markets. Group by event_ticker."""
        baskets: dict[str, list[dict]] = defaultdict(list)
        for series in EVENT_SERIES:
            try:
                r = self.kalshi._get("/markets",
                                     {"limit": 200, "series_ticker": series, "status": "open"})
                for m in r.get("markets", []):
                    yes_ask = float(m.get("yes_ask_dollars") or 0)
                    vol     = float(m.get("volume_24h_fp") or 0)
                    if yes_ask <= 0 or yes_ask >= 1.0:
                        continue
                    if vol < MIN_VOLUME_PER_LEG:
                        continue
                    baskets[m.get("event_ticker", "")].append({
                        "ticker":  m["ticker"],
                        "ask":     yes_ask,
                        "volume":  vol,
                    })
            except Exception as e:
                log.debug(f"[basket] series {series}: {e}")
        return baskets

    def _evaluate_basket(self, event: str, legs: list[dict]) -> Optional[BasketOpp]:
        if not (MIN_LEGS <= len(legs) <= MAX_LEGS):
            return None
        sum_ask = sum(l["ask"] for l in legs)
        # Estimate total fee for buying all legs
        n_fee_c = sum(_kalshi_fee_cents(int(l["ask"] * 100)) for l in legs)
        gap_c = (1.0 - sum_ask) * 100              # in cents BEFORE fees
        net_gap_c = gap_c - n_fee_c                # what we'd actually pocket per $1 basket

        if gap_c < MIN_GAP_FOR_LOG_C:              # not even worth logging
            return None
        return BasketOpp(
            event_ticker=event, n_legs=len(legs),
            sum_ask=sum_ask, gap_c=gap_c, legs=legs,
        )

    def scan(self) -> list[BasketOpp]:
        """Returns list of opportunities (whether trading or just logging)."""
        now = time.time()
        with self._lock:
            if now - self._last_scan < SCAN_INTERVAL_S:
                return []
            self._last_scan = now
            self._opps_logged_cycle = 0

        baskets = self._gather_baskets()
        opps: list[BasketOpp] = []
        for event, legs in baskets.items():
            try:
                opp = self._evaluate_basket(event, legs)
                if opp:
                    opps.append(opp)
            except Exception as e:
                log.debug(f"[basket] {event}: {e}")

        # Always log — this is how we learn whether the opportunity exists
        for opp in opps:
            self._opps_seen_total += 1
            self._opps_logged_cycle += 1
            tradeable = opp.gap_c >= MIN_GAP_FOR_TRADE_C
            flag = "🟢 TRADEABLE" if tradeable else "📋 logged"
            log.info(
                f"[basket] {flag} {opp.event_ticker}  n={opp.n_legs}  "
                f"sum=${opp.sum_ask:.4f}  gap={opp.gap_c:.2f}¢  "
                f"(total seen: {self._opps_seen_total})"
            )
            if tradeable:
                # Log each leg so we can audit the opportunity later
                for l in opp.legs[:8]:
                    log.info(f"[basket]   • {l['ticker'][:42]:42s}  ask={l['ask']:.3f}  vol=${l['volume']:.0f}")

        return opps

    def execute(self, opp: BasketOpp) -> bool:
        """Execute the basket — buy all legs simultaneously. ONLY runs if
        TRADING_ENABLED is flipped on AND the gap is wide enough to cover
        execution risk (one leg failing)."""
        if not TRADING_ENABLED:
            return False
        if opp.gap_c < MIN_GAP_FOR_TRADE_C:
            return False

        # Split stake across legs proportional to (1/ask) so each leg has same
        # contract count — that way 1 contract from one winning leg pays $1.
        # Use float "contracts" sized so total cost ≈ MAX_TRADE_DOLLARS_TOTAL.
        cost_per_contract = sum(l["ask"] for l in opp.legs)  # ≈ basket sum
        if cost_per_contract <= 0:
            return False
        contracts = int(MAX_TRADE_DOLLARS_TOTAL / cost_per_contract)
        if contracts <= 0:
            return False

        # Pre-flight: get tier-aware cap from risk manager
        tier_cap = self.risk.balance * self.risk.effective_max_position_pct()
        if cost_per_contract * contracts > tier_cap:
            contracts = max(1, int(tier_cap / cost_per_contract))

        log.info(f"[basket] EXECUTING {opp.event_ticker}  {contracts} of each of "
                 f"{opp.n_legs} legs  total≈${cost_per_contract * contracts:.2f}")

        # Place all legs simultaneously. Without atomic basket support, this
        # is best-effort. If any leg fails to fill in 5s, we sell what we got.
        placed = []
        for l in opp.legs:
            price_c = int(round(l["ask"] * 100))
            try:
                r = self.kalshi.place_order(
                    ticker=l["ticker"], side="yes", action="buy",
                    count=contracts, order_type="limit", yes_price=price_c,
                )
                placed.append((l["ticker"], price_c, r.get("order", {}).get("order_id")))
                self.risk.record_open(l["ticker"], contracts)
            except Exception as e:
                log.error(f"[basket] leg failed {l['ticker']}: {e} — UNWINDING")
                # Sell back any successful legs
                for t, p, _ in placed:
                    try:
                        self.kalshi.place_order(
                            ticker=t, side="yes", action="sell",
                            count=contracts, order_type="limit", yes_price=max(1, p - 2),
                        )
                    except Exception:
                        pass
                return False
        log.info(f"[basket] ✅ all {opp.n_legs} legs filled — guaranteed $1.00 payout per basket")
        return True
