"""
Cross-market arbitrage: Kalshi vs Polymarket.

Logic: find Kalshi markets whose YES price differs from the
equivalent Polymarket market by more than the fee + min_edge.
Kalshi fee: ~3% of winnings. Polymarket fee: ~2%.

If Kalshi YES < Polymarket YES by enough → buy YES on Kalshi.
If Kalshi YES > Polymarket YES by enough → buy NO on Kalshi (equivalent to shorting YES).
"""

import re
import time
from typing import Optional
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


KALSHI_FEE = 0.03
POLY_FEE   = 0.02
TOTAL_FEE  = KALSHI_FEE + POLY_FEE - 0.01  # slight reduction to catch more opps


def _normalize(text: str) -> str:
    """Lowercase + strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower())


def _keyword_overlap(a: str, b: str) -> float:
    wa = set(_normalize(a).split())
    wb = set(_normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class ArbitrageStrategy:
    def __init__(
        self,
        kalshi: KalshiClient,
        poly: PolymarketClient,
        risk: RiskManager,
        min_edge: float = 0.04,
    ):
        self.kalshi = kalshi
        self.poly = poly
        self.risk = risk
        self.min_edge = min_edge
        self._poly_cache: list[dict] = []
        self._last_entry: dict[str, float] = {}   # ticker -> timestamp
        self._cooldown = 600  # 10 min between re-entries on same ticker

    def _refresh_poly(self):
        try:
            self._poly_cache = self.poly.get_markets(limit=200)
        except Exception as e:
            log.warning(f"[arb] Polymarket fetch failed: {e}")

    def _find_poly_match(self, kalshi_title: str) -> Optional[dict]:
        best, best_score = None, 0.4   # minimum threshold
        for m in self._poly_cache:
            q = m.get("question", "")
            score = _keyword_overlap(kalshi_title, q)
            if score > best_score:
                best, best_score = m, score
        return best

    def scan(self) -> list[dict]:
        """Return list of arbitrage opportunities."""
        self._refresh_poly()
        if not self._poly_cache:
            return []

        kalshi_markets = get_liquid_markets(self.kalshi, min_volume=0)

        opps = []
        for km in kalshi_markets:
            k_yes_ask = float(km.get("yes_ask_dollars") or 0)
            k_yes_bid = float(km.get("yes_bid_dollars") or 0)
            if not k_yes_ask or not k_yes_bid:
                continue

            pm = self._find_poly_match(km.get("title", ""))
            if not pm:
                continue

            p_yes = self.poly.parse_yes_price(pm)
            if p_yes is None:
                continue

            # Edge: how much cheaper is Kalshi YES vs Polymarket YES?
            # Buy YES on Kalshi, implicitly hold equivalent on Polymarket
            edge_buy_yes = p_yes - k_yes_ask - TOTAL_FEE
            # Edge: buy NO on Kalshi (= short YES), Polymarket says YES is overpriced
            edge_buy_no  = k_yes_bid - p_yes - TOTAL_FEE

            if edge_buy_yes > self.min_edge:
                opps.append({
                    "type": "arb",
                    "side": "yes",
                    "ticker": km["ticker"],
                    "kalshi_price": k_yes_ask,
                    "poly_price": p_yes,
                    "edge": round(edge_buy_yes, 4),
                    "kalshi_title": km.get("title"),
                    "poly_question": pm.get("question"),
                })
            elif edge_buy_no > self.min_edge:
                opps.append({
                    "type": "arb",
                    "side": "no",
                    "ticker": km["ticker"],
                    "kalshi_price": 1 - k_yes_bid,  # NO price
                    "poly_price": 1 - p_yes,
                    "edge": round(edge_buy_no, 4),
                    "kalshi_title": km.get("title"),
                    "poly_question": pm.get("question"),
                })

        opps.sort(key=lambda x: x["edge"], reverse=True)
        return opps

    def execute(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_f = opp["kalshi_price"]
        price_c = int(round(price_f * 100))
        edge    = opp["edge"]

        contracts = self.risk.kelly_contracts(price_c, price_f + edge, max_contracts=50)
        if contracts == 0:
            return False

        if time.time() - self._last_entry.get(ticker, 0) < self._cooldown:
            return False

        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[arb] Skipped {ticker}: {reason}")
            return False

        try:
            result = self.kalshi.place_order(
                ticker=ticker,
                side=side,
                action="buy",
                count=contracts,
                order_type="limit",
                yes_price=price_c if side == "yes" else None,
                no_price=price_c  if side == "no"  else None,
            )
            self.risk.record_open(ticker, contracts)
            self._last_entry[ticker] = time.time()
            log.info(
                f"[arb] BUY {side.upper()} {ticker} x{contracts} @ {price_c}¢  "
                f"edge={edge*100:.1f}¢  order_id={result.get('order', {}).get('order_id')}"
            )
            return True
        except Exception as e:
            log.error(f"[arb] Order failed {ticker}: {e}")
            return False
