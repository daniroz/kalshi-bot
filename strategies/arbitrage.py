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
import math
from typing import Optional
from collections import Counter
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.fill_tracker import FillTracker
from utils.logger import log


KALSHI_FEE = 0.03
POLY_FEE   = 0.02
TOTAL_FEE  = KALSHI_FEE + POLY_FEE - 0.01


STOPWORDS = {"the","a","an","is","in","of","to","and","or","will","by","on",
             "at","be","it","for","as","this","that","with","are","was","not",
             "above","below","close","end","yes","no"}

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())

def _tokens(text: str) -> list[str]:
    return [t for t in _normalize(text).split() if t not in STOPWORDS and len(t) > 1]

def _tfidf_score(a: str, b: str, corpus: list[str]) -> float:
    """TF-IDF cosine similarity between two strings given a corpus."""
    all_docs = corpus + [a, b]
    N = len(all_docs)
    df: Counter = Counter()
    for doc in all_docs:
        for tok in set(_tokens(doc)):
            df[tok] += 1

    def vec(text: str) -> dict:
        toks = _tokens(text)
        tf = Counter(toks)
        total = len(toks) or 1
        return {t: (tf[t] / total) * math.log(N / (df[t] + 1)) for t in tf}

    va, vb = vec(a), vec(b)
    keys = set(va) & set(vb)
    if not keys:
        return 0.0
    dot = sum(va[k] * vb[k] for k in keys)
    na  = math.sqrt(sum(v*v for v in va.values())) or 1
    nb  = math.sqrt(sum(v*v for v in vb.values())) or 1
    return dot / (na * nb)

def _extract_numbers(text: str) -> set[str]:
    """Extract numeric tokens (years, dollar amounts, percentages)."""
    return set(re.findall(r'\b\d+\.?\d*\b', text))


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
        self._poly_titles: list[str] = []
        self._last_entry: dict[str, float] = {}
        self._cooldown = 600
        self._fills = FillTracker(kalshi)

    def _refresh_poly(self):
        try:
            self._poly_cache = self.poly.get_markets(limit=200)
            self._poly_titles = [m.get("question", "") for m in self._poly_cache]
        except Exception as e:
            log.warning(f"[arb] Polymarket fetch failed: {e}")

    def _find_poly_match(self, kalshi_title: str) -> Optional[dict]:
        if not self._poly_cache:
            return None
        best, best_score = None, 0.25   # min threshold (higher = stricter)
        k_nums = _extract_numbers(kalshi_title)
        for m in self._poly_cache:
            q = m.get("question", "")
            # Numbers must overlap (year, strike price, etc.)
            p_nums = _extract_numbers(q)
            if k_nums and p_nums and not (k_nums & p_nums):
                continue   # different numbers = different markets
            score = _tfidf_score(kalshi_title, q, self._poly_titles)
            if score > best_score:
                best, best_score = m, score
        return best

    def scan(self) -> list[dict]:
        """Return list of arbitrage opportunities."""
        self._fills.check_stale(self.risk)
        self._refresh_poly()
        if not self._poly_cache:
            return []

        kalshi_markets = get_liquid_markets(self.kalshi, min_volume=0)

        # Sports markets (per-game and season-level) are handled by the sports strategy.
        # Polymarket doesn't reliably carry matching markets for these.
        SKIP_PREFIXES = ("KXNBAGAME", "KXNHLGAME", "KXMLBGAME", "KXNBA", "KXNHL", "KXMLB")

        opps = []
        for km in kalshi_markets:
            if km["ticker"].startswith(SKIP_PREFIXES):
                continue

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
            order_id = result.get("order", {}).get("order_id", "")
            self.risk.record_open(ticker, contracts)
            self._last_entry[ticker] = time.time()
            self._fills.track(order_id, ticker, contracts)
            log.info(
                f"[arb] BUY {side.upper()} {ticker} x{contracts} @ {price_c}¢  "
                f"edge={edge*100:.1f}¢  order_id={order_id}"
            )
            return True
        except Exception as e:
            log.error(f"[arb] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            return False
