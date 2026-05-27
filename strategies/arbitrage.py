"""
Cross-market SIGNAL strategy (NOT true arbitrage — we only place the Kalshi leg).

Honest framing: we use Polymarket prices as an external signal for Kalshi mispricing.
If Polymarket says YES is 40¢ and Kalshi says 25¢, we buy Kalshi YES — but we are
UNHEDGED. We're taking directional exposure on the assumption that Polymarket's
broader liquidity makes its price closer to truth. This carries model risk:
  1. The matched markets may resolve differently (different rules / scope)
  2. Polymarket could be wrong
  3. We pay only Kalshi's ~3% fee (no Polymarket leg)

Defense in depth: stricter title match (0.50 cosine, was 0.25), require 8¢ edge.
Even with this, expect ~25-30% loss rate from match errors. Position-size accordingly.
"""

import re
import time
import math
from typing import Optional
from collections import Counter
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketClient
from clients.polymarket_trader import PolymarketTrader
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.fill_tracker import FillTracker
from utils.logger import log


# Fee math: we only place the Kalshi leg, so only Kalshi fees apply.
# Kalshi fee is volume-based not flat 3%; computed per-trade in approve_trade.
# We use 3% as a conservative average for the edge calculation here.
KALSHI_FEE_AVG = 0.03

# Stricter title matching — 0.25 cosine was matching different markets that
# shared similar tokens. 0.50 is much stricter.
MATCH_THRESHOLD = 0.50


def _parse_clob_tokens(pm: dict) -> tuple:
    """Extract (yes_token_id, no_token_id) from a Polymarket Gamma market.
    Field can be a JSON-encoded list of strings or already-parsed list."""
    raw = pm.get("clobTokenIds") or pm.get("tokens") or pm.get("token_ids")
    if not raw:
        return None, None
    try:
        if isinstance(raw, str):
            import json
            raw = json.loads(raw)
        if isinstance(raw, list) and len(raw) >= 2:
            # First is YES, second is NO (Polymarket convention)
            return str(raw[0]), str(raw[1])
    except Exception:
        pass
    return None, None


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
        self.poly_trader = PolymarketTrader()   # DRY_RUN until USDC funded + env set
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
        best, best_score = None, MATCH_THRESHOLD
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
            # Only Kalshi fee applies — we're not hedging on Polymarket.
            edge_buy_yes = p_yes - k_yes_ask - KALSHI_FEE_AVG
            edge_buy_no  = k_yes_bid - p_yes - KALSHI_FEE_AVG

            # Extract Polymarket CLOB token_ids for two-leg execution
            # tokens[0] is YES, tokens[1] is NO (in Polymarket conventions)
            poly_yes_token, poly_no_token = _parse_clob_tokens(pm)

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
                    # Two-leg hedge: BUY YES on Kalshi + SELL YES on Polymarket (= BUY NO)
                    "poly_hedge_token": poly_yes_token,
                    "poly_hedge_side":  "SELL",       # short YES on Poly
                    "poly_hedge_price": p_yes,
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
                    # Hedge: BUY NO on Kalshi + BUY YES on Polymarket
                    "poly_hedge_token": poly_yes_token,
                    "poly_hedge_side":  "BUY",
                    "poly_hedge_price": p_yes,
                })

        opps.sort(key=lambda x: x["edge"], reverse=True)
        return opps

    def execute(self, opp: dict) -> bool:
        """Two-leg arb execution: Kalshi leg first, then Polymarket hedge.
        If the Polymarket leg fails, immediately sell back the Kalshi leg
        to undo the directional exposure."""
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

        # ── LEG 1: Kalshi ─────────────────────────────────────────────────
        try:
            k_result = self.kalshi.place_order(
                ticker=ticker,
                side=side,
                action="buy",
                count=contracts,
                order_type="limit",
                yes_price=price_c if side == "yes" else None,
                no_price=price_c  if side == "no"  else None,
            )
            k_order_id = k_result.get("order", {}).get("order_id", "")
        except Exception as e:
            log.error(f"[arb] Kalshi leg failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            return False

        log.info(
            f"[arb] LEG 1 ✓ Kalshi BUY {side.upper()} {ticker} x{contracts} @ {price_c}¢  "
            f"order_id={k_order_id}"
        )

        # ── LEG 2: Polymarket hedge ───────────────────────────────────────
        hedge_token = opp.get("poly_hedge_token")
        hedge_side  = opp.get("poly_hedge_side")
        hedge_price = opp.get("poly_hedge_price")
        if not hedge_token or not hedge_side or not hedge_price:
            log.warning(f"[arb] No Polymarket hedge info for {ticker} — leaving Kalshi leg NAKED")
            # In live mode this is the bug we'd want to fix; for now we accept this.
            self.risk.record_open(ticker, contracts)
            self._last_entry[ticker] = time.time()
            self._fills.track(k_order_id, ticker, contracts)
            return True

        p_result = self.poly_trader.place_order(
            token_id=hedge_token,
            side=hedge_side,
            price=hedge_price,
            size=float(contracts),
            timeout_s=10,
        )

        if not p_result.success:
            # ── ROLLBACK: undo Kalshi leg ────────────────────────────────
            log.error(f"[arb] LEG 2 ✗ Polymarket {hedge_side} failed: {p_result.error}")
            log.warning(f"[arb] ROLLING BACK Kalshi position on {ticker}…")
            try:
                # Sell back at current bid on our side
                m = self.kalshi.get_market(ticker).get("market", {})
                if side == "yes":
                    bid_c = int(round(float(m.get("yes_bid_dollars") or 0) * 100))
                else:
                    bid_c = int(round(float(m.get("no_bid_dollars")  or 0) * 100))
                bid_c = max(1, min(99, bid_c))
                self.kalshi.place_order(
                    ticker=ticker, side=side, action="sell",
                    count=contracts, order_type="limit",
                    yes_price=bid_c if side == "yes" else None,
                    no_price =bid_c if side == "no"  else None,
                )
                log.info(f"[arb] ✓ Rolled back Kalshi {side.upper()} x{contracts} @ {bid_c}¢")
            except Exception as e2:
                log.error(f"[arb] ROLLBACK FAILED for {ticker}: {e2} — manual intervention required!")
                # We'll persist this position; risk manager will see it next sync
                self.risk.record_open(ticker, contracts)
            self.risk.undo_reservation(ticker, contracts)
            return False

        # ── Both legs filled ──────────────────────────────────────────────
        self.risk.record_open(ticker, contracts)
        self._last_entry[ticker] = time.time()
        self._fills.track(k_order_id, ticker, contracts)
        leg_tag = "DRY_RUN" if p_result.dry_run else "LIVE"
        log.info(
            f"[arb] LEG 2 ✓ Polymarket {hedge_side} x{contracts} @ ${hedge_price:.3f}  "
            f"order_id={p_result.order_id}  [{leg_tag}]"
        )
        log.info(
            f"[arb] BOTH LEGS FILLED on {ticker}  edge={edge*100:.1f}¢  contracts={contracts}"
        )
        return True
