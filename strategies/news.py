"""
News sentiment strategy.

Polls Reuters, AP, and NPR RSS feeds every 5 minutes for breaking headlines.
Matches headlines to liquid Kalshi markets using keyword/entity overlap.
If a relevant headline breaks and Kalshi's price hasn't moved yet, trades
in the direction suggested by the headline sentiment.

Sentiment is determined by comparing headline keywords against bullish/bearish
word lists relative to the market's YES outcome.

Entry: relevant headline + Kalshi price stale (< 2¢ move since headline)
Exit:  8-cent profit | 5-cent stop | 20 min hold max
"""

import re
import time
import math
import feedparser
import httpx
from datetime import datetime, timezone
from dataclasses import dataclass
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/Reuters/PoliticsNews",
    "https://feeds.apnews.com/rss/apf-business",
    "https://feeds.npr.org/1014/rss.xml",       # politics
    "https://feeds.npr.org/1006/rss.xml",        # economy
]

POLL_INTERVAL  = 300    # 5 min between RSS polls
MIN_EDGE       = 0.06   # 6 cent minimum
PRICE_STALE_C  = 3      # headline is only "fresh" if price moved < 3¢ after it
MAX_HOLD_S     = 1200   # exit after 20 min regardless
PROFIT_TARGET  = 8      # cents
STOP_LOSS      = 5      # cents
COOLDOWN_S     = 600


# Keywords that suggest YES is more likely (bullish for the outcome)
BULLISH_WORDS = {
    "rises", "rise", "surges", "surged", "beats", "beat", "above", "higher",
    "wins", "win", "passes", "passed", "gains", "gained", "increases", "increased",
    "jumps", "jumped", "rallies", "rallied", "up", "record", "strong", "better",
    "approves", "approved", "confirms", "confirmed", "supports", "expands",
}

BEARISH_WORDS = {
    "falls", "fall", "drops", "drop", "misses", "miss", "below", "lower",
    "loses", "lose", "fails", "failed", "declines", "declined", "decreases",
    "decreased", "cuts", "cut", "reduces", "reduced", "weak", "worse",
    "rejects", "rejected", "denies", "denied", "contracts", "shrinks",
}

# Market keyword → RSS keyword mapping
MARKET_KEYWORDS = {
    "fed":        ["fed", "federal reserve", "fomc", "interest rate", "powell"],
    "cpi":        ["cpi", "inflation", "consumer price", "prices rose", "prices fell"],
    "nfp":        ["jobs", "payroll", "unemployment", "labor", "hiring", "layoff"],
    "gdp":        ["gdp", "growth", "recession", "economy", "economic"],
    "btc":        ["bitcoin", "btc", "crypto", "cryptocurrency"],
    "eth":        ["ethereum", "eth"],
    "trump":      ["trump", "president", "white house", "executive order"],
    "spx":        ["s&p", "sp500", "stock market", "equities", "nasdaq", "dow"],
    "oil":        ["oil", "crude", "opec", "energy prices"],
    "gold":       ["gold", "precious metals"],
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _sentiment(headline: str) -> int:
    """Return +1 (bullish), -1 (bearish), 0 (neutral)."""
    words = set(_normalize(headline).split())
    bull = len(words & BULLISH_WORDS)
    bear = len(words & BEARISH_WORDS)
    if bull > bear:
        return 1
    if bear > bull:
        return -1
    return 0


def _market_category(title: str):
    t = title.lower()
    for cat, keywords in MARKET_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return cat
    return None


def _headline_matches_market(headline: str, market_title: str) -> bool:
    h_words = set(_normalize(headline).split()) - {"the", "a", "an", "is", "in", "of", "to", "and"}
    m_words = set(_normalize(market_title).split()) - {"the", "a", "an", "is", "in", "of", "to", "and", "will", "close"}
    overlap = len(h_words & m_words)
    return overlap >= 2


@dataclass
class NewsPosition:
    ticker: str
    side: str
    contracts: int
    entry_price_c: int
    entry_time: float
    high_water_c: int


class NewsStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi   = kalshi
        self.risk     = risk
        self._last_poll:    float = 0
        self._headlines:    list[dict] = []   # {text, ts, sentiment, category}
        self._price_before: dict[str, int] = {}   # ticker -> price before headline
        self._positions:    dict[str, NewsPosition] = {}
        self._last_entry:   dict[str, float] = {}

    def _poll_rss(self):
        if time.time() - self._last_poll < POLL_INTERVAL:
            return
        self._last_poll = time.time()
        new_headlines = []
        cutoff = time.time() - 3600   # only last hour

        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:
                    ts = 0
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        ts = time.mktime(entry.published_parsed)
                    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        ts = time.mktime(entry.updated_parsed)
                    else:
                        ts = time.time()
                    if ts < cutoff:
                        continue
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title} {summary}"
                    sent = _sentiment(text)
                    cat  = _market_category(text)
                    if cat:
                        new_headlines.append({
                            "text":      title,
                            "full":      text,
                            "ts":        ts,
                            "sentiment": sent,
                            "category":  cat,
                        })
            except Exception as e:
                log.debug(f"[news] RSS fetch failed {url}: {e}")

        if new_headlines:
            self._headlines = new_headlines
            log.info(f"[news] Loaded {len(new_headlines)} relevant headlines")

    def _snapshot_prices(self, markets: list[dict]):
        """Store current prices before we act on headlines."""
        for m in markets:
            ticker = m["ticker"]
            if ticker not in self._price_before:
                mid = int((float(m.get("yes_bid_dollars") or 0) + float(m.get("yes_ask_dollars") or 0)) / 2 * 100)
                self._price_before[ticker] = mid

    def _check_exits(self):
        now = time.time()
        for ticker, pos in list(self._positions.items()):
            try:
                m = self.kalshi.get_market(ticker).get("market", {})
                bid_c = int(float(m.get("yes_bid_dollars" if pos.side == "yes" else "no_bid_dollars") or 0) * 100)
                if not bid_c:
                    continue
                # Update high water mark
                if bid_c > pos.high_water_c:
                    pos.high_water_c = bid_c
                # Trailing stop: exit if 4¢ below high water
                trail_stop = pos.high_water_c - 4
                pnl = bid_c - pos.entry_price_c
                if now - pos.entry_time > MAX_HOLD_S:
                    self._exit(pos, f"max hold time  pnl={pnl:+}¢")
                elif pnl >= PROFIT_TARGET:
                    self._exit(pos, f"profit +{pnl}¢")
                elif pnl <= -STOP_LOSS:
                    self._exit(pos, f"stop {pnl}¢")
                elif bid_c < trail_stop:
                    self._exit(pos, f"trailing stop (peak={pos.high_water_c}¢ now={bid_c}¢)")
            except Exception as e:
                log.debug(f"[news] Exit check {ticker}: {e}")

    def _exit(self, pos: NewsPosition, reason: str):
        try:
            self.kalshi.place_order(
                ticker=pos.ticker, side=pos.side, action="sell",
                count=pos.contracts, order_type="market",
            )
            self.risk.record_close(pos.ticker, pos.contracts)
            log.info(f"[news] EXIT {pos.ticker} {pos.side.upper()} x{pos.contracts}  {reason}")
        except Exception as e:
            log.error(f"[news] Exit failed {pos.ticker}: {e}")
        finally:
            self._positions.pop(pos.ticker, None)

    def scan(self) -> list[dict]:
        self._poll_rss()
        if not self._headlines:
            return []

        markets = get_liquid_markets(self.kalshi, min_volume=50)
        self._snapshot_prices(markets)
        self._check_exits()

        signals = []
        now = time.time()
        recent_headlines = [h for h in self._headlines if now - h["ts"] < 3600]

        for m in markets:
            ticker = m["ticker"]
            if ticker in self._positions:
                continue
            if now - self._last_entry.get(ticker, 0) < COOLDOWN_S:
                continue

            yes_ask_c = int(float(m.get("yes_ask_dollars") or 0) * 100)
            yes_bid_c = int(float(m.get("yes_bid_dollars") or 0) * 100)
            no_ask_c  = int(float(m.get("no_ask_dollars")  or 0) * 100)
            if not yes_ask_c or not yes_bid_c:
                continue

            market_cat = _market_category(m.get("title", ""))
            if not market_cat:
                continue

            # Find the most recent matching headline
            best_h = None
            for h in sorted(recent_headlines, key=lambda x: x["ts"], reverse=True):
                if h["category"] == market_cat and h["sentiment"] != 0:
                    if _headline_matches_market(h["text"], m.get("title", "")):
                        best_h = h
                        break

            if not best_h:
                continue

            # Check if price has already moved in response (staleness check)
            pre_price = self._price_before.get(ticker, 0)
            mid_c = (yes_ask_c + yes_bid_c) // 2
            if pre_price and abs(mid_c - pre_price) > PRICE_STALE_C:
                continue   # market already moved, too late

            sentiment = best_h["sentiment"]
            age_min = (now - best_h["ts"]) / 60

            # Bullish headline → buy YES (expect price to rise)
            if sentiment == 1 and yes_ask_c < 70:
                edge = MIN_EDGE
                signals.append({
                    "ticker":    ticker,
                    "title":     m.get("title", "")[:60],
                    "side":      "yes",
                    "price_c":   yes_ask_c,
                    "edge":      edge,
                    "headline":  best_h["text"][:80],
                    "age_min":   round(age_min, 1),
                })
            # Bearish headline → buy NO (expect YES price to fall)
            elif sentiment == -1 and no_ask_c and no_ask_c < 70:
                edge = MIN_EDGE
                signals.append({
                    "ticker":    ticker,
                    "title":     m.get("title", "")[:60],
                    "side":      "no",
                    "price_c":   no_ask_c,
                    "edge":      edge,
                    "headline":  best_h["text"][:80],
                    "age_min":   round(age_min, 1),
                })

        signals.sort(key=lambda x: x["age_min"])   # freshest headlines first
        return signals

    def execute(self, signal: dict) -> bool:
        ticker  = signal["ticker"]
        side    = signal["side"]
        price_c = signal["price_c"]
        edge    = signal["edge"]

        contracts = self.risk.kelly_contracts(price_c, price_c / 100 + edge, max_contracts=30)
        contracts = max(1, contracts)

        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[news] Skipped {ticker}: {reason}")
            return False

        try:
            self.kalshi.place_order(
                ticker=ticker, side=side, action="buy",
                count=contracts, order_type="limit",
                yes_price=price_c if side == "yes" else None,
                no_price=price_c  if side == "no"  else None,
            )
            self.risk.record_open(ticker, contracts)
            self._last_entry[ticker] = time.time()
            self._positions[ticker] = NewsPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, entry_time=time.time(),
                high_water_c=price_c,
            )
            log.info(
                f"[news] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢"
                f"  headline: \"{signal['headline']}\""
                f"  age={signal['age_min']}min"
            )
            return True
        except Exception as e:
            log.error(f"[news] Order failed {ticker}: {e}")
            return False
