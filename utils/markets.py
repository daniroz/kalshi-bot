"""
Fetch liquid markets from known active Kalshi series.
The default /markets endpoint returns mostly illiquid KXMVE esports markets.
This module targets series with real trading volume.
"""

from clients.kalshi import KalshiClient
from utils.logger import log
import time

# Series with consistent real trading volume on Kalshi
LIQUID_SERIES = [
    # Financials
    "KXFED",
    "KXINX",
    "KXNDAQ",
    "KXBTC",
    "KXETH",
    "KXSPXCLOSE",
    "KXNASDAQCLOSE",
    "KXGDP",
    "KXCPI",
    "KXUNEMPLOYMENT",
    "KXNFP",
    "KXOIL",
    "KXGOLD",
    # Politics
    "KXTRUMPPOLLDAILY",
    # Sports — game winners (high volume, 24/7)
    "KXNBAGAME",
    "KXNHLGAME",
    "KXMLBGAME",
    "KXNBA",
    "KXNHL",
    "KXMLB",
]

_cache: list[dict] = []
_cache_ts: float = 0
CACHE_TTL = 120  # refresh every 2 min


def get_liquid_markets(kalshi: KalshiClient, min_volume: float = 0) -> list[dict]:
    global _cache, _cache_ts

    if time.time() - _cache_ts < CACHE_TTL and _cache:
        return [m for m in _cache if float(m.get("volume_24h_fp") or 0) >= min_volume]

    markets = []
    seen = set()

    for series in LIQUID_SERIES:
        try:
            r = kalshi._get("/markets", {"limit": 50, "series_ticker": series})
            for m in r.get("markets", []):
                if m["ticker"] not in seen and m.get("status") == "active":
                    yes_ask = float(m.get("yes_ask_dollars") or 0)
                    if yes_ask > 0:
                        markets.append(m)
                        seen.add(m["ticker"])
        except Exception as e:
            log.warning(f"[markets] {series} fetch failed: {e}")

    # Also grab top markets by volume from general endpoint
    try:
        r = kalshi._get("/markets", {"limit": 200})
        for m in r.get("markets", []):
            if m["ticker"] not in seen and m.get("status") == "active":
                vol = float(m.get("volume_24h_fp") or 0)
                if vol > 100:
                    markets.append(m)
                    seen.add(m["ticker"])
    except Exception:
        pass

    markets.sort(key=lambda m: float(m.get("volume_24h_fp") or 0), reverse=True)
    _cache = markets
    _cache_ts = time.time()

    log.info(f"[markets] Loaded {len(markets)} liquid markets")
    return [m for m in markets if float(m.get("volume_24h_fp") or 0) >= min_volume]
