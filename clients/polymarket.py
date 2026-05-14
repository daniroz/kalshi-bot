"""Polymarket public read-only client — no auth required."""

import httpx
from typing import Optional

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL  = "https://clob.polymarket.com"


class PolymarketClient:
    def __init__(self):
        self._client = httpx.Client(timeout=10.0)

    def get_markets(self, limit: int = 100, offset: int = 0, active: bool = True) -> list[dict]:
        params = {"limit": limit, "offset": offset, "active": str(active).lower()}
        r = self._client.get(f"{GAMMA_URL}/markets", params=params)
        r.raise_for_status()
        return r.json()

    def search_markets(self, query: str, limit: int = 20) -> list[dict]:
        params = {"q": query, "limit": limit, "active": "true"}
        r = self._client.get(f"{GAMMA_URL}/markets", params=params)
        r.raise_for_status()
        return r.json()

    def get_market_by_slug(self, slug: str) -> Optional[dict]:
        params = {"slug": slug}
        r = self._client.get(f"{GAMMA_URL}/markets", params=params)
        r.raise_for_status()
        data = r.json()
        return data[0] if data else None

    def get_clob_price(self, condition_id: str) -> Optional[dict]:
        """Get current best bid/ask from the CLOB for a condition."""
        try:
            r = self._client.get(f"{CLOB_URL}/book", params={"token_id": condition_id})
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def parse_yes_price(self, market: dict) -> Optional[float]:
        """Extract the YES mid-price (0-1) from a Gamma market object."""
        prices = market.get("outcomePrices")
        if not prices:
            return None
        try:
            # outcomePrices is a JSON-encoded list like '["0.54", "0.46"]'
            import json
            if isinstance(prices, str):
                prices = json.loads(prices)
            return float(prices[0])
        except Exception:
            return None
