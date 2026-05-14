"""Kalshi REST API client — uses official SDK auth."""

import httpx
import kalshi_python
from kalshi_python.api_client import ApiClient as _SdkClient


PROD_URL = "https://external-api.kalshi.com/trade-api/v2"
DEMO_URL = "https://external-api.demo.kalshi.co/trade-api/v2"


class KalshiClient:
    def __init__(self, api_key: str, key_file: str, demo: bool = True):
        self.base_url = DEMO_URL if demo else PROD_URL
        self.api_key = api_key
        self.demo = demo

        config = kalshi_python.Configuration(host=self.base_url)
        self._sdk = _SdkClient(config)
        self._sdk.set_kalshi_auth(api_key, key_file)

        self._client = httpx.Client(
            timeout=10.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    def _headers(self, method: str, path: str) -> dict:
        full_url = self.base_url + path
        hdrs = self._sdk.kalshi_auth.create_auth_headers(method.upper(), full_url)
        hdrs["Content-Type"] = "application/json"
        return hdrs

    def _get(self, path: str, params: dict = None):
        hdrs = self._headers("GET", path)
        r = self._client.get(self.base_url + path, headers=hdrs, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict):
        hdrs = self._headers("POST", path)
        r = self._client.post(self.base_url + path, headers=hdrs, json=body)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str):
        hdrs = self._headers("DELETE", path)
        r = self._client.delete(self.base_url + path, headers=hdrs)
        r.raise_for_status()
        return r.json()

    # ── Market data ──────────────────────────────────────────────────────────

    def get_markets(self, limit: int = 200, cursor: str = None, status: str = None) -> dict:
        params = {"limit": limit}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets", params)

    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        return self._get(f"/markets/{ticker}/orderbook", {"depth": depth})

    def get_trades(self, ticker: str, limit: int = 100) -> dict:
        return self._get(f"/markets/{ticker}/trades", {"limit": limit})

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balance(self) -> dict:
        r = self._get("/portfolio/balance")
        # Kalshi returns balance in cents — normalize to dollars
        if "balance" in r and r["balance"] > 1000:
            r["balance"] = r["balance"] / 100
        return r

    def get_positions(self) -> dict:
        return self._get("/portfolio/positions")

    def get_fills(self, ticker: str = None, limit: int = 100) -> dict:
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return self._get("/portfolio/fills", params)

    # ── Orders ────────────────────────────────────────────────────────────────

    def get_orders(self, status: str = "resting") -> dict:
        return self._get("/portfolio/orders", {"status": status})

    def place_order(
        self,
        ticker: str,
        side: str,       # "yes" or "no"
        action: str,     # "buy" or "sell"
        count: int,      # number of contracts
        order_type: str, # "limit" or "market"
        yes_price: int = None,  # cents (1-99)
        no_price: int = None,
        client_order_id: str = None,
    ) -> dict:
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._delete(f"/portfolio/orders/{order_id}")

    def cancel_all_orders(self):
        orders = self.get_orders(status="resting").get("orders", [])
        cancelled = []
        for o in orders:
            try:
                self.cancel_order(o["order_id"])
                cancelled.append(o["order_id"])
            except Exception:
                pass
        return cancelled
