"""
Polymarket order-execution client — DRY_RUN by default.

The existing PolymarketClient is read-only (Gamma API for market data).
This module adds the second leg of cross-platform arb: actually placing
orders on Polymarket's CLOB.

REAL order placement needs:
  - py-clob-client (or manual EIP-712 signing)
  - USDC funded on Polygon (your bridge from L1)
  - POLYMARKET_PRIVATE_KEY (EOA) + POLYMARKET_SAFE_ADDRESS (your Safe wallet)
  - POLYMARKET_API_KEY / POLYMARKET_API_SECRET / POLYMARKET_API_PASSPHRASE

Until those are set, DRY_RUN mode logs intended orders without placing them.
When you're ready:
  1. Fund USDC on Polygon to your Safe wallet
  2. pip install py-clob-client
  3. Set the env vars above
  4. Flip POLYMARKET_TRADE_ENABLED=true in .env
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from utils.logger import log


@dataclass
class PolyOrderResult:
    success:     bool
    order_id:    Optional[str]
    filled_qty:  float
    avg_price:   float
    error:       Optional[str] = None
    dry_run:     bool = False


def _trade_enabled() -> bool:
    """Real trading is enabled only when explicitly opted in AND credentials exist."""
    if os.getenv("POLYMARKET_TRADE_ENABLED", "false").lower() != "true":
        return False
    required = ("POLYMARKET_PRIVATE_KEY","POLYMARKET_SAFE_ADDRESS")
    return all(os.environ.get(k) for k in required)


class PolymarketTrader:
    """Order placement on Polymarket. Falls back to DRY_RUN when not configured."""

    def __init__(self):
        self.enabled = _trade_enabled()
        self._client = None
        if self.enabled:
            try:
                # Lazy import — only if user opted in
                from py_clob_client.client import ClobClient    # type: ignore
                from py_clob_client.constants import POLYGON   # type: ignore
                self._client = ClobClient(
                    host="https://clob.polymarket.com",
                    key=os.environ["POLYMARKET_PRIVATE_KEY"],
                    chain_id=POLYGON,
                    funder=os.environ["POLYMARKET_SAFE_ADDRESS"],
                    signature_type=2,  # Safe wallet
                )
                # Optional API credentials
                if os.environ.get("POLYMARKET_API_KEY"):
                    self._client.set_api_creds({
                        "key":        os.environ["POLYMARKET_API_KEY"],
                        "secret":     os.environ["POLYMARKET_API_SECRET"],
                        "passphrase": os.environ["POLYMARKET_API_PASSPHRASE"],
                    })
                log.info("[poly-trader] live trading ENABLED")
            except ImportError:
                log.warning("[poly-trader] py-clob-client not installed — falling back to DRY_RUN")
                self.enabled = False
            except Exception as e:
                log.error(f"[poly-trader] init failed: {e} — DRY_RUN")
                self.enabled = False
        else:
            log.info("[poly-trader] DRY_RUN mode (no POLYMARKET_TRADE_ENABLED or missing keys)")

    def place_order(
        self,
        token_id:  str,
        side:      str,         # "BUY" or "SELL"
        price:     float,       # 0.0 - 1.0
        size:      float,       # # of contracts
        timeout_s: int = 10,
    ) -> PolyOrderResult:
        """Place an order on Polymarket CLOB. Returns result with fill info."""
        if not self.enabled or not self._client:
            log.info(f"[poly-trader] DRY_RUN: would {side} {size} @ ${price:.4f}  token={token_id[:12]}...")
            return PolyOrderResult(
                success=True, order_id=f"dry_{int(time.time())}",
                filled_qty=size, avg_price=price, dry_run=True,
            )

        # REAL order — limit order, GTC. The CLOB SDK does signing automatically.
        try:
            from py_clob_client.clob_types import OrderArgs    # type: ignore
            args = OrderArgs(price=price, size=size, side=side.upper(), token_id=token_id)
            signed = self._client.create_order(args)
            resp = self._client.post_order(signed)   # GTC by default
            return PolyOrderResult(
                success=bool(resp.get("success")),
                order_id=resp.get("orderID"),
                filled_qty=float(resp.get("makingAmount", 0)),
                avg_price=price,
                error=resp.get("errorMsg"),
            )
        except Exception as e:
            return PolyOrderResult(False, None, 0.0, 0.0, error=str(e))

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order."""
        if not self.enabled or not self._client:
            log.info(f"[poly-trader] DRY_RUN: would cancel {order_id}")
            return True
        try:
            self._client.cancel(order_id)
            return True
        except Exception as e:
            log.error(f"[poly-trader] cancel failed {order_id}: {e}")
            return False
