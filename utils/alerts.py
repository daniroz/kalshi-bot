"""
Push notifications via ntfy.sh (free, no account needed).

Setup:
  1. Install ntfy app on your phone (iOS/Android)
  2. Add NTFY_TOPIC=your-unique-topic to .env  (e.g. daniroz-kalshi-bot)
  3. In the app, subscribe to that topic

Then get alerts on every big trade, error, or balance drop.
"""

import os
import httpx
from utils.logger import log

_TOPIC = os.getenv("NTFY_TOPIC", "")
_BASE  = "https://ntfy.sh"


def _send(title: str, body: str, priority: str = "default", tags: list[str] = None):
    if not _TOPIC:
        return
    try:
        headers = {
            "Title":    title,
            "Priority": priority,
        }
        if tags:
            headers["Tags"] = ",".join(tags)
        httpx.post(f"{_BASE}/{_TOPIC}", content=body, headers=headers, timeout=5)
    except Exception as e:
        log.debug(f"[alerts] send failed: {e}")


def alert_trade(action: str, strategy: str, ticker: str, side: str, qty: int, price_c: int, edge: float):
    emoji = "📈" if action == "ENTER" else "📉"
    title = f"{emoji} {action} — {strategy.upper()}"
    body  = f"{ticker}\n{side.upper()} x{qty} @ {price_c}¢  edge={edge*100:.1f}¢"
    _send(title, body, priority="default", tags=["chart_increasing" if action == "ENTER" else "chart_decreasing"])


def alert_error(strategy: str, msg: str):
    _send(f"⚠️ Bot Error — {strategy}", msg, priority="high", tags=["warning"])


def alert_balance(balance: float, pnl: float, pnl_pct: float):
    emoji = "🟢" if pnl >= 0 else "🔴"
    _send(
        f"{emoji} Balance Update",
        f"${balance:.2f}  ({'+' if pnl>=0 else ''}{pnl:.2f}, {pnl_pct:+.1f}%)",
        priority="low",
    )


def alert_halt(reason: str):
    _send("🛑 Bot Halted", reason, priority="urgent", tags=["rotating_light"])
