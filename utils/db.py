"""
SQLite trade log — local mirror of Kalshi fill history + bot signals.

Why: bot.log is 1GB+ of text. Hard to query "what's my P&L by strategy this week"
out of that. This module gives you a real database you can hit with SQL.

Tables:
  fills              — every Kalshi fill (synced from /portfolio/fills)
  signals            — every approve_trade decision (accepted + rejected)
  equity_snapshots   — periodic balance snapshots (for charting / drawdown)

CLI:
  python -m utils.db init               create schema
  python -m utils.db sync               pull latest fills from Kalshi
  python -m utils.db report [--days N]  P&L summary by strategy
  python -m utils.db fills [--days N]   list recent fills
  python -m utils.db signals [--days N] list recent signals
  python -m utils.db schema             dump table schemas

The fills sync is idempotent (INSERT OR IGNORE on fill_id) so it's safe to
run repeatedly. The bot calls sync_recent_fills() every N cycles automatically.

Strategy attribution: we map each fill to its likely strategy via ticker
prefix patterns (KXBTC* → settle-crypto, KXNHLGAME-* → MM or settle-sports,
etc.). Where the prefix is ambiguous (sports tickers fill from both sports
strategies and the market maker), we mark 'mm-or-settle-sports' rather than
guess. For exact attribution we'd need active tagging at order-placement
time — not worth the refactor right now.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "kalshi_bot.sqlite"


# ── Schema ──────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS fills (
    fill_id          TEXT PRIMARY KEY,
    created_time     TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    side             TEXT NOT NULL,
    action           TEXT NOT NULL,
    count            REAL NOT NULL,
    yes_price_cents  INTEGER,
    no_price_cents   INTEGER,
    fee_cents        REAL,
    order_id         TEXT,
    strategy_guess   TEXT,
    inserted_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fills_time     ON fills(created_time);
CREATE INDEX IF NOT EXISTS idx_fills_ticker   ON fills(ticker);
CREATE INDEX IF NOT EXISTS idx_fills_strategy ON fills(strategy_guess);

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    side            TEXT,
    price_cents     INTEGER,
    contracts       INTEGER,
    edge_dollars    REAL,
    accepted        INTEGER NOT NULL,
    reject_reason   TEXT,
    strategy_guess  TEXT
);

CREATE INDEX IF NOT EXISTS idx_signals_time      ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_ticker    ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_strategy  ON signals(strategy_guess);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    timestamp       TEXT PRIMARY KEY,
    equity          REAL NOT NULL,
    cash            REAL NOT NULL,
    position_value  REAL NOT NULL,
    open_positions  INTEGER NOT NULL
);
"""


# ── Strategy-attribution heuristic ──────────────────────────────────────────
def guess_strategy(ticker: str) -> str:
    """Best-effort attribution from ticker prefix. 'unknown' when ambiguous."""
    t = ticker.upper()
    if t.startswith(("KXBTC-", "KXETH-", "KXDOGE-", "KXXRP-")):
        return "settle-crypto"
    if t.startswith(("KXINX-", "KXSPXCLOSE-", "KXNDAQ-", "KXNASDAQCLOS", "KXDJI-")):
        return "settle-stocks"
    if t.startswith("KXWTI-"):
        return "settle-comm"
    if t.startswith(("KXEURUSD-", "KXUSDJPY-")):
        return "settle-fx"
    if t.startswith(("KXHIGH", "KXLOW")):
        return "settle-weather"
    # NHL/NBA/MLB tickers can come from MM OR settle-sports OR the predictive sports strategy
    if t.startswith(("KXNHLGAME-", "KXNBAGAME-", "KXMLBGAME-")):
        return "mm-or-sports"
    # Macro markets only MM quotes them today
    if t.startswith(("KXFED-", "KXCPI-", "KXNFP-", "KXUNEMPLOYMENT-", "KXGDP-", "KXGOLD-")):
        return "mm"
    return "unknown"


# ── Connection / init ───────────────────────────────────────────────────────
_conn: sqlite3.Connection | None = None

def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(DB_PATH), isolation_level=None, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
    return _conn

def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)


# ── Write paths ─────────────────────────────────────────────────────────────
def log_signal(*, ticker: str, side: str | None, price_cents: int | None,
               contracts: int | None, edge_dollars: float | None,
               accepted: bool, reject_reason: str | None = None) -> None:
    """Called from risk.approve_trade for every decision."""
    try:
        get_conn().execute(
            """INSERT INTO signals
               (timestamp, ticker, side, price_cents, contracts,
                edge_dollars, accepted, reject_reason, strategy_guess)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             ticker, side, price_cents, contracts, edge_dollars,
             1 if accepted else 0, reject_reason, guess_strategy(ticker)),
        )
    except Exception:
        # Never let logging take down a trade
        pass

def log_equity(equity: float, cash: float, position_value: float, open_positions: int) -> None:
    try:
        get_conn().execute(
            """INSERT OR REPLACE INTO equity_snapshots
               (timestamp, equity, cash, position_value, open_positions)
               VALUES (?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             equity, cash, position_value, open_positions),
        )
    except Exception:
        pass


# ── Fill sync from Kalshi ───────────────────────────────────────────────────
def sync_fills_from_kalshi(kalshi, max_pages: int = 20) -> int:
    """Pull /portfolio/fills (paginated) and INSERT OR IGNORE new ones.
    Returns the count of NEW fills inserted."""
    init_db()
    conn = get_conn()
    cursor = None
    inserted = 0
    for _ in range(max_pages):
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        try:
            r = kalshi._get("/portfolio/fills", params)
        except Exception:
            break
        batch = r.get("fills", [])
        if not batch:
            break

        for f in batch:
            fill_id = f.get("trade_id") or f.get("fill_id") or ""
            if not fill_id:
                continue
            ticker = f.get("market_ticker", "") or f.get("ticker", "")
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO fills
                       (fill_id, created_time, ticker, side, action,
                        count, yes_price_cents, no_price_cents, fee_cents,
                        order_id, strategy_guess)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        fill_id,
                        f.get("created_time", ""),
                        ticker,
                        (f.get("side") or "").lower(),
                        (f.get("action") or "").lower(),
                        float(f.get("count_fp") or f.get("count") or 0),
                        int(round(float(f.get("yes_price_dollars") or 0) * 100)) if f.get("yes_price_dollars") else None,
                        int(round(float(f.get("no_price_dollars") or 0) * 100)) if f.get("no_price_dollars") else None,
                        float(f.get("fee_cost") or 0),
                        f.get("order_id"),
                        guess_strategy(ticker),
                    ),
                )
                if conn.total_changes:
                    inserted += 1
            except Exception:
                continue

        cursor = r.get("cursor")
        if not cursor:
            break
    return inserted


# ── Read / report paths ─────────────────────────────────────────────────────
def _since(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")

def report_pnl(days: int = 7) -> None:
    """Print a P&L summary by strategy_guess for the last N days."""
    init_db()
    conn = get_conn()
    since = _since(days)

    rows = conn.execute(
        """SELECT strategy_guess,
                  COUNT(*) as fills,
                  SUM(count) as total_contracts,
                  SUM(fee_cents)/100.0 as total_fees_usd
           FROM fills
           WHERE created_time >= ?
           GROUP BY strategy_guess
           ORDER BY fills DESC""",
        (since,),
    ).fetchall()

    if not rows:
        print(f"No fills in last {days} days. (Run `python -m utils.db sync` first.)")
        return

    print(f"\n=== Fill activity, last {days} day(s) ===")
    print(f"{'strategy':22s} {'fills':>7s} {'contracts':>11s} {'fees ($)':>10s}")
    print("-" * 55)
    for r in rows:
        sg = r["strategy_guess"] or "unknown"
        print(f"{sg:22s} {r['fills']:>7d} {r['total_contracts']:>11.1f} {r['total_fees_usd']:>10.2f}")

    # Signal acceptance rates by strategy
    print(f"\n=== Signal accept/reject, last {days} day(s) ===")
    sig_rows = conn.execute(
        """SELECT strategy_guess,
                  SUM(accepted) as accepted,
                  COUNT(*) as total
           FROM signals
           WHERE timestamp >= ?
           GROUP BY strategy_guess
           ORDER BY total DESC""",
        (since,),
    ).fetchall()
    if sig_rows:
        print(f"{'strategy':22s} {'accepted':>9s} {'total':>7s} {'rate':>7s}")
        print("-" * 50)
        for r in sig_rows:
            sg = r["strategy_guess"] or "unknown"
            rate = (r['accepted'] / r['total'] * 100) if r['total'] else 0
            print(f"{sg:22s} {r['accepted']:>9d} {r['total']:>7d} {rate:>6.1f}%")
    else:
        print("(no signals logged yet — signals start logging after main.py picks up this build)")


def list_fills(days: int = 1, strategy: str | None = None) -> None:
    init_db()
    conn = get_conn()
    q = "SELECT * FROM fills WHERE created_time >= ?"
    args: list = [_since(days)]
    if strategy:
        q += " AND strategy_guess = ?"
        args.append(strategy)
    q += " ORDER BY created_time DESC LIMIT 100"
    rows = conn.execute(q, args).fetchall()
    if not rows:
        print("No fills.")
        return
    print(f"{'time':20s} {'ticker':38s} {'side':4s} {'act':4s} {'qty':>6s} {'$':>5s} {'strat':18s}")
    print("-" * 110)
    for r in rows:
        t = (r['created_time'] or '')[:19]
        tkr = r['ticker'][:36]
        price = r['yes_price_cents'] if r['side'] == 'yes' else r['no_price_cents']
        print(f"{t:20s} {tkr:38s} {r['side']:4s} {r['action']:4s} {r['count']:>6.1f} "
              f"{(str(price)+'c') if price else '?':>5s} {r['strategy_guess'] or '?':18s}")


def list_signals(days: int = 1, strategy: str | None = None, accepted_only: bool = False) -> None:
    init_db()
    conn = get_conn()
    q = "SELECT * FROM signals WHERE timestamp >= ?"
    args: list = [_since(days)]
    if strategy:
        q += " AND strategy_guess = ?"
        args.append(strategy)
    if accepted_only:
        q += " AND accepted = 1"
    q += " ORDER BY timestamp DESC LIMIT 200"
    rows = conn.execute(q, args).fetchall()
    if not rows:
        print("No signals.")
        return
    for r in rows:
        ok = "✓" if r['accepted'] else "✗"
        t = (r['timestamp'] or '')[:19]
        tkr = r['ticker'][:36]
        reason = r['reject_reason'] or ''
        print(f"{ok} {t} {tkr:38s} {r['side'] or '?':3s} {r['price_cents'] or '?':>3}c "
              f"x{r['contracts'] or 0:<4} {r['strategy_guess'] or '?':18s} {reason[:50]}")


# ── CLI ─────────────────────────────────────────────────────────────────────
def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(prog="python -m utils.db")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create / migrate the schema")
    sub.add_parser("sync", help="pull latest fills from Kalshi")
    sub.add_parser("schema", help="print table schemas")

    rp = sub.add_parser("report", help="P&L + activity summary")
    rp.add_argument("--days", type=int, default=7)

    fp = sub.add_parser("fills", help="list recent fills")
    fp.add_argument("--days", type=int, default=1)
    fp.add_argument("--strategy")

    sp = sub.add_parser("signals", help="list recent signals (approve_trade decisions)")
    sp.add_argument("--days", type=int, default=1)
    sp.add_argument("--strategy")
    sp.add_argument("--accepted-only", action="store_true")

    args = p.parse_args()

    if args.cmd == "init":
        init_db()
        print(f"Schema created at {DB_PATH}")
    elif args.cmd == "schema":
        init_db()
        rows = get_conn().execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
        for r in rows:
            print(r['sql'] + ';\n')
    elif args.cmd == "sync":
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        from clients.kalshi import KalshiClient
        k = KalshiClient(
            api_key=os.environ["KALSHI_API_KEY"],
            key_file=str(Path(__file__).parent.parent / "kalshi_private.pem"),
            demo=False,
        )
        init_db()
        n = sync_fills_from_kalshi(k)
        print(f"Inserted {n} new fill(s).")
    elif args.cmd == "report":
        report_pnl(days=args.days)
    elif args.cmd == "fills":
        list_fills(days=args.days, strategy=args.strategy)
    elif args.cmd == "signals":
        list_signals(days=args.days, strategy=args.strategy, accepted_only=args.accepted_only)


if __name__ == "__main__":
    _cli()
