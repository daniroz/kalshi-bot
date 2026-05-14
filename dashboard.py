"""
Kalshi bot dashboard.

  .venv/bin/python dashboard.py

Then open http://localhost:5555
"""

import os
import re
import json
import time
import threading
from collections import deque, defaultdict
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template_string
from dotenv import load_dotenv

load_dotenv()

from clients.kalshi import KalshiClient

app = Flask(__name__)

LOG_FILE       = os.path.join(os.path.dirname(__file__), "kalshi_bot.log")
HISTORY_FILE   = os.path.join(os.path.dirname(__file__), "balance_history.json")
SNAPSHOT_EVERY = 60   # seconds between balance snapshots
STARTING_BAL   = float(os.getenv("STARTING_BALANCE", 250))

# ── Kalshi client (shared) ────────────────────────────────────────────────────
_kalshi = KalshiClient(
    api_key  = os.environ["KALSHI_API_KEY"],
    key_file = os.path.join(os.path.dirname(__file__), "kalshi_private.pem"),
    demo     = os.getenv("DEMO_MODE", "true").lower() == "true",
)

# ── Caches ────────────────────────────────────────────────────────────────────
_cache = {"data": None, "ts": 0}
CACHE_TTL = 5    # seconds — protect Kalshi rate limit

_market_cache: dict[str, tuple[float, dict]] = {}
MARKET_TTL = 30

# ── Balance history snapshotter ───────────────────────────────────────────────

def _load_history() -> list[dict]:
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def _save_history(history: list[dict]):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history[-2000:], f)   # keep last ~33h at 1/min
    except Exception:
        pass

def _snapshot_loop():
    while True:
        try:
            bal = _kalshi.get_balance()
            balance = float(bal.get("balance", 0))
            r = _kalshi._get("/portfolio/positions", {"limit": 100})
            events = r.get("event_positions", [])
            realized = sum(float(e.get("realized_pnl_dollars", 0)) for e in events)
            fees     = sum(float(e.get("fees_paid_dollars", 0))    for e in events)
            exposure = sum(float(e.get("event_exposure_dollars", 0)) for e in events)
            equity   = balance + exposure   # equity = cash + open exposure value

            hist = _load_history()
            hist.append({
                "ts":       int(time.time()),
                "balance":  round(balance, 2),
                "equity":   round(equity, 2),
                "realized": round(realized, 2),
                "fees":     round(fees, 2),
            })
            _save_history(hist)
        except Exception as e:
            print(f"[snapshot] {e}")
        time.sleep(SNAPSHOT_EVERY)

threading.Thread(target=_snapshot_loop, daemon=True).start()

# ── Log parsing ───────────────────────────────────────────────────────────────

STRAT_TAGS = ["arb","mm","mis","smart","sports","crypto","weather","intra","ob","mom"]
STRAT_DISPLAY = {
    "arb": "Arbitrage", "mm": "Market Maker", "mis": "Mispricing",
    "smart": "Smart Money", "sports": "Sports", "crypto": "Crypto",
    "weather": "Weather", "intra": "Intraday", "ob": "Orderbook", "mom": "Momentum",
}

def parse_log():
    """Read the bot log and extract trades, errors, cycle count, per-strategy counts."""
    trades = deque(maxlen=100)
    log_lines = deque(maxlen=120)
    cycle = 0
    errors = 0
    strat_counts = defaultdict(lambda: {"enter": 0, "quote": 0, "exit": 0, "error": 0})

    try:
        with open(LOG_FILE, "r", errors="replace") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 500_000))   # last ~500KB
            lines = f.readlines()
    except FileNotFoundError:
        return {"trades": [], "log": [], "cycle": 0, "errors": 0, "strat_counts": {}}

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = re.search(r'Cycle (\d+) done', line)
        if m:
            cycle = max(cycle, int(m.group(1)))

        is_error = "ERROR" in line or "Strategy error" in line
        if is_error:
            errors += 1

        # Classify for log feed
        if is_error:
            cls = "log-error"
        elif any(k in line for k in ["BUY ", "ENTER ", "EXIT ", "Quoted"]):
            cls = "log-trade"
        elif "WARN" in line:
            cls = "log-warn"
        else:
            cls = "log-info"

        short = re.sub(r'HTTP Request:.*?"HTTP.*?"', '', line)
        short = re.sub(r'\s+', ' ', short).strip()
        if short and "HTTP Request" not in short:
            log_lines.append({"text": short[:160], "cls": cls})

        # Identify strategy
        strat = None
        for s in STRAT_TAGS:
            if f"[{s}]" in line:
                strat = s
                break
        if not strat:
            continue

        if is_error:
            strat_counts[strat]["error"] += 1
            continue

        action = None
        if "BUY " in line or "ENTER " in line:
            action = "ENTER"
        elif "EXIT " in line:
            action = "EXIT"
        elif "Quoted" in line:
            action = "QUOTE"
        if not action:
            continue

        strat_counts[strat][action.lower()] += 1

        # Extract details from the line
        tm = re.search(r'(\d{2}:\d{2}:\d{2})', line)
        tk = re.search(r'(KX[\w\-\.]+)', line)
        qty_m = re.search(r'x(\d+)', line)
        price_m = re.search(r'@\s*(\d+)¢', line)
        edge_m = re.search(r'edge[=\s]*([\d.]+)¢', line)

        side = "—"
        if " YES " in line or "side=yes" in line.lower():
            side = "YES"
        elif " NO " in line or "side=no" in line.lower():
            side = "NO"

        trades.append({
            "time":     tm.group(1) if tm else "",
            "strategy": strat,
            "action":   action,
            "ticker":   (tk.group(1) if tk else "—")[:24],
            "side":     side,
            "qty":      qty_m.group(1) if qty_m else "—",
            "price":    f"{price_m.group(1)}¢" if price_m else "—",
            "edge":     f"{edge_m.group(1)}¢" if edge_m else "—",
        })

    return {
        "trades":       list(trades)[-30:],
        "log":          list(log_lines),
        "cycle":        cycle,
        "errors":       errors,
        "strat_counts": dict(strat_counts),
    }

# ── Live data from Kalshi ─────────────────────────────────────────────────────

def get_market_price(ticker: str) -> tuple[float, float]:
    """Return (yes_bid, yes_ask) for a market, cached briefly."""
    now = time.time()
    cached = _market_cache.get(ticker)
    if cached and now - cached[0] < MARKET_TTL:
        m = cached[1]
        return float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
    try:
        r = _kalshi.get_market(ticker)
        m = r.get("market", {})
        _market_cache[ticker] = (now, m)
        return float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
    except Exception:
        return 0.0, 0.0


def fetch_portfolio():
    """Fetch positions, fills, balance, compute PnLs."""
    now = time.time()
    if _cache["data"] and now - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    out = {
        "balance":          0.0,
        "exposure":         0.0,
        "equity":           0.0,
        "realized_pnl":     0.0,
        "unrealized_pnl":   0.0,
        "fees":             0.0,
        "open_positions":   [],
        "open_count":       0,
        "fills":            [],
        "fill_count":       0,
        "today_realized":   0.0,
        "today_fees":       0.0,
        "win_count":        0,
        "loss_count":       0,
        "by_event":         [],
    }

    try:
        bal = _kalshi.get_balance()
        out["balance"] = round(float(bal.get("balance", 0)), 2)
    except Exception as e:
        print(f"[balance] {e}")

    try:
        r = _kalshi._get("/portfolio/positions", {"limit": 200})
        market_positions = r.get("market_positions", [])
        events = r.get("event_positions", [])

        realized_total = sum(float(e.get("realized_pnl_dollars", 0)) for e in events)
        fees_total     = sum(float(e.get("fees_paid_dollars", 0))    for e in events)
        exposure_total = sum(float(e.get("event_exposure_dollars", 0)) for e in events)
        out["realized_pnl"] = round(realized_total, 2)
        out["fees"]         = round(fees_total, 2)
        out["exposure"]     = round(exposure_total, 2)
        out["equity"]       = round(out["balance"] + exposure_total, 2)

        # Open positions (where current position != 0)
        unreal = 0.0
        opens = []
        for p in market_positions:
            qty = int(float(p.get("position", 0) or 0))
            if qty == 0:
                continue
            ticker = p.get("ticker") or p.get("market_ticker")
            mkt_exp = float(p.get("market_exposure_dollars", 0) or 0)
            avg_cost_c = (mkt_exp / abs(qty) * 100) if qty else 0

            bid, ask = get_market_price(ticker)
            # Mark = bid for YES (qty>0), 1-ask for NO (qty<0)
            if qty > 0:
                mark_c = int(bid * 100)
                pnl = (bid - avg_cost_c / 100) * qty
                side = "YES"
            else:
                mark_c = int((1 - ask) * 100)
                pnl = ((1 - ask) - avg_cost_c / 100) * abs(qty)
                side = "NO"
            unreal += pnl
            opens.append({
                "ticker":      ticker,
                "side":        side,
                "qty":         abs(qty),
                "avg_cost_c":  round(avg_cost_c, 1),
                "mark_c":      mark_c,
                "exposure":    round(mkt_exp, 2),
                "pnl":         round(pnl, 2),
                "pnl_pct":     round((pnl / mkt_exp * 100) if mkt_exp else 0, 1),
            })

        opens.sort(key=lambda p: p["pnl"])
        out["open_positions"] = opens
        out["open_count"]     = len(opens)
        out["unrealized_pnl"] = round(unreal, 2)

        # Per-event breakdown for table
        ev_summary = []
        for e in events:
            if float(e.get("total_cost_dollars", 0)) == 0 and float(e.get("event_exposure_dollars", 0)) == 0:
                continue
            ev_summary.append({
                "event":    e.get("event_ticker", ""),
                "cost":     round(float(e.get("total_cost_dollars", 0)), 2),
                "realized": round(float(e.get("realized_pnl_dollars", 0)), 2),
                "fees":     round(float(e.get("fees_paid_dollars", 0)), 2),
                "exposure": round(float(e.get("event_exposure_dollars", 0)), 2),
            })
        ev_summary.sort(key=lambda x: x["realized"])
        out["by_event"] = ev_summary

    except Exception as e:
        print(f"[positions] {e}")

    # Fills
    try:
        r = _kalshi._get("/portfolio/fills", {"limit": 50})
        fills = r.get("fills", [])
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_realized = 0.0
        today_fees = 0.0
        wins, losses = 0, 0
        recent = []
        for f in fills:
            ts_str = f.get("created_time", "")
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                t_disp = dt.astimezone().strftime("%H:%M:%S")
                is_today = dt.strftime("%Y-%m-%d") == today_str
            except Exception:
                t_disp = ""
                is_today = False
            fee = float(f.get("fee_cost", 0) or 0)
            side = f.get("side", "")
            action = f.get("action", "")
            count = int(float(f.get("count_fp", 0)))
            yes_p = float(f.get("yes_price_dollars", 0) or 0)
            no_p  = float(f.get("no_price_dollars", 0) or 0)
            price = yes_p if side == "yes" else no_p
            if is_today:
                today_fees += fee
                if action == "sell":   # sells are exits → count as realized
                    today_realized -= fee
            recent.append({
                "time":   t_disp,
                "ticker": f.get("market_ticker", "")[:24],
                "side":   side.upper(),
                "action": action.upper(),
                "qty":    count,
                "price":  f"{int(price*100)}¢",
                "fee":    f"${fee:.3f}",
            })
        out["fills"]          = recent[:30]
        out["fill_count"]     = len(fills)
        out["today_fees"]     = round(today_fees, 2)
        out["today_realized"] = round(today_realized, 2)
    except Exception as e:
        print(f"[fills] {e}")

    # Win/loss from events (closed events with non-zero realized)
    try:
        for e in out["by_event"]:
            if e["exposure"] == 0:    # event fully closed
                if e["realized"] > 0: out["win_count"] = out.get("win_count", 0) + 1
                elif e["realized"] < 0: out["loss_count"] = out.get("loss_count", 0) + 1
    except Exception:
        pass

    _cache["data"] = out
    _cache["ts"] = now
    return out


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.route("/api/snapshot")
def api_snapshot():
    log_data = parse_log()
    portfolio = fetch_portfolio()

    total_pnl = round(portfolio["realized_pnl"] + portfolio["unrealized_pnl"], 2)
    total_pnl_pct = round((total_pnl / STARTING_BAL * 100), 2) if STARTING_BAL else 0

    strategies_env = {
        "arb":     ("Arbitrage",    "STRATEGY_ARBITRAGE"),
        "mm":      ("Market Maker", "STRATEGY_MARKET_MAKER"),
        "mis":     ("Mispricing",   "STRATEGY_MISPRICING"),
        "smart":   ("Smart Money",  "STRATEGY_SMART_MONEY"),
        "weather": ("Weather",      "STRATEGY_WEATHER"),
        "ob":      ("Orderbook",    "STRATEGY_ORDERBOOK"),
        "mom":     ("Momentum",     "STRATEGY_MOMENTUM"),
        "intra":   ("Intraday",     "STRATEGY_INTRADAY"),
        "sports":  ("Sports",       "STRATEGY_SPORTS"),
        "crypto":  ("Crypto",       "STRATEGY_CRYPTO"),
    }
    strategies = []
    for tag, (name, env) in strategies_env.items():
        c = log_data["strat_counts"].get(tag, {})
        strategies.append({
            "tag":     tag,
            "name":    name,
            "on":      os.getenv(env, "true").lower() == "true",
            "enters":  c.get("enter", 0),
            "quotes":  c.get("quote", 0),
            "exits":   c.get("exit", 0),
            "errors":  c.get("error", 0),
        })

    history = _load_history()

    return jsonify({
        "balance":         portfolio["balance"],
        "equity":          portfolio["equity"],
        "exposure":        portfolio["exposure"],
        "realized_pnl":    portfolio["realized_pnl"],
        "unrealized_pnl":  portfolio["unrealized_pnl"],
        "total_pnl":       total_pnl,
        "total_pnl_pct":   total_pnl_pct,
        "fees":            portfolio["fees"],
        "today_realized":  portfolio["today_realized"],
        "today_fees":      portfolio["today_fees"],
        "open_count":      portfolio["open_count"],
        "open_positions":  portfolio["open_positions"],
        "fills":           portfolio["fills"],
        "fill_count":      portfolio["fill_count"],
        "win_count":       portfolio["win_count"],
        "loss_count":      portfolio["loss_count"],
        "by_event":        portfolio["by_event"],
        "cycle":           log_data["cycle"],
        "errors":          log_data["errors"],
        "strategies":      strategies,
        "trades":          log_data["trades"],
        "log":             log_data["log"],
        "history":         history[-720:],   # last 12h at 1/min
        "starting_bal":    STARTING_BAL,
        "now":             datetime.now().strftime("%H:%M:%S"),
    })


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Kalshi Bot</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{background:#0a0a0a;color:#e0e0e0;font-family:'SF Mono','JetBrains Mono','Fira Code',monospace;font-size:13px}
  a{color:inherit}
  h2{font-size:11px;text-transform:uppercase;letter-spacing:2.5px;color:#777;margin-bottom:10px;font-weight:600}
  .green{color:#00e676}.red{color:#ff5252}.yellow{color:#ffd740}.blue{color:#40c4ff}.gray{color:#666}
  .muted{color:#555}
  .header{display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid #1a1a1a;background:#0d0d0d}
  .header h1{font-size:14px;letter-spacing:4px;text-transform:uppercase;font-weight:700}
  .dot{width:8px;height:8px;border-radius:50%;background:#00e676;display:inline-block;margin-right:8px;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .stale .dot{background:#ff5252;animation:none}
  .pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;background:#1a1a1a;border:1px solid #2a2a2a;color:#999}
  .grid-kpi{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;padding:14px 20px 6px}
  .card{background:#121212;border:1px solid #1f1f1f;border-radius:6px;padding:14px}
  .card .label{color:#666;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;font-weight:600}
  .card .value{font-size:24px;font-weight:700;margin-top:6px;font-variant-numeric:tabular-nums}
  .card .sub{font-size:11px;color:#666;margin-top:3px;font-variant-numeric:tabular-nums}
  .section{padding:6px 20px 20px}
  .two-col{display:grid;grid-template-columns:2fr 1fr;gap:14px}
  .three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  th{text-align:left;color:#555;font-size:10px;text-transform:uppercase;letter-spacing:1px;padding:7px 8px;border-bottom:1px solid #1a1a1a;font-weight:600}
  td{padding:7px 8px;border-bottom:1px solid #161616;font-size:12px}
  tr:hover td{background:#161616}
  .tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700;letter-spacing:.5px}
  .tag-arb{background:#0e2a14;color:#00e676}
  .tag-mm{background:#0e1f2a;color:#40c4ff}
  .tag-mis{background:#2a2410;color:#ffd740}
  .tag-smart{background:#251025;color:#ea80fc}
  .tag-sports{background:#10282a;color:#80deea}
  .tag-crypto{background:#2a1410;color:#ff6e40}
  .tag-weather{background:#101428;color:#82b1ff}
  .tag-intra{background:#222;color:#e0e0e0}
  .tag-ob{background:#142a18;color:#b9f6ca}
  .tag-mom{background:#2a1612;color:#ff8a65}
  .log-box{background:#0c0c0c;border:1px solid #1a1a1a;border-radius:6px;padding:10px;height:340px;overflow-y:auto;font-size:11px;line-height:1.7}
  .log-entry{white-space:pre-wrap;word-break:break-word;border-left:2px solid transparent;padding-left:6px;margin:1px 0}
  .log-info{color:#777}
  .log-trade{color:#00e676;border-color:#00e676}
  .log-error{color:#ff5252;border-color:#ff5252}
  .log-warn{color:#ffd740;border-color:#ffd740}
  .panel{background:#121212;border:1px solid #1f1f1f;border-radius:6px;padding:14px}
  .strategy-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
  .strat-card{background:#0e0e0e;border:1px solid #1f1f1f;border-radius:6px;padding:10px}
  .strat-card.off{opacity:.35}
  .strat-card .name{font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#bbb}
  .strat-card .stats{display:flex;gap:10px;margin-top:8px;font-size:11px;font-variant-numeric:tabular-nums}
  .strat-card .stat{display:flex;flex-direction:column}
  .strat-card .stat .v{font-weight:700;font-size:13px}
  .strat-card .stat .l{color:#555;font-size:9px;text-transform:uppercase;letter-spacing:1px}
  .chart-wrap{height:200px;position:relative}
  .empty{color:#444;text-align:center;padding:24px;font-size:12px}
  .ago{color:#444;font-size:10px}
  .nav{display:flex;gap:6px}
  .nav .pill{cursor:default}
</style>
</head>
<body class="" id="body">

<div class="header" id="header">
  <h1><span class="dot"></span>Kalshi Bot</h1>
  <div class="nav">
    <span class="pill" id="cycle-pill">Cycle —</span>
    <span class="pill" id="errors-pill">0 errors</span>
    <span class="pill" id="time-pill">—</span>
  </div>
</div>

<div class="grid-kpi">
  <div class="card"><div class="label">Cash Balance</div><div class="value" id="balance">—</div><div class="sub" id="balance-sub">started at $— </div></div>
  <div class="card"><div class="label">Equity (cash+open)</div><div class="value" id="equity">—</div><div class="sub" id="equity-sub">— exposure</div></div>
  <div class="card"><div class="label">Total P&L</div><div class="value" id="total-pnl">—</div><div class="sub" id="total-pnl-sub">— %</div></div>
  <div class="card"><div class="label">Realized P&L</div><div class="value" id="realized">—</div><div class="sub" id="realized-sub">fees $—</div></div>
  <div class="card"><div class="label">Unrealized P&L</div><div class="value" id="unrealized">—</div><div class="sub" id="unrealized-sub">— open</div></div>
  <div class="card"><div class="label">Win / Loss</div><div class="value" id="winloss">—</div><div class="sub" id="winloss-sub">closed events</div></div>
</div>

<div class="section">
  <div class="panel">
    <h2>Equity Curve (last 12h)</h2>
    <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
  </div>
</div>

<div class="section">
  <h2>Strategies</h2>
  <div class="strategy-grid" id="strategies"></div>
</div>

<div class="section">
  <h2>Open Positions</h2>
  <div class="panel" style="padding:0">
    <table>
      <thead><tr>
        <th>Ticker</th><th>Side</th><th>Qty</th><th>Avg Cost</th><th>Mark</th><th>Exposure</th><th>Unrealized P&L</th><th>%</th>
      </tr></thead>
      <tbody id="positions-body"></tbody>
    </table>
  </div>
</div>

<div class="section">
  <div class="two-col">
    <div class="panel" style="padding:0">
      <div style="padding:12px 14px 0"><h2>Recent Fills (real exchange data)</h2></div>
      <table>
        <thead><tr>
          <th>Time</th><th>Ticker</th><th>Action</th><th>Side</th><th>Qty</th><th>Price</th><th>Fee</th>
        </tr></thead>
        <tbody id="fills-body"></tbody>
      </table>
    </div>
    <div class="panel">
      <h2>Live Log</h2>
      <div class="log-box" id="log-box"></div>
    </div>
  </div>
</div>

<div class="section">
  <div class="two-col">
    <div class="panel" style="padding:0">
      <div style="padding:12px 14px 0"><h2>By Event (closed + open)</h2></div>
      <table>
        <thead><tr>
          <th>Event</th><th>Cost</th><th>Realized</th><th>Fees</th><th>Open Exp.</th>
        </tr></thead>
        <tbody id="events-body"></tbody>
      </table>
    </div>
    <div class="panel" style="padding:0">
      <div style="padding:12px 14px 0"><h2>Recent Bot Activity (from log)</h2></div>
      <table>
        <thead><tr>
          <th>Time</th><th>Strat</th><th>Action</th><th>Ticker</th><th>Side</th><th>Qty</th><th>Price</th>
        </tr></thead>
        <tbody id="trades-body"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
let chart = null;
let lastCycle = null;
let lastUpdate = 0;

function fmt$(n) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  const sign = n >= 0 ? "" : "-";
  return sign + "$" + Math.abs(n).toFixed(2);
}
function fmtSign$(n) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  const sign = n >= 0 ? "+" : "";
  return sign + "$" + n.toFixed(2);
}
function colorClass(n){ return n > 0 ? "green" : (n < 0 ? "red" : "gray"); }
function el(id){ return document.getElementById(id); }

function updateKPI(d) {
  el("balance").textContent = fmt$(d.balance);
  el("balance-sub").textContent = "started at $" + d.starting_bal.toFixed(2);

  el("equity").textContent = fmt$(d.equity);
  el("equity-sub").textContent = "$" + d.exposure.toFixed(2) + " open exposure";

  const tp = el("total-pnl");
  tp.textContent = fmtSign$(d.total_pnl);
  tp.className = "value " + colorClass(d.total_pnl);
  el("total-pnl-sub").textContent = (d.total_pnl_pct >= 0 ? "+" : "") + d.total_pnl_pct.toFixed(2) + "% from start";

  const rp = el("realized");
  rp.textContent = fmtSign$(d.realized_pnl);
  rp.className = "value " + colorClass(d.realized_pnl);
  el("realized-sub").textContent = "fees $" + d.fees.toFixed(2);

  const up = el("unrealized");
  up.textContent = fmtSign$(d.unrealized_pnl);
  up.className = "value " + colorClass(d.unrealized_pnl);
  el("unrealized-sub").textContent = d.open_count + " open position" + (d.open_count===1?"":"s");

  el("winloss").textContent = d.win_count + " / " + d.loss_count;
  el("winloss").className = "value " + (d.win_count > d.loss_count ? "green" : (d.win_count < d.loss_count ? "red" : "gray"));
  const total = d.win_count + d.loss_count;
  el("winloss-sub").textContent = total ? (Math.round(d.win_count/total*100) + "% win rate") : "no closed events";

  el("cycle-pill").textContent = "Cycle " + (d.cycle || "—");
  el("errors-pill").textContent = d.errors + " errors";
  el("errors-pill").className = "pill" + (d.errors > 0 ? " red" : "");
  el("time-pill").textContent = d.now;

  if (lastCycle !== null && d.cycle === lastCycle && (Date.now()/1000 - lastUpdate) > 60) {
    document.body.classList.add("stale");
  } else {
    document.body.classList.remove("stale");
  }
  if (d.cycle !== lastCycle) { lastCycle = d.cycle; lastUpdate = Date.now()/1000; }
}

function updatePositions(positions){
  const tb = el("positions-body");
  if (!positions.length) { tb.innerHTML = '<tr><td colspan="8" class="empty">No open positions</td></tr>'; return; }
  tb.innerHTML = positions.map(p => `
    <tr>
      <td>${p.ticker}</td>
      <td><span class="tag tag-${p.side==='YES'?'arb':'crypto'}">${p.side}</span></td>
      <td>${p.qty}</td>
      <td>${p.avg_cost_c}¢</td>
      <td>${p.mark_c}¢</td>
      <td>$${p.exposure.toFixed(2)}</td>
      <td class="${colorClass(p.pnl)}">${fmtSign$(p.pnl)}</td>
      <td class="${colorClass(p.pnl_pct)}">${p.pnl_pct>=0?'+':''}${p.pnl_pct}%</td>
    </tr>
  `).join("");
}

function updateFills(fills){
  const tb = el("fills-body");
  if (!fills.length) { tb.innerHTML = '<tr><td colspan="7" class="empty">No fills yet</td></tr>'; return; }
  tb.innerHTML = fills.map(f => `
    <tr>
      <td class="muted">${f.time}</td>
      <td>${f.ticker}</td>
      <td class="${f.action==='BUY'?'green':'red'}">${f.action}</td>
      <td>${f.side}</td>
      <td>${f.qty}</td>
      <td>${f.price}</td>
      <td class="muted">${f.fee}</td>
    </tr>
  `).join("");
}

function updateEvents(events){
  const tb = el("events-body");
  if (!events.length) { tb.innerHTML = '<tr><td colspan="5" class="empty">No event activity</td></tr>'; return; }
  tb.innerHTML = events.map(e => `
    <tr>
      <td>${e.event}</td>
      <td>$${e.cost.toFixed(2)}</td>
      <td class="${colorClass(e.realized)}">${fmtSign$(e.realized)}</td>
      <td class="muted">$${e.fees.toFixed(2)}</td>
      <td>$${e.exposure.toFixed(2)}</td>
    </tr>
  `).join("");
}

function updateTrades(trades){
  const tb = el("trades-body");
  if (!trades.length) { tb.innerHTML = '<tr><td colspan="7" class="empty">No bot activity yet</td></tr>'; return; }
  tb.innerHTML = trades.slice().reverse().map(t => `
    <tr>
      <td class="muted">${t.time}</td>
      <td><span class="tag tag-${t.strategy}">${t.strategy}</span></td>
      <td class="${t.action==='ENTER'?'green':(t.action==='EXIT'?'red':'blue')}">${t.action}</td>
      <td>${t.ticker}</td>
      <td>${t.side}</td>
      <td>${t.qty}</td>
      <td>${t.price}</td>
    </tr>
  `).join("");
}

function updateStrategies(strats){
  el("strategies").innerHTML = strats.map(s => `
    <div class="strat-card ${s.on?'':'off'}">
      <div class="name">${s.name}</div>
      <div class="stats">
        <div class="stat"><span class="v green">${s.enters||0}</span><span class="l">enters</span></div>
        <div class="stat"><span class="v blue">${s.quotes||0}</span><span class="l">quotes</span></div>
        <div class="stat"><span class="v">${s.exits||0}</span><span class="l">exits</span></div>
        <div class="stat"><span class="v ${s.errors?'red':''}">${s.errors||0}</span><span class="l">errs</span></div>
      </div>
    </div>
  `).join("");
}

function updateLog(lines){
  el("log-box").innerHTML = lines.map(l =>
    `<div class="log-entry ${l.cls}">${l.text.replace(/</g,"&lt;")}</div>`
  ).join("");
  el("log-box").scrollTop = el("log-box").scrollHeight;
}

function updateChart(history){
  if (!history.length) return;
  const labels = history.map(h => {
    const d = new Date(h.ts * 1000);
    return d.getHours().toString().padStart(2,"0") + ":" + d.getMinutes().toString().padStart(2,"0");
  });
  const balData = history.map(h => h.balance);
  const eqData  = history.map(h => h.equity);
  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = eqData;
    chart.data.datasets[1].data = balData;
    chart.update("none");
  } else {
    const ctx = document.getElementById("equityChart").getContext("2d");
    chart = new Chart(ctx, {
      type:"line",
      data:{labels:labels, datasets:[
        {label:"Equity",  data:eqData,  borderColor:"#00e676", backgroundColor:"rgba(0,230,118,0.08)", fill:true, tension:.25, pointRadius:0, borderWidth:2},
        {label:"Cash",    data:balData, borderColor:"#40c4ff", backgroundColor:"transparent",         fill:false, tension:.25, pointRadius:0, borderWidth:1.5, borderDash:[4,4]},
      ]},
      options:{
        responsive:true, maintainAspectRatio:false, animation:false,
        plugins:{legend:{labels:{color:"#888", font:{size:11}}}},
        scales:{
          x:{grid:{color:"#1a1a1a"}, ticks:{color:"#555", maxTicksLimit:10, font:{size:10}}},
          y:{grid:{color:"#1a1a1a"}, ticks:{color:"#555", font:{size:10}, callback:v=>"$"+v.toFixed(0)}},
        },
      },
    });
  }
}

async function refresh(){
  try {
    const r = await fetch("/api/snapshot");
    const d = await r.json();
    updateKPI(d);
    updatePositions(d.open_positions);
    updateFills(d.fills);
    updateEvents(d.by_event);
    updateTrades(d.trades);
    updateStrategies(d.strategies);
    updateLog(d.log);
    updateChart(d.history);
  } catch(e) {
    console.error(e);
  }
}

refresh();
setInterval(refresh, 5000);
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    print("Dashboard: http://localhost:5555")
    app.run(host="0.0.0.0", port=5555, debug=False, use_reloader=False)
