"""Kalshi bot dashboard — http://localhost:5555"""

import os, re, json, time, threading
from collections import defaultdict
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template_string
from dotenv import load_dotenv

load_dotenv()
from clients.kalshi import KalshiClient

app     = Flask(__name__)
LOG_FILE     = os.path.join(os.path.dirname(__file__), "bot.log")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "balance_history.json")
STARTING_BAL = float(os.getenv("STARTING_BALANCE", 285))

_kalshi = KalshiClient(
    api_key  = os.environ["KALSHI_API_KEY"],
    key_file = os.path.join(os.path.dirname(__file__), "kalshi_private.pem"),
    demo     = os.getenv("DEMO_MODE", "true").lower() == "true",
)
_cache = {"data": None, "ts": 0}

# ── History ───────────────────────────────────────────────────────────────────

def _load_history():
    try:
        with open(HISTORY_FILE) as f: return json.load(f)
    except: return []

def _save_history(h):
    try:
        with open(HISTORY_FILE, "w") as f: json.dump(h[-43200:], f)
    except: pass

def _snapshot_loop():
    while True:
        try:
            bal     = _kalshi.get_balance()
            cash    = float(bal.get("balance", 0))
            pos_val = float(bal.get("portfolio_value", 0)) / 100
            equity  = round(cash + pos_val, 2)
            h = _load_history()
            # Sanity check: skip readings where equity jumps/drops >50% in one snapshot.
            # This filters out Kalshi API double-counting during market settlement windows.
            if h:
                prev = h[-1]["equity"]
                if prev > 0 and (equity / prev > 1.5 or equity / prev < 0.5):
                    print(f"[snap] Skipping outlier equity=${equity:.2f} (prev=${prev:.2f}) — likely settlement artifact")
                    time.sleep(30)
                    continue
            h.append({"ts": int(time.time()), "equity": equity, "cash": round(cash, 2)})
            _save_history(h)
        except Exception as e:
            print(f"[snap] {e}")
        time.sleep(30)

threading.Thread(target=_snapshot_loop, daemon=True).start()

# ── Portfolio ─────────────────────────────────────────────────────────────────

def fetch_portfolio():
    now = time.time()
    if _cache["data"] and now - _cache["ts"] < 5:
        return _cache["data"]

    out = {
        "equity": 0.0, "cash": 0.0, "pos_val": 0.0,
        "realized_pnl": 0.0, "fees": 0.0,
        "open_positions": [], "fills": [],
    }

    try:
        bal = _kalshi.get_balance()
        out["cash"]    = round(float(bal.get("balance", 0)), 2)
        out["pos_val"] = round(float(bal.get("portfolio_value", 0)) / 100, 2)
        out["equity"]  = round(out["cash"] + out["pos_val"], 2)
    except Exception as e:
        print(f"[bal] {e}")

    try:
        r      = _kalshi._get("/portfolio/positions", {"limit": 200})
        events = r.get("event_positions", [])
        out["realized_pnl"] = round(sum(float(e.get("realized_pnl_dollars", 0) or 0) for e in events), 2)
        out["fees"]         = round(sum(float(e.get("fees_paid_dollars", 0)    or 0) for e in events), 2)

        mkt_pos = r.get("market_positions", [])
        opens   = []
        _mcache = {}
        for p in mkt_pos:
            fp = float(p.get("position_fp") or 0)
            if not fp: continue
            ticker = p.get("ticker") or p.get("market_ticker", "")
            try:
                m  = _kalshi.get_market(ticker).get("market", {})
                _mcache[ticker] = m
                bid = float(m.get("yes_bid_dollars") or 0)
                ask = float(m.get("yes_ask_dollars") or 0)
            except: bid = ask = 0

            if fp > 0:
                side   = "YES"
                mark_c = int(bid * 100)
                val    = round(bid * fp, 2)
            else:
                side   = "NO"
                mark_c = int((1 - ask) * 100) if ask else 0
                val    = round((1 - ask) * abs(fp), 2) if ask else 0

            opens.append({
                "ticker": ticker, "side": side,
                "qty": round(abs(fp), 2), "mark_c": mark_c, "val": val,
            })
        opens.sort(key=lambda x: x["val"], reverse=True)
        out["open_positions"] = opens
    except Exception as e:
        print(f"[pos] {e}")

    try:
        # Fetch all fills via pagination so we have full cost-basis history
        fills = []
        cursor = None
        for _ in range(20):   # max 20 pages = 2000 fills
            params = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            r2 = _kalshi._get("/portfolio/fills", params)
            batch = r2.get("fills", [])
            fills.extend(batch)
            cursor = r2.get("cursor")
            if not cursor or not batch:
                break
        today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # P&L calculation — handles both same-side and cross-side (market maker) positions.
        #
        # Kalshi positions can be opened/closed across sides:
        #   BUY YES  → long YES  (close with SELL YES)
        #   SELL YES → long NO   (close with SELL NO or BUY YES)
        #   BUY NO   → long NO   (close with SELL NO)
        #   SELL NO  → long YES  (close with SELL YES or BUY NO)
        #
        # We track a unified NO-equivalent cost basis per ticker.
        #   Opening long YES at p  → YES basis = p
        #   Opening long NO  at p  → NO basis = p
        #   SELL YES at p          → opens NO, NO equivalent cost = 1 - p
        #   SELL NO  at p          → opens YES, YES equivalent cost = 1 - p

        # avg_cost[ticker] = {"yes": price, "no": price}  (None if unknown)
        avg_cost: dict = {}   # ticker -> {"yes": float|None, "no": float|None}
        qty_held: dict = {}   # ticker -> {"yes": float, "no": float}

        def get_basis(ticker, direction):
            return avg_cost.get(ticker, {}).get(direction)

        def add_position(ticker, direction, price, count):
            if ticker not in avg_cost:
                avg_cost[ticker] = {"yes": None, "no": None}
                qty_held[ticker] = {"yes": 0.0,  "no": 0.0}
            held = qty_held[ticker][direction]
            old  = avg_cost[ticker][direction] or price
            new_held = held + count
            avg_cost[ticker][direction] = (old * held + price * count) / new_held if new_held else price
            qty_held[ticker][direction] = new_held

        def close_position(ticker, direction, price, count):
            basis = get_basis(ticker, direction)
            if basis is None:
                return None   # opened before our 200-fill window
            pnl = round((price - basis) * count, 2)
            held = qty_held.get(ticker, {}).get(direction, count)
            if ticker in qty_held:
                qty_held[ticker][direction] = max(0, held - count)
            return pnl

        parsed = []
        for f in reversed(fills):
            side   = f.get("side", "")
            action = f.get("action", "")
            count  = float(f.get("count_fp", 0) or 0)
            yes_p  = float(f.get("yes_price_dollars", 0) or 0)
            no_p   = float(f.get("no_price_dollars",  0) or 0)
            ticker = f.get("market_ticker", "")
            pnl    = None

            if action == "buy" and side == "yes":
                # Opens long YES position
                add_position(ticker, "yes", yes_p, count)
            elif action == "buy" and side == "no":
                # Opens long NO position
                add_position(ticker, "no", no_p, count)
            elif action == "sell" and side == "yes":
                # Could be: closing a long YES, OR opening a long NO (market maker ask)
                yes_basis = get_basis(ticker, "yes")
                if yes_basis is not None and qty_held.get(ticker, {}).get("yes", 0) > 0:
                    # Closing a long YES position
                    pnl = close_position(ticker, "yes", yes_p, count)
                else:
                    # Opening a long NO position (market maker sold YES short)
                    # NO equivalent cost = 1 - yes_price received
                    add_position(ticker, "no", 1 - yes_p, count)
            elif action == "sell" and side == "no":
                # Could be: closing a long NO, OR opening a long YES
                no_basis = get_basis(ticker, "no")
                if no_basis is not None and qty_held.get(ticker, {}).get("no", 0) > 0:
                    # Closing a long NO position
                    pnl = close_position(ticker, "no", no_p, count)
                else:
                    # Opening a long YES position (sold NO short)
                    add_position(ticker, "yes", 1 - no_p, count)

            price = yes_p if side == "yes" else no_p
            parsed.append((f, price, pnl))

        recent = []
        for f, price, pnl in reversed(parsed):
            ts = f.get("created_time", "")
            try:
                dt     = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                t_disp = dt.astimezone().strftime("%H:%M")
                is_today = dt.strftime("%Y-%m-%d") == today
            except: t_disp = ""; is_today = False
            side   = f.get("side", "")
            action = f.get("action", "")
            count  = float(f.get("count_fp", 0) or 0)
            fee    = float(f.get("fee_cost", 0) or 0)
            recent.append({
                "time": t_disp, "today": is_today,
                "ticker": f.get("market_ticker", "")[:30],
                "side": side.upper(), "action": action.upper(),
                "qty": round(count, 1), "price_c": int(price * 100),
                "fee": fee, "pnl": pnl,
            })
        out["fills"] = recent[:50]

        # Attach live unrealized P&L to each open position using the cost basis we just built
        for pos in out.get("open_positions", []):
            tkr  = pos["ticker"]
            side = pos["side"]          # "YES" or "NO"
            qty  = pos["qty"]
            mark = pos["mark_c"] / 100  # current mark in dollars
            try:
                direction = "yes" if side == "YES" else "no"
                basis = avg_cost.get(tkr, {}).get(direction)
                if basis is None:
                    # Try the other direction (market maker cross-side opens)
                    other = "no" if direction == "yes" else "yes"
                    basis = avg_cost.get(tkr, {}).get(other)
                pos["unrealized_pnl"] = round((mark - basis) * qty, 2) if basis is not None else None
                pos["entry_c"]        = int(round(basis * 100))       if basis is not None else None
                pos["cost_basis"]     = round(basis * qty, 2)         if basis is not None else None
            except Exception:
                pos["unrealized_pnl"] = None
                pos["entry_c"] = None
                pos["cost_basis"] = None

    except Exception as e:
        print(f"[fills] {e}")

    _cache["data"] = out
    _cache["ts"]   = now
    return out

# ── Log parser ────────────────────────────────────────────────────────────────

STRAT_TAGS = ["arb","mm","mis","smart","sports","crypto","weather","intra","ob","mom",
              "news","cal","settle","settle-crypto","settle-stocks","settle-sports",
              "settle-comm","settle-fx"]
STRAT_NAMES = {"arb":"Arb","mm":"MM","mis":"Mispricing","smart":"Smart$",
               "sports":"Sports","crypto":"Crypto","weather":"Weather","intra":"Intraday",
               "ob":"Orderbook","mom":"Momentum","news":"News","cal":"Calendar",
               "settle":"Settle-Wx",
               "settle-crypto":"Settle-Crypto",
               "settle-stocks":"Settle-Stocks",
               "settle-sports":"Settle-Sports",
               "settle-comm":"Settle-Comm",
               "settle-fx":"Settle-FX"}
# Map dashboard short tags → the env var names main.py actually reads
STRAT_ENV = {"arb":"ARBITRAGE","mm":"MARKET_MAKER","mis":"MISPRICING","smart":"SMART_MONEY",
             "sports":"SPORTS","crypto":"CRYPTO","weather":"WEATHER","intra":"INTRADAY",
             "ob":"ORDERBOOK","mom":"MOMENTUM","news":"NEWS","cal":"CALENDAR",
             "settle":"SETTLEMENT",
             "settle-crypto":"SETTLEMENT_CRYPTO",
             "settle-stocks":"SETTLEMENT_STOCKS",
             "settle-sports":"SETTLEMENT_SPORTS",
             "settle-comm":"SETTLEMENT_COMM",
             "settle-fx":"SETTLEMENT_FX"}

def parse_log():
    cycle = 0
    errors = 0
    counts = defaultdict(lambda: {"e": 0, "x": 0, "q": 0, "err": 0})
    try:
        with open(LOG_FILE, "r", errors="replace") as f:
            f.seek(0, 2); size = f.tell()
            f.seek(max(0, size - 400_000))
            lines = f.readlines()
    except: return {"cycle": 0, "errors": 0, "counts": {}}
    for raw in lines:
        line = raw.strip()
        if not line or "HTTP Request" in line: continue
        m = re.search(r'Cycle (\d+) done', line)
        if m: cycle = max(cycle, int(m.group(1)))
        if "ERROR" in line or "Strategy error" in line: errors += 1
        strat = next((s for s in STRAT_TAGS if f"[{s}]" in line), None)
        if not strat: continue
        if "ERROR" in line: counts[strat]["err"] += 1
        elif "ENTER" in line or "BUY " in line or "ARB " in line: counts[strat]["e"] += 1
        elif "EXIT" in line: counts[strat]["x"] += 1
        elif "Quoted" in line: counts[strat]["q"] += 1
    return {"cycle": cycle, "errors": errors, "counts": dict(counts)}

# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/snapshot")
def api_snapshot():
    port  = fetch_portfolio()
    log   = parse_log()
    hist  = _load_history()

    # P&L calculations
    alltime_pnl  = round(port["equity"] - STARTING_BAL, 2)
    alltime_pct  = round(alltime_pnl / STARTING_BAL * 100, 1) if STARTING_BAL else 0

    day_ago_ts = time.time() - 86400
    day_snap   = next((h for h in reversed(hist) if h["ts"] <= day_ago_ts), None)
    today_pnl  = round(port["equity"] - (day_snap["equity"] if day_snap else STARTING_BAL), 2)
    today_pct  = round(today_pnl / (day_snap["equity"] if day_snap else STARTING_BAL) * 100, 1)

    strats = []
    for tag in STRAT_TAGS:
        c = log["counts"].get(tag, {})
        strats.append({
            "tag": tag, "name": STRAT_NAMES[tag],
            "on":  os.getenv(f"STRATEGY_{STRAT_ENV.get(tag, tag.upper())}", "true").lower() == "true",
            "e": c.get("e",0), "x": c.get("x",0), "q": c.get("q",0), "err": c.get("err",0),
        })

    return jsonify({
        "equity": port["equity"], "cash": port["cash"], "pos_val": port["pos_val"],
        "alltime_pnl": alltime_pnl, "alltime_pct": alltime_pct,
        "today_pnl": today_pnl, "today_pct": today_pct,
        "realized_pnl": port["realized_pnl"], "fees": port["fees"],
        "open_positions": port["open_positions"],
        "open_count": len(port["open_positions"]),
        "fills": port["fills"],
        "cycle": log["cycle"], "errors": log["errors"],
        "strategies": strats,
        "history": hist,
        "starting_bal": STARTING_BAL,
        "now": datetime.now().strftime("%H:%M:%S"),
    })

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Kalshi Bot</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0d;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','Segoe UI',sans-serif;font-size:14px}
.mono{font-family:'SF Mono','Fira Code',monospace}

/* Header */
.hdr{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;border-bottom:1px solid #1e1e1e}
.hdr-left{display:flex;align-items:center;gap:16px}
.hdr h1{font-size:15px;font-weight:700;letter-spacing:2px;text-transform:uppercase}
.dot{width:8px;height:8px;border-radius:50%;background:#00e676;display:inline-block;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.badge{padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;background:#1a1a1a;border:1px solid #2a2a2a;color:#777}
.badge.err{color:#ff5252;border-color:#3a1515;background:#1a0a0a}

/* KPI row */
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;padding:16px 24px 8px}
.kpi{background:#111;border:1px solid #1e1e1e;border-radius:8px;padding:14px 16px}
.kpi .lbl{font-size:11px;color:#555;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px}
.kpi .val{font-size:26px;font-weight:700;letter-spacing:-1px;font-variant-numeric:tabular-nums}
.kpi .sub{font-size:11px;color:#555;margin-top:3px}
.green{color:#00e676}.red{color:#ff5252}.white{color:#fff}.muted{color:#555}

/* Chart */
.chart-section{padding:8px 24px 16px}
.chart-card{background:#111;border:1px solid #1e1e1e;border-radius:8px;padding:16px;position:relative}
.chart-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
.chart-title{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#666}
.timebtns{display:flex;gap:4px}
.tbtn{padding:4px 12px;border-radius:6px;border:1px solid #2a2a2a;background:transparent;color:#555;font-size:12px;font-weight:600;cursor:pointer;transition:all .15s}
.tbtn:hover{color:#aaa;border-color:#444}
.tbtn.active{background:#1e1e1e;color:#e0e0e0;border-color:#3a3a3a}
.chart-wrap{height:220px;position:relative;cursor:crosshair}
#chart-tooltip{
  position:fixed;pointer-events:none;display:none;
  background:rgba(18,18,18,0.95);border:1px solid #2a2a2a;border-radius:8px;
  padding:8px 12px;font-family:'SF Mono','Fira Code',monospace;
  box-shadow:0 4px 24px rgba(0,0,0,.6);z-index:9999;white-space:nowrap;
  backdrop-filter:blur(8px);
}
#chart-tooltip .tt-time{font-size:11px;color:#555;margin-bottom:4px}
#chart-tooltip .tt-val{font-size:18px;font-weight:700;color:#00e676;letter-spacing:-0.5px}
#chart-tooltip .tt-chg{font-size:11px;margin-top:2px}

/* Two columns */
.cols{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 24px 16px}
.col-card{background:#111;border:1px solid #1e1e1e;border-radius:8px;overflow:hidden}
.col-hdr{padding:12px 16px 0;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#555;margin-bottom:8px;display:flex;align-items:center;gap:10px}
.col-hdr .hint{font-size:10px;color:#333;font-weight:500;text-transform:none;letter-spacing:0}

/* Tables */
table{width:100%;border-collapse:collapse}
th{padding:8px 16px;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#444;font-weight:600;border-bottom:1px solid #1a1a1a;text-align:left}
table.sortable th{cursor:pointer;user-select:none;transition:color .12s}
table.sortable th:hover{color:#aaa}
table.sortable th.sorted{color:#e0e0e0}
table.sortable th .arrow{display:inline-block;margin-left:4px;opacity:0;font-size:9px;transition:opacity .12s}
table.sortable th.sorted .arrow{opacity:1;color:#00e676}
td{padding:8px 16px;font-size:13px;border-bottom:1px solid #161616}
tr:last-child td{border-bottom:none}
tr:hover td{background:#151515}
.tag-y{background:#0e2a14;color:#00e676;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700}
.tag-n{background:#2a0e0e;color:#ff5252;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700}
.empty{text-align:center;color:#333;padding:24px;font-size:12px}

</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-left">
    <h1><span class="dot"></span> Kalshi Bot</h1>
    <span class="badge mono" id="cycle-badge">—</span>
    <span class="badge mono" id="time-badge">—</span>
  </div>
  <span class="badge err" id="err-badge" style="display:none"></span>
</div>

<div class="kpis">
  <div class="kpi">
    <div class="lbl">Portfolio</div>
    <div class="val white" id="equity">—</div>
    <div class="sub" id="cash-sub">—</div>
  </div>
  <div class="kpi">
    <div class="lbl">Today's P&L</div>
    <div class="val" id="today-pnl">—</div>
    <div class="sub" id="today-sub">—</div>
  </div>
  <div class="kpi">
    <div class="lbl">All-time P&L</div>
    <div class="val" id="alltime-pnl">—</div>
    <div class="sub" id="alltime-sub">—</div>
  </div>
  <div class="kpi">
    <div class="lbl">Open Positions</div>
    <div class="val white" id="open-count">—</div>
    <div class="sub" id="pos-val-sub">—</div>
  </div>
  <div class="kpi">
    <div class="lbl">Unrealized P&L</div>
    <div class="val" id="unrealized-pnl">—</div>
    <div class="sub" id="unrealized-sub">across open positions</div>
  </div>
</div>

<div class="chart-section">
  <div class="chart-card">
    <div class="chart-header">
      <div class="chart-title">Equity</div>
      <div class="timebtns">
        <button class="tbtn active" onclick="setRange('1D')">1D</button>
        <button class="tbtn" onclick="setRange('1W')">1W</button>
        <button class="tbtn" onclick="setRange('1M')">1M</button>
        <button class="tbtn" onclick="setRange('ALL')">ALL</button>
      </div>
    </div>
    <div class="chart-wrap"><canvas id="chart"></canvas></div>
  <div id="chart-tooltip"><div class="tt-time" id="tt-time"></div><div class="tt-val" id="tt-val"></div><div class="tt-chg" id="tt-chg"></div></div>
  </div>
</div>

<div class="cols">
  <div class="col-card">
    <div class="col-hdr">Open Positions <span class="hint">click a column to sort</span></div>
    <table class="sortable" id="pos-table">
      <thead><tr>
        <th data-key="ticker"         data-type="str">Market</th>
        <th data-key="side"           data-type="str">Side</th>
        <th data-key="qty"            data-type="num">Qty</th>
        <th data-key="entry_c"        data-type="num">Entry</th>
        <th data-key="cost_basis"     data-type="num">Cost</th>
        <th data-key="mark_c"         data-type="num">Mark</th>
        <th data-key="val"            data-type="num">Value</th>
        <th data-key="unrealized_pnl" data-type="num">Unreal P&amp;L</th>
      </tr></thead>
      <tbody id="pos-body"></tbody>
    </table>
  </div>
  <div class="col-card">
    <div class="col-hdr">Recent Fills</div>
    <table>
      <thead><tr><th>Time</th><th>Market</th><th>Side</th><th>Action</th><th>Qty</th><th>Price</th><th>P&amp;L</th></tr></thead>
      <tbody id="fills-body"></tbody>
    </table>
  </div>
</div>

<script>
let chart = null, allHistory = [], activeRange = '1D';
let _positions = [];
// Default sort: largest value at top, matching the previous behavior.
let _sort = { key: 'val', type: 'num', dir: 'desc' };

function _cmp(a, b, key, type, dir) {
  let av = a[key], bv = b[key];
  // Push nulls to bottom regardless of direction
  if (av == null && bv == null) return 0;
  if (av == null) return 1;
  if (bv == null) return -1;
  if (type === 'num') {
    av = Number(av); bv = Number(bv);
  } else {
    av = String(av).toLowerCase(); bv = String(bv).toLowerCase();
  }
  if (av < bv) return dir === 'asc' ? -1 :  1;
  if (av > bv) return dir === 'asc' ?  1 : -1;
  return 0;
}

function renderPositions() {
  const tbl = document.getElementById('pos-table');
  // Refresh header arrows
  tbl.querySelectorAll('thead th').forEach(th => {
    const isSorted = th.dataset.key === _sort.key;
    th.classList.toggle('sorted', isSorted);
    let arrow = th.querySelector('.arrow');
    if (!arrow) {
      arrow = document.createElement('span');
      arrow.className = 'arrow';
      th.appendChild(arrow);
    }
    arrow.textContent = isSorted ? (_sort.dir === 'asc' ? '▲' : '▼') : '';
  });

  const pb = $('pos-body');
  if (!_positions.length) {
    pb.innerHTML = '<tr><td colspan="8" class="empty">No open positions</td></tr>';
    return;
  }

  const sorted = _positions.slice().sort((a, b) => _cmp(a, b, _sort.key, _sort.type, _sort.dir));

  pb.innerHTML = sorted.map(p => {
    const upnl = p.unrealized_pnl;
    const pnlCell = upnl == null
      ? '<td class="muted">—</td>'
      : `<td class="mono ${upnl>=0?'green':'red'}" style="font-weight:600">${upnl>=0?'+':''}$${Math.abs(upnl).toFixed(2)}</td>`;
    const entryCell = p.entry_c == null
      ? '<td class="muted">—</td>'
      : `<td class="mono" style="color:#888">${p.entry_c}¢</td>`;
    const costCell = p.cost_basis == null
      ? '<td class="muted">—</td>'
      : `<td class="mono" style="color:#aaa">$${p.cost_basis.toFixed(2)}</td>`;
    return `<tr>
      <td class="mono" style="font-size:12px;color:#bbb">${p.ticker.replace(/KXMLBGAME-|KXNHLGAME-|KXNBAGAME-/,'')}</td>
      <td><span class="${p.side==='YES'?'tag-y':'tag-n'}">${p.side}</span></td>
      <td>${p.qty}</td>
      ${entryCell}
      ${costCell}
      <td class="mono">${p.mark_c}¢</td>
      <td class="mono">$${p.val.toFixed(2)}</td>
      ${pnlCell}
    </tr>`;
  }).join('');
}

// One-time: attach click handlers to sortable headers
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('#pos-table thead th').forEach(th => {
    th.addEventListener('click', () => {
      const key  = th.dataset.key;
      const type = th.dataset.type || 'str';
      if (_sort.key === key) {
        _sort.dir = _sort.dir === 'asc' ? 'desc' : 'asc';
      } else {
        _sort.key  = key;
        _sort.type = type;
        // Numbers default to descending (biggest first); strings to ascending.
        _sort.dir  = type === 'num' ? 'desc' : 'asc';
      }
      renderPositions();
    });
  });
});
let _chartSampled = [];  // keep raw sampled data for crosshair interpolation

const $ = id => document.getElementById(id);
const fmt = (n, sign=false) => {
  if (n==null||isNaN(n)) return '—';
  const s = sign ? (n>=0?'+':'-') : (n<0?'-':'');
  return s + '$' + Math.abs(n).toFixed(2);
};
const cc = n => n>0?'green':n<0?'red':'muted';

// ── Crosshair plugin ─────────────────────────────────────────────────────────
const crosshairPlugin = {
  id: 'crosshair',
  afterDraw(chart) {
    const cx = chart._crosshairX;
    if (cx == null) return;
    const {ctx, chartArea: {top, bottom, left, right}, scales} = chart;
    if (cx < left || cx > right) return;

    // Interpolate Y value at cursor pixel
    const ratio = (cx - left) / (right - left);
    const ds = chart.data.datasets[0].data;
    const rawIdx = ratio * (ds.length - 1);
    const i0 = Math.floor(rawIdx), i1 = Math.min(i0 + 1, ds.length - 1);
    const t  = rawIdx - i0;
    const val = ds[i0] + (ds[i1] - ds[i0]) * t;
    const yPx = scales.y.getPixelForValue(val);

    ctx.save();

    // Vertical rule
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.moveTo(cx, top);
    ctx.lineTo(cx, bottom);
    ctx.stroke();

    // Horizontal rule to y-axis
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.setLineDash([3, 6]);
    ctx.moveTo(left, yPx);
    ctx.lineTo(cx, yPx);
    ctx.stroke();

    ctx.setLineDash([]);

    // Glow halo
    ctx.beginPath();
    ctx.arc(cx, yPx, 7, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,230,118,0.15)';
    ctx.fill();

    // Dot
    ctx.beginPath();
    ctx.arc(cx, yPx, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#00e676';
    ctx.shadowColor = '#00e676';
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = '#0d0d0d';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.restore();
  }
};

function _attachCrosshair(chartInst) {
  const canvas = chartInst.canvas;
  const tt = $('chart-tooltip');

  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const {chartArea: {left, right}, scales} = chartInst;
    const mx = e.clientX - rect.left;

    if (mx < left || mx > right) {
      chartInst._crosshairX = null;
      tt.style.display = 'none';
      chartInst.draw();
      return;
    }

    chartInst._crosshairX = mx;

    // Interpolate value
    const ds = chartInst.data.datasets[0].data;
    const ratio = (mx - left) / (right - left);
    const rawIdx = ratio * (ds.length - 1);
    const i0 = Math.floor(rawIdx), i1 = Math.min(i0 + 1, ds.length - 1);
    const t  = rawIdx - i0;
    const val = ds[i0] + (ds[i1] - ds[i0]) * t;

    // Timestamp interpolation from raw sampled data
    let timeStr = '';
    if (_chartSampled.length > 1) {
      const si0 = Math.min(i0, _chartSampled.length - 1);
      const si1 = Math.min(i1, _chartSampled.length - 1);
      const ts = _chartSampled[si0].ts + (_chartSampled[si1].ts - _chartSampled[si0].ts) * t;
      const d = new Date(ts * 1000);
      if (activeRange === '1D') {
        timeStr = d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
      } else {
        timeStr = d.toLocaleDateString([], {month:'short', day:'numeric'}) + '  ' +
                  d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
      }
    }

    // P&L vs first visible point
    const first = ds[0];
    const chg = val - first;
    const chgPct = first ? (chg / first * 100) : 0;
    const chgSign = chg >= 0 ? '+' : '';

    $('tt-time').textContent = timeStr;
    $('tt-val').textContent  = '$' + val.toFixed(2);
    const chgEl = $('tt-chg');
    chgEl.textContent = `${chgSign}$${chg.toFixed(2)}  (${chgSign}${chgPct.toFixed(2)}%)`;
    chgEl.style.color = chg >= 0 ? '#00e676' : '#ff5252';

    // Position tooltip — keep it on screen
    const TW = 170, TH = 70;
    let tx = e.clientX + 16;
    let ty = e.clientY - TH / 2;
    if (tx + TW > window.innerWidth - 8) tx = e.clientX - TW - 16;
    if (ty < 8) ty = 8;
    if (ty + TH > window.innerHeight - 8) ty = window.innerHeight - TH - 8;
    tt.style.left = tx + 'px';
    tt.style.top  = ty + 'px';
    tt.style.display = 'block';

    chartInst.draw();
  });

  canvas.addEventListener('mouseleave', () => {
    chartInst._crosshairX = null;
    tt.style.display = 'none';
    chartInst.draw();
  });
}
// ─────────────────────────────────────────────────────────────────────────────

function setRange(r) {
  activeRange = r;
  document.querySelectorAll('.tbtn').forEach(b => b.classList.toggle('active', b.textContent===r));
  updateChart(allHistory);
}

function filterHistory(hist, range) {
  const now = Date.now()/1000;
  const cuts = {'1D': 86400, '1W': 604800, '1M': 2592000};
  const cut = cuts[range];
  if (!cut) return hist;
  return hist.filter(h => h.ts >= now - cut);
}

function updateChart(hist) {
  const data = filterHistory(hist, activeRange);
  if (!data.length) return;

  // Downsample to max 800 points for performance
  const step = Math.max(1, Math.floor(data.length / 800));
  const sampled = data.filter((_, i) => i % step === 0);
  _chartSampled = sampled;

  const labels = sampled.map(h => {
    const d = new Date(h.ts * 1000);
    if (activeRange === '1D') return d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
    return d.toLocaleDateString([], {month:'short',day:'numeric'}) + ' ' + d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  });
  const eq = sampled.map(h => h.equity);

  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = eq;
    chart.update('none');
    return;
  }

  const ctx2d = $('chart').getContext('2d');
  // Gradient fill
  const grad = ctx2d.createLinearGradient(0, 0, 0, 220);
  grad.addColorStop(0,   'rgba(0,230,118,0.18)');
  grad.addColorStop(0.6, 'rgba(0,230,118,0.04)');
  grad.addColorStop(1,   'rgba(0,230,118,0)');

  chart = new Chart(ctx2d, {
    type: 'line',
    plugins: [crosshairPlugin],
    data: {
      labels,
      datasets: [{
        data: eq,
        borderColor: '#00e676',
        backgroundColor: grad,
        fill: true,
        tension: 0.35,
        pointRadius: 0,
        pointHoverRadius: 0,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'none' },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      },
      scales: {
        x: {
          grid: { color: '#161616' },
          ticks: { color: '#444', maxTicksLimit: 8, font: { size: 10 } },
        },
        y: {
          grid: { color: '#161616' },
          ticks: { color: '#444', font: { size: 10 }, callback: v => '$'+v.toFixed(0) },
        },
      },
    },
  });

  _attachCrosshair(chart);
}

function render(d) {
  // KPIs
  $('equity').textContent = fmt(d.equity);
  $('cash-sub').textContent = `$${d.cash.toFixed(2)} cash · $${d.pos_val.toFixed(2)} positions`;

  $('today-pnl').textContent = fmt(d.today_pnl, true);
  $('today-pnl').className = 'val ' + cc(d.today_pnl);
  $('today-sub').textContent = (d.today_pct >= 0 ? '+' : '') + d.today_pct.toFixed(1) + '% vs 24h ago';

  $('alltime-pnl').textContent = fmt(d.alltime_pnl, true);
  $('alltime-pnl').className = 'val ' + cc(d.alltime_pnl);
  $('alltime-sub').textContent = (d.alltime_pct >= 0 ? '+' : '') + d.alltime_pct.toFixed(1) + '% from $' + d.starting_bal.toFixed(0);

  $('open-count').textContent = d.open_count;
  $('pos-val-sub').textContent = '$' + d.pos_val.toFixed(2) + ' at mark';

  // Unrealized P&L — sum across all positions that have a known basis
  const knownPnl = d.open_positions.filter(p => p.unrealized_pnl != null);
  if (knownPnl.length) {
    const totalUpnl = knownPnl.reduce((s, p) => s + p.unrealized_pnl, 0);
    $('unrealized-pnl').textContent = fmt(totalUpnl, true);
    $('unrealized-pnl').className = 'val ' + cc(totalUpnl);
    $('unrealized-sub').textContent = knownPnl.length + ' of ' + d.open_count + ' positions w/ basis';
  } else {
    $('unrealized-pnl').textContent = '—';
    $('unrealized-pnl').className = 'val muted';
  }

  // Header badges
  $('cycle-badge').textContent = 'Cycle ' + (d.cycle || '—');
  $('time-badge').textContent = d.now;
  const eb = $('err-badge');
  if (d.errors > 0) { eb.style.display=''; eb.textContent = d.errors + ' errors'; }
  else eb.style.display = 'none';

  // Positions
  _positions = d.open_positions.slice();
  renderPositions();

  // Fills
  const fb = $('fills-body');
  if (!d.fills.length) {
    fb.innerHTML = '<tr><td colspan="7" class="empty">No fills yet</td></tr>';
  } else {
    fb.innerHTML = d.fills.map(f => `
      <tr>
        <td class="muted mono" style="font-size:11px">${f.time}</td>
        <td class="mono" style="font-size:11px;color:#bbb">${f.ticker.replace(/KXMLBGAME-|KXNHLGAME-|KXNBAGAME-/,'')}</td>
        <td>${f.side}</td>
        <td class="${f.action==='BUY'?'green':'red'}" style="font-weight:600">${f.action}</td>
        <td>${f.qty}</td>
        <td class="mono">${f.price_c}¢</td>
        <td class="mono ${f.action==='SELL'?(f.pnl>=0?'green':'red'):'muted'}" style="font-weight:600">
          ${f.action==='SELL' && f.pnl!=null ? (f.pnl>=0?'+':'')+f.pnl.toFixed(2) : '—'}
        </td>
      </tr>`).join('');
  }

  // Chart
  allHistory = d.history;
  updateChart(allHistory);
}

async function refresh() {
  try {
    const d = await (await fetch('/api/snapshot')).json();
    render(d);
  } catch(e) { console.error(e); }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""

@app.route("/")
def index(): return render_template_string(HTML)

if __name__ == "__main__":
    print("Dashboard → http://localhost:5555")
    app.run(host="0.0.0.0", port=5555, debug=False, use_reloader=False)
