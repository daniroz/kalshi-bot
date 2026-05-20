# Kalshi Trading Bot

Automated trading bot for [Kalshi](https://kalshi.com) prediction markets. Multiple strategies, real risk management, live monitoring dashboard.

> **⚠️ Real money.** `DEMO_MODE=false` in `.env` makes this live. Read the risk section before flipping that switch.

---

## Quick start

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: add KALSHI_API_KEY and put your private key at kalshi_private.pem
# Optionally tune anything in config.yaml

python main.py             # live bot
python main.py --scan      # one-shot opportunity scan, no orders
python dashboard.py        # open http://localhost:5555 in your browser
python backtest.py --help  # historical replay (see "Backtesting" below)
```

The bot runs as a macOS LaunchAgent in production (`launchctl stop / start com.kalshibot`). Don't `nohup` it manually — you'll end up with duplicate processes fighting over the same API key.

---

## Philosophy

After ~36 hours of intense iteration and one real-money disaster (-$12 on a stale-data SPX trade), this codebase converged on three principles:

1. **Observe, don't predict.** The most reliable edges come from comparing the Kalshi order book to verifiable, real-world data (temperatures, prices, scores). Predictive models are turned off by default until backtests prove positive EV with current risk rules.
2. **Listen to the market.** If our model says "99% locked" but the order book prices it at 1¢, the market knows something we don't. A `MIN_PRICE_C = 20` floor on every strategy refuses these trades.
3. **Fees are not a rounding error.** Kalshi charges `0.07 × price × (1-price)` per contract per leg — at 50¢ that's 1.75¢ each way, 3.5¢ round trip. Every strategy now does fee-aware edge math.

---

## Strategy roster

The bot has **18 strategies** organized in two families. Each is its own file in `strategies/`. Enable/disable flags live in `config.yaml` (or `.env` overrides). Currently 9 are ON, 9 are OFF.

### ✅ Structural-edge strategies (3 ON)

These exploit mechanical inefficiencies — bid-ask spread capture, internal pricing violations, cross-market price gaps. Don't depend on us being smarter than the market.

| Strategy | File | What it does |
|---|---|---|
| **Market Maker** | `market_maker.py` | Quotes both sides of liquid markets (≥$5k 24h volume, ≥6¢ source spread), captures the spread minus fees. Quotes 2¢ inside the touch on each side, exits half-fills in 45s, cancels orphan orders on startup. |
| **Mispricing** | `mispricing.py` | Internal Kalshi arbitrage: when YES_ask + NO_ask < $1 minus 2-leg fees, buy both for a guaranteed payoff. Min 6¢ edge. |
| **Arbitrage** | `arbitrage.py` | Cross-market signal vs Polymarket. We only place the Kalshi leg (renamed from "arb" because we don't hedge). 0.50 TF-IDF title match required, 8¢ min edge to absorb model risk. |

### ✅ Verified-settlement strategies (6 ON)

These compare Kalshi prices to real-world data feeds in the final hours/minutes before settlement. If reality has already locked in the outcome and the market still prices it ≤89¢ (with ≥10¢ post-fee edge), we trade.

Every settlement strategy inherits the same defensive layers:
- **MARKET-SANITY floor** (`MIN_PRICE_C = 20` for most, 25¢ for forex) — refuse if order book disagrees by >50¢
- **Tier-aware sizing** — position size scales with the cash-tier ladder
- **Per-contract edge ≥ 10¢ after fees**
- **Same-game conflict block** via the risk manager

| Strategy | File | Data source | Resolution timing |
|---|---|---|---|
| **Weather** | `settlement.py` | Open-Meteo hourly observed temps | End-of-day per city local TZ; trades 0.5–8h before resolve |
| **Crypto** | `settlement_crypto.py` | Coinbase spot (BTC, ETH, DOGE, XRP) | CF Benchmarks RTI 60-sec avg at hourly bars; trades 1–90 min before resolve |
| **Stocks** | `settlement_stocks.py` | Yahoo `^GSPC`/`^NDX`/`^DJI` | 4pm ET equity close; trades **3–30 min before** (60s window proved deadly — see "Lessons") |
| **Sports** | `settlement_sports.py` | ESPN live scoreboard | Game end; trades when win prob ≥ 99.5%, lead ≥ {8 NBA / 2 NHL / 3 MLB}, time-remaining ≤ {6 min NBA·NHL / 3 innings MLB} |
| **Commodities** | `settlement_commodities.py` | Yahoo `CL=F` | NYMEX 2:30pm ET settle (WTI crude). Active 9am–14:30 ET weekdays only |
| **Forex** | `settlement_forex.py` | Yahoo `EURUSD=X` / `USDJPY=X` | 10am ET open price (point-in-time, riskiest of the 6). Extra defenses: 90s hard stop, macro-release blackout ±15 min of 8:30/10:00/14:00 ET releases |

### ⚠️ Predictive strategies (9 OFF)

These existed earlier and lost money. They're disabled pending backtest evidence that they have positive EV under current risk rules (the `approve_trade` per-contract-edge bug they suffered from is now fixed).

| Strategy | File | Why it's off |
|---|---|---|
| **Smart$** | `smart_money.py` | Volume-spike fading produced thin per-contract edges |
| **Sports** (predictive) | `sports.py` | In-game win-prob model that was wrong for same-series-different-game tickers (now fixed — needs backtest) |
| **Crypto** (predictive) | `crypto.py` | Per-contract edges too thin to overcome fees |
| **Weather** (predictive) | `weather.py` | Forecast-based — model risk + over-concentration in single-day weather, lost ~$60 |
| **Intraday** | `intraday.py` | Mean-reversion plays inside the day; signal quality low |
| **Orderbook** | `orderbook.py` | Order-book imbalance signal; too noisy in thin markets |
| **Momentum** | `momentum.py` | Trend continuation; couldn't separate signal from noise |
| **News** | `news.py` | RSS → market reaction. Interpretation is model-heavy, fragile |
| **Calendar** | `calendar.py` | Economic-release plays; data is point-in-time, hard to verify |

To re-enable any of these, flip its `enabled: true` in `config.yaml`. Re-running it through `backtest.py` first is strongly recommended.

---

## Risk management

The risk system lives in `risk/manager.py` (`approve_trade()` is the bouncer). Every order goes through it. Layers, in the order they fire:

1. **Halt check** — daily loss limit hit → all entries blocked.
2. **Cash emergency** — cash < 10% of portfolio → blocked. (Tier 4)
3. **Same-game conflict** — already holding opposite side of same game → blocked. Threshold suffixes like `-T90` are exempted (they're independent markets, not "sides").
4. **$5 minimum trade size** — kills dust trades whose fees exceed expected edge.
5. **Tier-adjusted position cap** — initial entry can't exceed `5% × tier_mult × balance`. (Tier mults: 1.0 / 0.7 / 0.4 / 0.2 / 0 across healthy → emergency.)
6. **Tier-adjusted total exposure cap** — total per-ticker exposure can't exceed `12% × tier_mult × balance`.
7. **Per-contract edge floor** — `edge_dollars / contracts ≥ tier-adjusted min_edge`. This is the critical fix from yesterday — strategies pass total edge dollars, we divide by contracts to compare apples-to-apples.

### Cash-tier ladder

Instead of a hard 35% cash wall (which blocked ALL trading when breached), the tier ladder raises the bar gracefully:

| Cash % | Tier | Edge required | Position size |
|---|---|---|---|
| ≥ 35% | 0 healthy | 1× | 1× |
| 25–35% | 1 mild | 1.5× | 0.7× |
| 15–25% | 2 moderate | 2.5× | 0.4× |
| 10–15% | 3 tight | 4.0× | 0.2× |
| < 10% | 4 emergency | hard block | hard block |

`coach.py` runs every cycle and updates the tier based on real Kalshi cash balance.

### What's `coach.py`?

A monitoring layer that runs alongside the strategies each cycle. It:
- Computes and applies the cash tier
- Warns (warn-only, no fire-sales) on over-concentration in any single game
- Tracks fill rate and idle time
- Prints a full performance breakdown every 10 minutes

It does NOT trade. After a real loss yesterday from an auto-reducer trying to sell concentrated positions, all the "active" actions were converted to warnings.

---

## Configuration

Two files:

### `config.yaml` — all tunable parameters
Per-strategy enable flag + key tunables. Validated on load. See the file itself for the full schema.

### `.env` — secrets and runtime overrides
Only `KALSHI_API_KEY`, `DEMO_MODE`, and emergency overrides (`MIN_EDGE_THRESHOLD`, etc.) live here. Never commit this file — it's in `.gitignore`.

Override precedence (highest wins):
1. `.env` variable
2. `config.yaml` value
3. Hardcoded default in code

---

## Dashboard

`python dashboard.py` serves `http://localhost:5555`:

- Portfolio equity chart with crosshair tooltip and 1D/1W/1M/ALL ranges
- KPI tiles: equity, today's P&L, all-time P&L, open positions, unrealized P&L
- **Open positions table** with Entry, Mark, Value, Unrealized P&L (color-coded)
- **Recent fills table** with realized P&L on sells
- **Strategy panel** — which are on/off, fills per strategy this session

Phone access: dashboard listens on `0.0.0.0:5555`. On home WiFi, point your phone Safari at `http://<your-mac-ip>:5555`. From anywhere, use [Tailscale](https://tailscale.com) — install on Mac + phone, hit the Tailscale IP.

---

## Backtesting

`backtest.py` replays Kalshi historical market data and pipes it through each strategy's `_evaluate()` function, simulating fills at the actual order-book ask. Useful for:

- Validating a new strategy before turning it on live
- Re-evaluating the 9 OFF strategies against current risk rules
- Tuning per-strategy `min_edge` / `max_trade_dollars` etc.

```bash
python backtest.py --strategy settle-crypto --days 30
python backtest.py --strategy settle-stocks --start 2026-05-01 --end 2026-05-19
python backtest.py --all --days 7
```

Reports: total return, Sharpe (annualized), win rate, avg edge captured per trade, max drawdown, by-strategy breakdown. Output also saved as CSV in `backtest_results/`.

See `backtest.py --help` for full options and the inline docstring for limitations (Kalshi's historical orderbook is sparse for older markets; trade-history fallback estimates slippage from spread).

---

## Lessons learned (the expensive ones)

Documenting these so future-me doesn't repeat them:

1. **Yahoo Finance `regularMarketPrice` can be 1-2 minutes stale during fast markets.** Cost us $12 on a KXINX SPX trade — bot saw 7364, real SPX was already in the high-6000s. **Fix:** use the latest 1-min bar close from `/v8/chart`, plus the MARKET-SANITY floor below.

2. **If your model says "locked YES" but the market prices YES at 1¢, the market is right.** Bot bought YES at 1¢ on locked-by-our-model SPX > 7050. SPX closed below 7050. -$11.81 burned. **Fix:** `MIN_PRICE_C = 20` (25 for forex) across every settlement strategy.

3. **Per-strategy "edge" needs to be PER-CONTRACT, not TOTAL.** Strategies were passing `edge × qty` to `approve_trade`, which compared total dollars to `min_edge`. So 50 contracts × 0.5¢ edge = $0.25 passed a 3¢ threshold. **Fix:** `approve_trade` now divides edge_dollars by contracts before comparison.

4. **Same-game NO+NO is a guaranteed loss.** Market maker quotes were going short YES on both teams of an NHL game (= long NO on both = one MUST resolve YES = guaranteed loss). **Fix:** `same_game_conflict()` check in risk manager + orphan-order cleanup at MM startup.

5. **`order_type="market"` is rejected by Kalshi.** All 7 strategies had this in their `_exit()` paths and were silently failing 400 every time. **Fix:** limit at current bid for our side. Fills immediately against resting bid, same effect as market.

6. **The Polymarket "arbitrage" wasn't arbitrage.** We placed only the Kalshi leg — naked exposure dressed up as risk-free. Renamed in docs to "cross-market signal" and bumped min_edge to 8¢ to absorb model risk in title matching.

7. **A cash floor that blocks ALL new entries when breached is the wrong shape.** Cash is supposed to be deployable for high-conviction trades — a hard wall makes us less able to do the thing cash is for. **Fix:** graduated tier ladder (above).

8. **Live sports models can't tell the difference between a series' Game 2 and Game 4.** Bot was applying live in-game probability from Game 2 to a future Game 4 in the same series. -$13 in two trades. **Fix:** `_today_kalshi_date()` check — only act on tickers dated today.

---

## File tour

```
main.py                      Main loop: strategies in a threadpool, 15s cycle
dashboard.py                 Flask server for the local UI
coach.py                     Monitoring + cash-tier enforcement
config.yaml                  All tunables (single source of truth)
backtest.py                  Historical replay framework
clients/
  kalshi.py                  Kalshi REST + auth (RSA signature)
  polymarket.py              Polymarket public-API client
risk/
  manager.py                 The bouncer. approve_trade() is THE gate
strategies/
  market_maker.py            ✅ Structural
  mispricing.py              ✅ Structural
  arbitrage.py               ✅ Structural (signal, not arb)
  settlement.py              ✅ Verified — weather
  settlement_crypto.py       ✅ Verified — BTC/ETH/DOGE/XRP
  settlement_stocks.py       ✅ Verified — SPX/NDX/DJI
  settlement_sports.py       ✅ Verified — late-game NBA/NHL/MLB
  settlement_commodities.py  ✅ Verified — WTI
  settlement_forex.py        ✅ Verified — EUR/USD, USD/JPY
  weather.py                 ⚠️ OFF — predictive
  sports.py                  ⚠️ OFF — predictive (helper for settle-sports)
  crypto.py                  ⚠️ OFF — predictive
  smart_money.py             ⚠️ OFF — volume-spike
  intraday.py                ⚠️ OFF — mean-reversion
  orderbook.py               ⚠️ OFF — book imbalance
  momentum.py                ⚠️ OFF — trend
  news.py                    ⚠️ OFF — RSS sentiment
  calendar.py                ⚠️ OFF — economic releases
utils/
  markets.py                 Liquid-markets fetcher
  logger.py                  Rich-formatted logger
  state.py                   Persisted open positions across restarts
  alerts.py                  External alerting hooks
  fill_tracker.py            Fill history
```

---

## License

Personal use. Not affiliated with Kalshi.
