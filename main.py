"""
Kalshi trading bot — main loop.

Strategies running in rotation:
  1. Arbitrage      — Kalshi vs Polymarket price discrepancies
  2. Market-making  — quote both sides on liquid markets
  3. Mispricing     — internal arb + near-expiry stale prices
  4. Smart money    — volume-spike detection on liquid markets
  5. Weather        — Open-Meteo forecast vs Kalshi weather markets
  6. Orderbook      — orderbook depth imbalance signals
  7. Momentum       — price momentum over last N cycles

Usage:
  python main.py            # live loop
  python main.py --scan     # scan once, print opportunities, no orders
"""

import os
import sys
import time
import argparse
from dotenv import load_dotenv
from rich.table import Table
from rich.live import Live

load_dotenv()

from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketClient
from risk.manager import RiskManager, RiskConfig
from strategies.arbitrage import ArbitrageStrategy
from strategies.market_maker import MarketMakerStrategy
from strategies.mispricing import MispricingStrategy
from strategies.smart_money import SmartMoneyStrategy
from strategies.weather import WeatherStrategy
from strategies.orderbook import OrderbookStrategy
from strategies.momentum import MomentumStrategy
from strategies.intraday import IntradayStrategy
from strategies.sports import SportsStrategy
from strategies.crypto import CryptoStrategy
from strategies.news import NewsStrategy
from strategies.calendar import CalendarStrategy
from strategies.settlement import SettlementStrategy
from strategies.settlement_crypto import CryptoSettlementStrategy
from strategies.settlement_stocks import StockSettlementStrategy
from strategies.settlement_sports import SportsSettlementStrategy
from strategies.settlement_commodities import CommoditySettlementStrategy
from strategies.settlement_forex import ForexSettlementStrategy
from strategies.ml_calibration import MLCalibrationStrategy
from strategies.favorite_bias import FavoriteBiasStrategy
from coach import Coach
from utils.logger import log, console
from utils.alerts import alert_trade, alert_error, alert_balance, alert_halt


def build_clients(demo: bool):
    api_key  = os.environ["KALSHI_API_KEY"]
    key_file = os.path.join(os.path.dirname(__file__), "kalshi_private.pem")
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"Private key file not found: {key_file}")
    kalshi = KalshiClient(api_key=api_key, key_file=key_file, demo=demo)
    poly   = PolymarketClient()
    return kalshi, poly


def build_risk() -> RiskManager:
    # Read from config.yaml (env vars still override via utils.config)
    from utils.config import config
    cfg = RiskConfig(**config.risk_config_dict())
    return RiskManager(cfg)


def print_status(risk: RiskManager, cycle: int):
    s = risk.status()
    table = Table(title=f"Bot Status — cycle {cycle}", show_header=False, box=None)
    table.add_row("Balance",       f"${s['balance']:.2f}")
    table.add_row("Day PnL",       f"${s['daily_pnl']:+.2f}  ({-s['daily_loss_pct']:+.1f}%)")
    table.add_row("Open positions", str(s['open_positions']))
    table.add_row("Halted",        "[red]YES[/red]" if s['halted'] else "[green]NO[/green]")
    if s["halted"]:
        table.add_row("Reason", s["halt_reason"])
    console.print(table)


def scan_mode(kalshi, poly, risk):
    """Print opportunities without placing orders."""
    arb     = ArbitrageStrategy(kalshi, poly, risk)
    mis     = MispricingStrategy(kalshi, risk)
    smart   = SmartMoneyStrategy(kalshi, risk)
    weather = WeatherStrategy(kalshi, risk)
    ob      = OrderbookStrategy(kalshi, risk)
    mom     = MomentumStrategy(kalshi, risk)

    console.rule("[bold]Arbitrage opportunities")
    opps = arb.scan()
    if opps:
        for o in opps[:10]:
            console.print(
                f"  [green]{o['ticker']}[/green]  side={o['side']}  "
                f"kalshi={o['kalshi_price']*100:.0f}¢  poly={o['poly_price']*100:.0f}¢  "
                f"edge=[bold]{o['edge']*100:.1f}¢[/bold]"
            )
            console.print(f"    K: {o['kalshi_title']}")
            console.print(f"    P: {o['poly_question']}")
    else:
        console.print("  None found")

    console.rule("[bold]Mispricing opportunities")
    mis_opps = mis.scan()
    if mis_opps:
        for o in mis_opps[:10]:
            console.print(
                f"  [yellow]{o['ticker']}[/yellow]  type={o['type']}  "
                f"edge=[bold]{o['edge']*100:.1f}¢[/bold]  {o.get('title','')[:60]}"
            )
    else:
        console.print("  None found")

    console.rule("[bold]Smart money (volume spikes)")
    smart_opps = smart.scan()
    if smart_opps:
        for o in smart_opps[:10]:
            console.print(
                f"  [cyan]{o['ticker']}[/cyan]  side={o['side']}  "
                f"vol_ratio={o['vol_ratio']}x  {o['title'][:60]}"
            )
    else:
        console.print("  None found")

    console.rule("[bold]Weather edges")
    wx_opps = weather.scan()
    if wx_opps:
        for o in wx_opps[:10]:
            console.print(
                f"  [blue]{o['ticker']}[/blue]  side={o['side']}  "
                f"model={o['model_prob']*100:.0f}%  kalshi={o['kalshi_prob']*100:.0f}%  "
                f"edge=[bold]{o['edge']*100:.0f}¢[/bold]  {o['title'][:50]}"
            )
    else:
        console.print("  None found")

    console.rule("[bold]Orderbook imbalance")
    ob_opps = ob.scan()
    if ob_opps:
        for o in ob_opps[:10]:
            console.print(
                f"  [magenta]{o['ticker']}[/magenta]  side={o['side']}  "
                f"ratio={o['ratio']}  yes={o['yes_qty']} no={o['no_qty']}  {o['title'][:50]}"
            )
    else:
        console.print("  None found")

    console.rule("[bold]Price momentum")
    mom_opps = mom.scan()
    if mom_opps:
        for o in mom_opps[:10]:
            console.print(
                f"  [white]{o['ticker']}[/white]  side={o['side']}  "
                f"move={o['move_c']}¢  {o['title'][:50]}"
            )
    else:
        console.print("  None found")


def main_loop(kalshi, poly, risk):
    coach   = Coach(kalshi, risk)
    # Arbitrage min_edge raised to 8¢ — title matching has model risk; need fat margin.
    arb     = ArbitrageStrategy(kalshi, poly, risk,
                                min_edge=float(os.getenv("ARB_MIN_EDGE", 0.08)))
    mm      = MarketMakerStrategy(kalshi, risk)
    # Mispricing internal-arb pays fees on BOTH legs; needs to beat 4¢+ of fees.
    mis     = MispricingStrategy(kalshi, risk,
                                 min_edge=float(os.getenv("MIS_MIN_EDGE", 0.06)))
    smart   = SmartMoneyStrategy(kalshi, risk)
    weather = WeatherStrategy(kalshi, risk)
    ob      = OrderbookStrategy(kalshi, risk)
    mom     = MomentumStrategy(kalshi, risk)
    intra   = IntradayStrategy(kalshi, risk)
    sports  = SportsStrategy(kalshi, risk)
    crypto  = CryptoStrategy(kalshi, risk)
    news    = NewsStrategy(kalshi, risk)
    cal     = CalendarStrategy(kalshi, risk)
    settle        = SettlementStrategy(kalshi, risk)
    settle_crypto = CryptoSettlementStrategy(kalshi, risk)
    settle_stocks = StockSettlementStrategy(kalshi, risk)
    settle_sports = SportsSettlementStrategy(kalshi, risk)
    settle_comm   = CommoditySettlementStrategy(kalshi, risk)
    settle_fx     = ForexSettlementStrategy(kalshi, risk)
    ml_cal        = MLCalibrationStrategy(kalshi, risk)
    favbias       = FavoriteBiasStrategy(kalshi, risk)

    # On restart, block re-entry into any ticker risk manager already holds.
    # Strategies use _last_entry to enforce per-ticker cooldowns.
    _now = time.time()
    for _ticker in risk.open_positions:
        for _strat in (arb, mis, weather, ob, intra, sports, crypto, news, cal):
            if hasattr(_strat, "_last_entry"):
                _strat._last_entry.setdefault(_ticker, _now)

    # Strategy enable flags read from config.yaml (env vars still override via utils.config)
    from utils.config import config as _cfg
    use_arb      = _cfg.strategy_enabled("arbitrage")
    use_mm       = _cfg.strategy_enabled("market_maker")
    use_mis      = _cfg.strategy_enabled("mispricing")
    use_smart    = _cfg.strategy_enabled("smart_money")
    use_weather  = _cfg.strategy_enabled("weather")
    use_ob       = _cfg.strategy_enabled("orderbook")
    use_mom      = _cfg.strategy_enabled("momentum")
    use_intra    = _cfg.strategy_enabled("intraday")
    use_sports   = _cfg.strategy_enabled("sports")
    use_crypto   = _cfg.strategy_enabled("crypto")
    use_news     = _cfg.strategy_enabled("news")
    use_cal      = _cfg.strategy_enabled("calendar")
    use_settle        = _cfg.strategy_enabled("settlement_weather")
    use_settle_crypto = _cfg.strategy_enabled("settlement_crypto")
    use_settle_stocks = _cfg.strategy_enabled("settlement_stocks")
    use_settle_sports = _cfg.strategy_enabled("settlement_sports")
    use_settle_comm   = _cfg.strategy_enabled("settlement_commodities")
    use_settle_fx     = _cfg.strategy_enabled("settlement_forex")
    use_ml_cal        = _cfg.strategy_enabled("ml_calibration")
    use_favbias       = _cfg.strategy_enabled("favorite_bias")

    cycle = 0
    LOOP_INTERVAL      = 15  # seconds between cycles
    BALANCE_SYNC_EVERY = 1   # sync balance every cycle (~15s) — critical for accurate 5% sizing
    STATUS_EVERY       = 2   # print status every ~30s

    log.info("Bot started. Press Ctrl-C to stop cleanly.")

    # Sync real balance before first cycle so position sizing uses actual funds.
    # Total portfolio value = cash balance + open position value (portfolio_value is in cents).
    try:
        bal = kalshi.get_balance()
        cash = float(bal.get("balance", 0))
        pos_value = float(bal.get("portfolio_value", 0)) / 100
        risk.update_balance(cash + pos_value)
        log.info(f"Balance synced: ${risk.balance:.2f} (cash=${cash:.2f} + positions=${pos_value:.2f})")
    except Exception as e:
        log.warning(f"Initial balance sync failed: {e}")

    try:
        while True:
            cycle += 1

            if cycle % BALANCE_SYNC_EVERY == 0:
                try:
                    bal = kalshi.get_balance()
                    cash = float(bal.get("balance", 0))
                    pos_value = float(bal.get("portfolio_value", 0)) / 100
                    total = cash + pos_value
                    risk.update_balance(total)
                    # Snapshot equity to SQLite for charting / drawdown queries
                    try:
                        from utils.db import log_equity
                        log_equity(total, cash, pos_value, len(risk.open_positions))
                    except Exception:
                        pass
                    # Send balance alert every 30 min
                    if cycle % 120 == 0:
                        starting = float(os.getenv("STARTING_BALANCE", 250))
                        alert_balance(total, total - starting, (total - starting) / starting * 100)
                except Exception as e:
                    log.warning(f"Balance sync failed: {e}")

            # Sync new fills from Kalshi into SQLite every 20 cycles (~5 min).
            # Idempotent (INSERT OR IGNORE on fill_id) so safe at any cadence.
            if cycle % 20 == 0:
                try:
                    from utils.db import sync_fills_from_kalshi
                    n = sync_fills_from_kalshi(kalshi, max_pages=3)
                    if n:
                        log.info(f"[db] synced {n} new fill(s) into SQLite")
                except Exception as e:
                    log.debug(f"[db] fill sync failed: {e}")

            # Sync open_positions with actual Kalshi position_fp every cycle.
            # This is the ground-truth guard: prevents the market maker from
            # accumulating large positions by re-quoting while the internal
            # reservation counter resets each cycle.
            risk.sync_positions(kalshi)

            # Coach: enforce limits, adjust edge, report performance
            coach.tick()

            if cycle % STATUS_EVERY == 0:
                print_status(risk, cycle)

            if risk.halted:
                log.warning(f"[bold red]HALTED:[/bold red] {risk.halt_reason}. Sleeping 5 min.")
                alert_halt(risk.halt_reason)
                mm.cancel_all()
                time.sleep(300)
                continue

            # ── Run all strategies in parallel threads ────────────────────
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def run_smart():
                if use_smart:
                    for opp in smart.scan()[:8]:
                        smart.execute(opp)

            def run_weather():
                if use_weather:
                    for opp in weather.scan()[:15]:
                        weather.execute(opp)

            def run_mis():
                if use_mis:
                    for opp in mis.scan()[:5]:
                        mis.execute(opp)

            def run_arb():
                if use_arb:
                    for opp in arb.scan()[:5]:
                        arb.execute(opp)

            def run_mm():
                if use_mm:
                    mm.refresh_quotes()

            def run_ob():
                if use_ob:
                    for opp in ob.scan()[:5]:
                        ob.execute(opp)

            def run_mom():
                if use_mom:
                    for opp in mom.scan()[:5]:
                        mom.execute(opp)

            def run_intra():
                if use_intra:
                    for opp in intra.scan()[:5]:
                        intra.execute(opp)

            def run_sports():
                if use_sports:
                    for opp in sports.scan()[:5]:
                        sports.execute(opp)

            def run_crypto():
                if use_crypto:
                    for opp in crypto.scan()[:5]:
                        crypto.execute(opp)

            def run_news():
                if use_news:
                    for opp in news.scan()[:5]:
                        news.execute(opp)

            def run_cal():
                if use_cal:
                    for opp in cal.scan()[:5]:
                        cal.execute(opp)

            def run_settle():
                if use_settle:
                    for opp in settle.scan()[:10]:
                        settle.execute(opp)

            def run_settle_crypto():
                if use_settle_crypto:
                    for opp in settle_crypto.scan()[:10]:
                        settle_crypto.execute(opp)

            def run_settle_stocks():
                if use_settle_stocks:
                    for opp in settle_stocks.scan()[:10]:
                        settle_stocks.execute(opp)

            def run_settle_sports():
                if use_settle_sports:
                    for opp in settle_sports.scan()[:10]:
                        settle_sports.execute(opp)

            def run_settle_comm():
                if use_settle_comm:
                    for opp in settle_comm.scan()[:10]:
                        settle_comm.execute(opp)

            def run_settle_fx():
                if use_settle_fx:
                    for opp in settle_fx.scan()[:10]:
                        settle_fx.execute(opp)

            def run_ml_cal():
                # FeatureRecorder runs INSIDE scan() regardless of use_ml_cal
                # so data is always being collected for future training.
                opps = ml_cal.scan()
                if use_ml_cal:
                    for opp in opps[:5]:
                        ml_cal.execute(opp)

            def run_favbias():
                if use_favbias:
                    for opp in favbias.scan()[:8]:
                        favbias.execute(opp)

            with ThreadPoolExecutor(max_workers=21) as ex:
                futures = [ex.submit(f) for f in (
                    run_smart, run_weather, run_mis, run_arb, run_mm,
                    run_ob, run_mom, run_intra, run_sports, run_crypto,
                    run_news, run_cal, run_settle,
                    run_settle_crypto, run_settle_stocks, run_settle_sports,
                    run_settle_comm, run_settle_fx, run_ml_cal, run_favbias,
                )]
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        import traceback, sys
                        tb = traceback.format_exc()
                        log.error(f"Strategy error: {tb}")
                        alert_error("main", str(e))
                        print(tb, file=sys.stderr, flush=True)

            log.info(f"Cycle {cycle} done. Sleeping {LOOP_INTERVAL}s.")
            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        log.info("Shutting down — cancelling open quotes...")
        mm.cancel_all()
        log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="Scan for opportunities only, no orders")
    parser.add_argument("--live", action="store_true", help="Force live (non-demo) mode")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt (for background service)")
    args = parser.parse_args()

    demo = not args.live and os.getenv("DEMO_MODE", "true").lower() == "true"
    if not demo:
        if not args.no_confirm:
            console.print("[bold red]LIVE MODE — real money will be traded[/bold red]")
            confirm = input("Type YES to confirm: ")
            if confirm.strip() != "YES":
                sys.exit("Aborted.")
    else:
        console.print("[bold yellow]DEMO MODE — no real money[/bold yellow]")

    kalshi, poly = build_clients(demo)
    risk = build_risk()

    if args.scan:
        scan_mode(kalshi, poly, risk)
    else:
        main_loop(kalshi, poly, risk)
