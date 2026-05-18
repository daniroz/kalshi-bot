"""
Market-making strategy: quote both sides of the spread and collect the difference.

Profitability math (the thing the old code got wrong):
  Kalshi fee per contract per leg = 0.07 * price * (1-price)
  At 50¢ that's 1.75¢. Round trip = 3.5¢ in fees.
  Quoting at mid-1/mid+1 captures 2¢ of spread → LOSES 1.5¢ per round trip.
  Quoting at mid-2/mid+2 captures 4¢ → breaks even at 50¢.
  Quoting at mid-3/mid+3 captures 6¢ → profits ~2.5¢ at 50¢.

We require source spreads of 6¢+, then quote inside by 3¢ each side so we
capture 6¢ minus fees. Adverse selection (half-fills against us) cuts realized
edge in practice — must monitor over hundreds of round trips.

Targets:
- 24h volume ≥ $5,000  (real liquidity; old $500 threshold attracted toxic flow)
- Raw spread ≥ 6¢
- Price between 15¢ and 85¢ (avoid extremes where fee % is wide vs price)
"""

import time
from dataclasses import dataclass
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log


MIN_VOLUME_24H    = 5000.0   # 10x stricter — only quote real liquidity
MIN_SPREAD_CENTS  = 6        # below 6¢ source spread we can't beat fees
QUOTE_INSIDE_C    = 2        # quote 2¢ inside the touch → captures (spread - 4¢)
MIN_PRICE_CENTS   = 15       # 50¢ is where fee % peaks; avoid extreme prices
MAX_PRICE_CENTS   = 85
MAX_MARKETS       = 3
HALF_FILL_EXIT_S  = 45       # was 300; directional drift on half-fills is the #1 leak

def _kalshi_fee_cents(price_c: int, contracts: int) -> float:
    """Kalshi fee formula in cents for a given price/size."""
    p = price_c / 100
    return 0.07 * p * (1 - p) * 100 * contracts


@dataclass
class HalfFill:
    ticker: str
    side: str        # "yes" or "no" — the side that filled
    contracts: int
    fill_ts: float
    resting_order_id: str   # the OTHER side still resting


class MarketMakerStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk = risk
        self._active_quotes: dict[str, tuple[str, str]] = {}   # ticker -> (bid_id, ask_id)
        self._half_fills:    dict[str, HalfFill] = {}          # ticker -> HalfFill
        self._orphan_cleanup_done: bool = False

    def _cleanup_orphan_orders(self):
        """One-time cleanup at MM startup: cancel every resting Kalshi order.
        Resting orders from previous bot sessions are NOT tracked in our in-memory
        _active_quotes dict, so they can fill silently and bypass approve_trade's
        same-game block. Fresh start = no surprises."""
        try:
            orders = self.kalshi.get_orders(status="resting").get("orders", [])
        except Exception as e:
            log.warning(f"[mm] Orphan cleanup: list orders failed: {e}")
            return
        n_cancelled = 0
        for o in orders:
            oid = o.get("order_id")
            if not oid:
                continue
            try:
                self.kalshi.cancel_order(oid)
                n_cancelled += 1
            except Exception:
                pass
        if n_cancelled:
            log.info(f"[mm] Orphan cleanup: cancelled {n_cancelled} stale resting orders from prior session")

    def _is_good_market(self, m: dict) -> bool:
        vol  = float(m.get("volume_24h_fp") or 0)
        bid  = float(m.get("yes_bid_dollars") or 0)
        ask  = float(m.get("yes_ask_dollars") or 0)
        spread_c = round((ask - bid) * 100)
        price_c  = round(bid * 100)
        return (
            vol >= MIN_VOLUME_24H
            and spread_c >= MIN_SPREAD_CENTS
            and MIN_PRICE_CENTS <= price_c <= MAX_PRICE_CENTS
        )

    def refresh_quotes(self):
        """Cancel stale quotes, check for one-sided fills, re-quote best markets."""
        if not self._orphan_cleanup_done:
            self._cleanup_orphan_orders()
            self._orphan_cleanup_done = True
        self._check_half_fills()

        markets = get_liquid_markets(self.kalshi, min_volume=MIN_VOLUME_24H)
        good = [m for m in markets if self._is_good_market(m)]
        good.sort(key=lambda m: m.get("volume_24h_fp", 0), reverse=True)

        # Build the set of game keys we already have exposure in — from held
        # positions AND resting half-fills. Never quote any team in those games.
        exposed_games: set = set()
        for ticker in self.risk.open_positions:
            exposed_games.add(self._game_key(ticker))
        for ticker in self._half_fills:
            exposed_games.add(self._game_key(ticker))

        # Never quote both sides of the same game — pick highest-volume ticker
        # per matchup and skip any game we already hold a position in.
        seen_games: set = set(exposed_games)   # start from already-held games
        deduped = []
        for m in good:
            game_key = self._game_key(m["ticker"])
            if game_key not in seen_games:
                seen_games.add(game_key)
                deduped.append(m)
        targets = deduped[:MAX_MARKETS]

        # Get all currently-resting order IDs from Kalshi
        try:
            resting_ids = {
                o["order_id"]
                for o in self.kalshi.get_orders(status="resting").get("orders", [])
            }
        except Exception:
            resting_ids = None   # API failure — skip fill check this cycle

        # Check each active quote pair for one-sided fills
        if resting_ids is not None:
            for ticker, (bid_id, ask_id) in list(self._active_quotes.items()):
                bid_resting = bid_id in resting_ids if bid_id else False
                ask_resting = ask_id in resting_ids if ask_id else False

                contracts = self.risk.open_positions.get(ticker, 1)
                if bid_id and not bid_resting and ask_id and ask_resting:
                    # Bid filled, ask still open — we're long YES
                    log.info(f"[mm] Half-fill detected {ticker}: bid filled, ask resting")
                    self._active_quotes.pop(ticker, None)
                    self._half_fills[ticker] = HalfFill(
                        ticker=ticker, side="yes", contracts=contracts,
                        fill_ts=time.time(), resting_order_id=ask_id,
                    )
                elif ask_id and not ask_resting and bid_id and bid_resting:
                    # Ask filled, bid still open — we're short YES (long NO)
                    log.info(f"[mm] Half-fill detected {ticker}: ask filled, bid resting")
                    self._active_quotes.pop(ticker, None)
                    self._half_fills[ticker] = HalfFill(
                        ticker=ticker, side="no", contracts=contracts,
                        fill_ts=time.time(), resting_order_id=bid_id,
                    )

        # Cancel existing quotes not in target list
        target_tickers = {m["ticker"] for m in targets}
        for ticker in list(self._active_quotes.keys()):
            if ticker not in target_tickers:
                self._cancel_quotes(ticker)

        # Post new quotes
        for m in targets:
            ticker = m["ticker"]
            if ticker in self._active_quotes or ticker in self._half_fills:
                continue
            self._post_quotes(m)

    def _check_half_fills(self):
        """Exit half-filled positions that have been open too long."""
        now = time.time()
        for ticker, hf in list(self._half_fills.items()):
            age = now - hf.fill_ts
            if age < HALF_FILL_EXIT_S:
                continue

            # Cancel the resting other leg
            try:
                self.kalshi.cancel_order(hf.resting_order_id)
            except Exception:
                pass

            # Exit the filled position at market (limit at current bid/ask)
            try:
                m = self.kalshi.get_market(ticker).get("market", {})
                if hf.side == "yes":
                    exit_price = int(float(m.get("yes_bid_dollars") or 0.01) * 100)
                    self.kalshi.place_order(
                        ticker=ticker, side="yes", action="sell",
                        count=hf.contracts, order_type="limit",
                        yes_price=max(1, exit_price),
                    )
                else:
                    exit_price = int(float(m.get("no_bid_dollars") or 0.01) * 100)
                    self.kalshi.place_order(
                        ticker=ticker, side="no", action="sell",
                        count=hf.contracts, order_type="limit",
                        no_price=max(1, exit_price),
                    )
                self.risk.record_close(ticker, hf.contracts)
                log.info(f"[mm] Half-fill exit {ticker} {hf.side.upper()} x{hf.contracts} after {age/60:.1f}m")
            except Exception as e:
                log.error(f"[mm] Half-fill exit failed {ticker}: {e}")
            finally:
                self._half_fills.pop(ticker, None)

    @staticmethod
    def _game_key(ticker: str) -> str:
        """Strip trailing -TEAM suffix to get the matchup key.
        Threshold markets (T90, T88.5) are independent, not 'sides' of same game."""
        parts = ticker.rsplit("-", 1)
        if len(parts) != 2 or len(parts[1]) > 4:
            return ticker
        suffix = parts[1]
        if suffix.startswith("T") and len(suffix) >= 2 and suffix[1].isdigit():
            return ticker   # threshold market — keep unique
        return parts[0]

    def _quote_contracts(self, ticker: str, price_c: int) -> int:
        """
        How many contracts to quote, based on the tier-adjusted risk caps.
        Initial entry:  up to effective_max_position_pct  of balance.
        Scaling total:  up to effective_max_position_scale_pct of balance.
        When cash is tight, these caps shrink so the bot still trades, just smaller.
        """
        price = price_c / 100
        if price <= 0:
            return 0
        balance  = self.risk.balance
        existing = self.risk.open_positions.get(ticker, 0)
        init_pct  = self.risk.effective_max_position_pct()
        scale_pct = self.risk.effective_max_position_scale_pct()
        if existing == 0:
            max_spend = balance * init_pct
        else:
            existing_notional = existing * price
            max_spend = balance * scale_pct - existing_notional
            max_spend = max(0, max_spend)
        contracts = int(max_spend / price)
        return max(1, contracts) if max_spend > 0 else 0

    def _post_quotes(self, m: dict):
        ticker   = m["ticker"]
        bid_c    = int(round(float(m.get("yes_bid_dollars") or 0) * 100))
        ask_c    = int(round(float(m.get("yes_ask_dollars") or 0) * 100))
        raw_spread = ask_c - bid_c
        if raw_spread < MIN_SPREAD_CENTS:
            return

        # Quote QUOTE_INSIDE_C inside the touch on each side. Round-trip we capture
        # raw_spread - 2*QUOTE_INSIDE_C in cents, then pay fees on both legs.
        our_bid  = bid_c + QUOTE_INSIDE_C
        our_ask  = ask_c - QUOTE_INSIDE_C
        if our_ask - our_bid < 2:
            return   # would invert; bail
        if our_bid < 1 or our_ask > 99:
            return

        # Honest edge calculation passed to approve_trade. The edge per round-trip
        # contract is (captured_spread - round_trip_fee). Half of that on bid leg.
        captured_c  = our_ask - our_bid          # cents per round trip
        contracts   = self._quote_contracts(ticker, our_bid)
        if contracts <= 0:
            return
        fee_round_c = (_kalshi_fee_cents(our_bid, 1) + _kalshi_fee_cents(our_ask, 1))
        edge_per_contract_dollars = (captured_c - fee_round_c) / 100
        if edge_per_contract_dollars <= 0:
            log.info(f"[mm] Skip {ticker}: net edge ≤ 0 after fees "
                     f"(captured={captured_c}¢ fees={fee_round_c:.1f}¢)")
            return

        ok_b, reason = self.risk.approve_trade(
            ticker, our_bid, contracts, edge_per_contract_dollars * contracts
        )
        if not ok_b:
            log.info(f"[mm] Skipping {ticker}: {reason}")
            return

        bid_id = ask_id = None
        try:
            r = self.kalshi.place_order(
                ticker=ticker, side="yes", action="buy",
                count=contracts, order_type="limit",
                yes_price=our_bid,
            )
            bid_id = r.get("order", {}).get("order_id")
            self.risk.record_open(ticker, contracts)
        except Exception as e:
            log.error(f"[mm] Bid failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            return

        try:
            r = self.kalshi.place_order(
                ticker=ticker, side="yes", action="sell",
                count=contracts, order_type="limit",
                yes_price=our_ask,
            )
            ask_id = r.get("order", {}).get("order_id")
        except Exception as e:
            log.error(f"[mm] Ask failed {ticker}: {e}")
            if bid_id:
                try:
                    self.kalshi.cancel_order(bid_id)
                except Exception:
                    pass
            self.risk.record_close(ticker, contracts)
            return

        if bid_id or ask_id:
            self._active_quotes[ticker] = (bid_id, ask_id)
            log.info(
                f"[mm] Quoted {ticker}  bid={our_bid}¢ ask={our_ask}¢ "
                f"capture={captured_c}¢ fees={fee_round_c:.1f}¢ "
                f"net_edge={edge_per_contract_dollars*100:.1f}¢/ct  x{contracts}"
            )

    def _cancel_quotes(self, ticker: str):
        bid_id, ask_id = self._active_quotes.pop(ticker, (None, None))
        for oid in (bid_id, ask_id):
            if oid:
                try:
                    self.kalshi.cancel_order(oid)
                except Exception:
                    pass
        # Use actual held contracts from risk manager (source of truth)
        held = self.risk.open_positions.get(ticker, 0)
        self.risk.record_close(ticker, held)

    def cancel_all(self):
        for ticker in list(self._active_quotes.keys()):
            self._cancel_quotes(ticker)
        # Also cancel resting legs from half-fills
        for hf in self._half_fills.values():
            try:
                self.kalshi.cancel_order(hf.resting_order_id)
                self.risk.record_close(hf.ticker, hf.contracts)
            except Exception:
                pass
        self._half_fills.clear()
