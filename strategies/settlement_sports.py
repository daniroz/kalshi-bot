"""
Verified Sports Settlement — late-game outcomes with ≥99.5% win probability.

Thesis: When a game is in its final minutes with an overwhelming lead, the
outcome is mathematically nearly-locked. ESPN provides live scores and time
remaining. We compute win probability from lead + time-remaining and only
trade when the result is ≥99.5% — and the market still prices it ≤92¢.

This is different from the existing sports.py strategy (which is now off):
  - sports.py: lots of trades, 7¢ edge, models small leads → high error rate
  - settlement_sports.py: very few trades, only "the game is essentially over"

Resolution check: only act on tickers dated TODAY (carry-over from the
date-mismatch bug we already fixed in sports.py — a live Game 2 cannot price
the future Game 4 in the same series).

Position size: $20–$40. Single-shot, no exit logic — the game ends, the
market resolves, we get paid.
"""

import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.markets import get_liquid_markets
from utils.logger import log

# Reuse the existing sports module's ESPN integration + win-prob math.
from strategies.sports import (
    SportsStrategy,
    nba_win_prob, nhl_win_prob, mlb_win_prob,
)


# ── Tunables ───────────────────────────────────────────────────────────────────
SCAN_INTERVAL_S      = 20
MIN_EDGE_C           = 10
MIN_TRADE_DOLLARS    = 5.0        # honors global risk-manager floor
MAX_TRADE_DOLLARS    = 35.0
MAX_PRICE_C          = 89         # ≥10¢ edge → 90+ unreachable; matches math
MIN_PRICE_C          = 20         # MARKET-SANITY: ESPN can lag a buzzer-beater; respect order book
MIN_WIN_PROB         = 0.995      # very high bar — the game must be essentially over

# Minimum lead in absolute units for each sport before we'll even check win_prob.
# This is a sanity gate; the win_prob threshold is the real filter.
MIN_LEAD_BY_SPORT = {"nba": 8, "nhl": 2, "mlb": 3}

# Maximum time-remaining in each sport — we only trade *late* in the game.
# Less = more conservative.
MAX_TIME_REMAINING = {
    "nba": 6.0,       # minutes
    "nhl": 6.0,       # minutes
    "mlb": 3.0,       # innings (mins_rem is reused as innings_rem for MLB)
}


@dataclass
class SportsSettlePosition:
    ticker:        str
    side:          str
    contracts:     int
    entry_price_c: int
    win_prob:      float


def _kalshi_fee_cents(price_c: int, contracts: int) -> float:
    p = price_c / 100
    return 0.07 * p * (1 - p) * 100 * contracts


# ── Strategy ───────────────────────────────────────────────────────────────────
class SportsSettlementStrategy:
    """
    Borrows the existing SportsStrategy for its ESPN polling and ticker parsing,
    but applies a much stricter win-probability filter and verifies the date.
    """

    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk   = risk
        # Use a dedicated SportsStrategy instance just for its data plumbing.
        # We never call its scan/execute — only its _fetch_espn + _parse_ticker_teams.
        self._helper = SportsStrategy(kalshi, risk)
        self._last_scan = 0.0
        self._positions: dict[str, SportsSettlePosition] = {}
        self._lock = threading.Lock()

    def _win_prob(self, game: dict, team_abbr: str) -> float:
        """Win probability for `team_abbr` given live game state."""
        sport    = game["sport"]
        mins_rem = game["mins_rem"]
        lead     = game["lead0"] if team_abbr == game["abbr0"] else -game["lead0"]
        if sport == "nba":  return nba_win_prob(lead, mins_rem)
        if sport == "nhl":  return nhl_win_prob(lead, mins_rem)
        if sport == "mlb":  return mlb_win_prob(lead, mins_rem)
        return 0.5

    def _evaluate(self, m: dict) -> Optional[dict]:
        ticker = m["ticker"]
        teams, team_abbr, ticker_date = self._helper._parse_ticker_teams(ticker)
        if not teams or not team_abbr:
            return None
        if ticker_date != self._helper._today_kalshi_date():
            return None

        t0, t1 = teams
        game = self._helper._live_games.get(f"{t0}{t1}") or \
               self._helper._live_games.get(f"{t1}{t0}")
        if not game:
            return None

        sport = game["sport"]
        mins_rem = game["mins_rem"]
        if mins_rem > MAX_TIME_REMAINING.get(sport, 0):
            return None

        # Sanity lead gate before computing prob
        my_lead = game["lead0"] if team_abbr == game["abbr0"] else -game["lead0"]
        min_lead = MIN_LEAD_BY_SPORT.get(sport, 99)

        yes_ask_c = int(round(float(m.get("yes_ask_dollars") or 0) * 100))
        no_ask_c  = int(round(float(m.get("no_ask_dollars")  or 0) * 100))
        if yes_ask_c == 0 or no_ask_c == 0:
            return None

        win_prob = self._win_prob(game, team_abbr)

        side = price_c = reason = None

        # Team is winning — buy YES on this team if it's cheap enough
        if my_lead >= min_lead and win_prob >= MIN_WIN_PROB and MIN_PRICE_C <= yes_ask_c <= MAX_PRICE_C:
            side, price_c = "yes", yes_ask_c
            reason = (f"{team_abbr} leads {my_lead} ({sport} {mins_rem:.1f}rem) "
                      f"win_prob={win_prob*100:.2f}%")
        elif my_lead >= min_lead and win_prob >= MIN_WIN_PROB and yes_ask_c < MIN_PRICE_C:
            log.warning(f"[settle-sports] SKIP {ticker}: {team_abbr} leads {my_lead} "
                        f"(win_prob={win_prob*100:.2f}%) but market prices YES at "
                        f"{yes_ask_c}¢ — possible score correction or ESPN lag.")
            return None
        # Team is losing badly — buy NO (= bet the OTHER team wins)
        elif my_lead <= -min_lead and (1 - win_prob) >= MIN_WIN_PROB and MIN_PRICE_C <= no_ask_c <= MAX_PRICE_C:
            side, price_c = "no", no_ask_c
            reason = (f"{team_abbr} trails {-my_lead} ({sport} {mins_rem:.1f}rem) "
                      f"loss_prob={(1-win_prob)*100:.2f}%")
        elif my_lead <= -min_lead and (1 - win_prob) >= MIN_WIN_PROB and no_ask_c < MIN_PRICE_C:
            log.warning(f"[settle-sports] SKIP {ticker}: {team_abbr} trails {-my_lead} "
                        f"but market prices NO at {no_ask_c}¢ — possible score correction or lag.")
            return None

        if side is None:
            return None

        edge_c = (100 - price_c) - _kalshi_fee_cents(price_c, 1)
        if edge_c < MIN_EDGE_C:
            return None

        return {
            "ticker":   ticker,
            "side":     side,
            "price_c":  price_c,
            "edge_c":   edge_c,
            "win_prob": win_prob if side == "yes" else (1 - win_prob),
            "reason":   reason,
        }

    def scan(self) -> list[dict]:
        now = time.time()
        with self._lock:
            if now - self._last_scan < SCAN_INTERVAL_S:
                return []
            self._last_scan = now

        # Refresh ESPN game state via the helper's method
        try:
            self._helper._fetch_espn()
        except Exception as e:
            log.debug(f"[settle-sports] ESPN poll failed: {e}")
            return []
        if not self._helper._live_games:
            return []

        markets = [m for m in get_liquid_markets(self.kalshi, min_volume=0)
                   if m["ticker"].startswith(("KXNBAGAME", "KXNHLGAME", "KXMLBGAME"))]

        opps = []
        for m in markets:
            ticker = m["ticker"]
            if ticker in self._positions:
                continue
            if self.risk.open_positions.get(ticker, 0) > 0:
                continue
            try:
                opp = self._evaluate(m)
                if opp:
                    opps.append(opp)
            except Exception as e:
                log.debug(f"[settle-sports] eval {ticker} failed: {e}")
        opps.sort(key=lambda x: x["edge_c"], reverse=True)
        return opps

    def execute(self, opp: dict) -> bool:
        ticker  = opp["ticker"]
        side    = opp["side"]
        price_c = opp["price_c"]
        edge_c  = opp["edge_c"]

        # Tier-aware sizing: target min(MAX_TRADE_DOLLARS, tier-adjusted cap).
        price = price_c / 100
        tier_cap_dollars = self.risk.balance * self.risk.effective_max_position_pct()
        target_dollars = min(MAX_TRADE_DOLLARS, tier_cap_dollars)
        if target_dollars < MIN_TRADE_DOLLARS:
            log.info(f"[settle-sports] Skip {ticker}: tier cap ${tier_cap_dollars:.2f} below floor ${MIN_TRADE_DOLLARS}")
            return False
        contracts = int(target_dollars / price)
        if contracts <= 0:
            return False

        with self._lock:
            if ticker in self._positions:
                return False
            self._positions[ticker] = None

        edge_dollars_total = (edge_c / 100) * contracts
        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge_dollars_total)
        if not ok:
            log.info(f"[settle-sports] Skip {ticker}: {reason}")
            self._positions.pop(ticker, None)
            return False

        try:
            self.kalshi.place_order(
                ticker=ticker, side=side, action="buy",
                count=contracts, order_type="limit",
                yes_price=price_c if side == "yes" else None,
                no_price =price_c if side == "no"  else None,
            )
            self.risk.record_open(ticker, contracts)
            self._positions[ticker] = SportsSettlePosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, win_prob=opp["win_prob"],
            )
            log.info(
                f"[settle-sports] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}¢  "
                f"edge={edge_c:.1f}¢  {opp['reason']}"
            )
            return True
        except Exception as e:
            log.error(f"[settle-sports] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            self._positions.pop(ticker, None)
            return False
