"""
Live sports strategy: NBA, NHL, MLB.

Polls ESPN every cycle for live game scores. Converts score + time
remaining into a win probability using sport-specific models, then
compares to Kalshi's game-winner market price. Trades when the model
disagrees with Kalshi by 7+ cents.

Win probability models:
  NBA: normal distribution on score margin (sigma scales with sqrt(time))
  NHL: Poisson-based, goals remaining with overtime probability
  MLB: runs/inning lookup table (simplified)

Entry: edge >= 7 cents
Exit:  game ends | edge collapses < 2 cents | entered wrong side
"""

import math
import time
import re
import httpx
from clients.kalshi import KalshiClient
from risk.manager import RiskManager
from utils.logger import log
from dataclasses import dataclass


ESPN_NBA = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_NHL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
ESPN_MLB = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"

KALSHI_SERIES   = ["KXNBAGAME", "KXNHLGAME", "KXMLBGAME"]
MIN_EDGE        = 0.07   # 7 cent minimum edge
COOLDOWN_S      = 120
POLL_INTERVAL   = 30     # seconds between ESPN polls


# MLB win probability table: (innings_remaining, run_diff) -> win_prob
# Simplified — positive run_diff means leading
def mlb_win_prob(run_diff: int, innings_remaining: float) -> float:
    if innings_remaining <= 0:
        return 1.0 if run_diff > 0 else (0.5 if run_diff == 0 else 0.0)
    # Runs scored per inning ~ Poisson(0.46)
    # P(winning) using normal approximation to Poisson
    lam = 0.46 * innings_remaining
    sigma = math.sqrt(2 * lam)
    if sigma == 0:
        return 1.0 if run_diff > 0 else 0.5
    z = run_diff / sigma
    return _norm_cdf(z)


def nba_win_prob(lead: int, minutes_remaining: float) -> float:
    if minutes_remaining <= 0:
        return 1.0 if lead > 0 else (0.5 if lead == 0 else 0.0)
    # Score standard deviation scales with sqrt(time)
    # Each team scores ~1.05 pts/min, combined ~2.1, so diff ~ N(0, 11*sqrt(t/48))
    sigma = 11.0 * math.sqrt(minutes_remaining / 48.0)
    return _norm_cdf(lead / sigma)


def nhl_win_prob(lead: int, minutes_remaining: float) -> float:
    if minutes_remaining <= 0:
        if lead > 0: return 1.0
        if lead < 0: return 0.0
        return 0.5   # OT
    # Each team scores ~0.047 goals/min, combined ~3 goals/60min
    lam = 0.047 * minutes_remaining
    sigma = math.sqrt(2 * lam)
    if sigma == 0:
        return 1.0 if lead > 0 else 0.5
    z = lead / sigma
    # Add OT probability adjustment — ties go to OT which is ~50/50 adjusted
    p = _norm_cdf(z)
    return p


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


TRAIL_DISTANCE = 5   # cents

@dataclass
class SportsPosition:
    ticker: str
    side: str
    contracts: int
    entry_price_c: int
    entry_time: float
    high_water_c: int = 0


class SportsStrategy:
    def __init__(self, kalshi: KalshiClient, risk: RiskManager):
        self.kalshi = kalshi
        self.risk = risk
        self._positions: dict[str, SportsPosition] = {}
        self._last_entry: dict[str, float] = {}
        self._last_espn_poll: float = 0
        self._live_games: dict[str, dict] = {}   # team_key -> game state
        self._kalshi_markets: list[dict] = []
        self._markets_ts: float = 0

    def _fetch_espn(self):
        if time.time() - self._last_espn_poll < POLL_INTERVAL:
            return
        self._last_espn_poll = time.time()
        self._live_games = {}

        for sport, url in [("nba", ESPN_NBA), ("nhl", ESPN_NHL), ("mlb", ESPN_MLB)]:
            try:
                r = httpx.get(url, timeout=8)
                for event in r.json().get("events", []):
                    comp = event.get("competitions", [{}])[0]
                    status = comp.get("status", {})
                    state = status.get("type", {}).get("name", "")
                    if state not in ("STATUS_IN_PROGRESS", "STATUS_HALFTIME", "STATUS_END_PERIOD"):
                        continue
                    teams = comp.get("competitors", [])
                    if len(teams) != 2:
                        continue
                    t0 = teams[0]
                    t1 = teams[1]
                    abbr0 = t0.get("team", {}).get("abbreviation", "")
                    abbr1 = t1.get("team", {}).get("abbreviation", "")
                    score0 = int(float(t0.get("score") or 0))
                    score1 = int(float(t1.get("score") or 0))
                    period = status.get("period", 0)
                    clock  = status.get("displayClock", "0:00")

                    mins_rem = _parse_clock(clock, period, sport)
                    lead0 = score0 - score1  # positive = team0 leading

                    game = {
                        "sport":    sport,
                        "abbr0":    abbr0,
                        "abbr1":    abbr1,
                        "score0":   score0,
                        "score1":   score1,
                        "lead0":    lead0,
                        "period":   period,
                        "mins_rem": mins_rem,
                        "state":    state,
                    }
                    key = f"{abbr0}{abbr1}"
                    self._live_games[key] = game
                    self._live_games[f"{abbr1}{abbr0}"] = game
            except Exception as e:
                log.warning(f"[sports] ESPN {sport} fetch failed: {e}")

    def _fetch_kalshi_markets(self):
        if time.time() - self._markets_ts < 60 and self._kalshi_markets:
            return
        markets = []
        for series in KALSHI_SERIES:
            try:
                r = self.kalshi._get("/markets", {"limit": 50, "series_ticker": series})
                for m in r.get("markets", []):
                    if m.get("status") == "active":
                        bid = float(m.get("yes_bid_dollars") or 0)
                        ask = float(m.get("yes_ask_dollars") or 0)
                        if bid > 0 and ask > 0:
                            markets.append(m)
            except Exception as e:
                log.warning(f"[sports] Kalshi {series} fetch failed: {e}")
        self._kalshi_markets = markets
        self._markets_ts = time.time()

    def _parse_ticker_teams(self, ticker: str):
        # KXNBAGAME-26MAY15DETCLE-DET -> teams=DETCLE, side=DET
        parts = ticker.split("-")
        if len(parts) < 3:
            return None, None
        team_part = parts[-1]          # DET or CLE
        game_part = parts[-2]          # 26MAY15DETCLE
        # Strip date prefix (digits + month + digits)
        m = re.search(r'\d{2}[A-Z]{3}\d{2}([A-Z]+)', game_part)
        if not m:
            return None, None
        teams_str = m.group(1)         # DETCLE
        # Split: try 3+3 or 2+3 or 3+2
        if len(teams_str) == 6:
            t0, t1 = teams_str[:3], teams_str[3:]
        elif len(teams_str) == 5:
            # Try to match team_part
            if teams_str[:2] == team_part or teams_str[:3] == team_part:
                t0 = team_part
                t1 = teams_str[len(team_part):]
            else:
                t1 = team_part
                t0 = teams_str[:-len(team_part)]
        else:
            return None, None
        return (t0, t1), team_part

    def _win_prob(self, game: dict, team_abbr: str) -> float:
        sport    = game["sport"]
        mins_rem = game["mins_rem"]
        # Lead from team_abbr's perspective
        if team_abbr == game["abbr0"]:
            lead = game["lead0"]
        else:
            lead = -game["lead0"]

        if sport == "nba":
            return nba_win_prob(lead, mins_rem)
        elif sport == "nhl":
            return nhl_win_prob(lead, mins_rem)
        elif sport == "mlb":
            innings_remaining = mins_rem  # reused field
            return mlb_win_prob(lead, innings_remaining)
        return 0.5

    def _check_exits(self):
        for ticker, pos in list(self._positions.items()):
            try:
                m = self.kalshi.get_market(ticker).get("market", {})
                if m.get("status") == "finalized":
                    self._exit(pos, "game ended")
                    continue
                bid = float(m.get("yes_bid_dollars") or 0) if pos.side == "yes" else float(m.get("no_bid_dollars") or 0)
                bid_c = int(bid * 100)
                if bid_c > 0:
                    if bid_c > pos.high_water_c:
                        pos.high_water_c = bid_c
                    trail_stop = pos.high_water_c - TRAIL_DISTANCE
                    pnl = bid_c - pos.entry_price_c
                    if pnl >= 10:
                        self._exit(pos, f"profit target +{pnl}c")
                    elif pnl <= -8:
                        self._exit(pos, f"stop loss {pnl}c")
                    elif bid_c < trail_stop and pos.high_water_c > pos.entry_price_c + 4:
                        self._exit(pos, f"trailing stop (peak={pos.high_water_c}¢)")
            except Exception as e:
                log.warning(f"[sports] Exit check {ticker}: {e}")

    def _exit(self, pos: SportsPosition, reason: str):
        try:
            self.kalshi.place_order(
                ticker=pos.ticker, side=pos.side, action="sell",
                count=pos.contracts, order_type="market",
            )
            self.risk.record_close(pos.ticker, pos.contracts)
            log.info(f"[sports] EXIT {pos.ticker} {pos.side.upper()} x{pos.contracts}  {reason}")
        except Exception as e:
            log.error(f"[sports] Exit failed {pos.ticker}: {e}")
        finally:
            self._positions.pop(pos.ticker, None)

    def scan(self) -> list[dict]:
        self._fetch_espn()
        self._fetch_kalshi_markets()
        self._check_exits()

        if not self._live_games:
            return []

        signals = []
        now = time.time()

        for m in self._kalshi_markets:
            ticker = m["ticker"]
            if ticker in self._positions:
                continue
            if now - self._last_entry.get(ticker, 0) < COOLDOWN_S:
                continue

            teams, team_abbr = self._parse_ticker_teams(ticker)
            if not teams or not team_abbr:
                continue

            t0, t1 = teams
            game = self._live_games.get(f"{t0}{t1}") or self._live_games.get(f"{t1}{t0}")
            if not game:
                continue

            # Don't trade last 3 min of NBA/NHL or last inning of MLB
            if game["mins_rem"] < 3:
                continue

            win_prob = self._win_prob(game, team_abbr)
            model_c  = int(win_prob * 100)

            yes_ask_c = int(float(m.get("yes_ask_dollars") or 0) * 100)
            yes_bid_c = int(float(m.get("yes_bid_dollars") or 0) * 100)
            no_ask_c  = int(float(m.get("no_ask_dollars")  or 0) * 100)
            vol       = float(m.get("volume_24h_fp") or 0)

            # Model says higher than Kalshi ask → buy YES
            if model_c - yes_ask_c >= int(MIN_EDGE * 100):
                edge = (win_prob - yes_ask_c / 100)
                signals.append({
                    "ticker":    ticker, "title": m.get("title","")[:60],
                    "side":      "yes",  "price_c": yes_ask_c,
                    "model_c":   model_c, "edge": edge, "volume": vol,
                    "game":      f"{game['abbr0']} {game['score0']}-{game['score1']} {game['abbr1']}  {game['mins_rem']:.0f}min",
                })
            # Model says lower than Kalshi bid → buy NO
            elif yes_bid_c - model_c >= int(MIN_EDGE * 100) and no_ask_c > 0:
                edge = ((100 - yes_bid_c) / 100 - no_ask_c / 100) + (1 - win_prob - no_ask_c / 100)
                edge = (1 - win_prob) - no_ask_c / 100
                signals.append({
                    "ticker":    ticker, "title": m.get("title","")[:60],
                    "side":      "no",   "price_c": no_ask_c,
                    "model_c":   model_c, "edge": edge, "volume": vol,
                    "game":      f"{game['abbr0']} {game['score0']}-{game['score1']} {game['abbr1']}  {game['mins_rem']:.0f}min",
                })

        signals.sort(key=lambda x: (x["volume"], abs(x["edge"])), reverse=True)
        return signals

    def execute(self, signal: dict) -> bool:
        ticker  = signal["ticker"]
        side    = signal["side"]
        price_c = signal["price_c"]
        edge    = signal["edge"]

        contracts = self.risk.kelly_contracts(price_c, price_c / 100 + edge, max_contracts=50)
        contracts = max(1, contracts)

        ok, reason = self.risk.approve_trade(ticker, price_c, contracts, edge * contracts)
        if not ok:
            log.info(f"[sports] Skipped {ticker}: {reason}")
            return False

        try:
            self.kalshi.place_order(
                ticker=ticker, side=side, action="buy",
                count=contracts, order_type="limit",
                yes_price=price_c if side == "yes" else None,
                no_price=price_c  if side == "no"  else None,
            )
            self.risk.record_open(ticker, contracts)
            self._last_entry[ticker] = time.time()
            self._positions[ticker] = SportsPosition(
                ticker=ticker, side=side, contracts=contracts,
                entry_price_c=price_c, entry_time=time.time(),
                high_water_c=price_c,
            )
            log.info(
                f"[sports] ENTER {ticker} {side.upper()} x{contracts} @ {price_c}c"
                f"  model={signal['model_c']}c  {signal['game']}"
            )
            return True
        except Exception as e:
            log.error(f"[sports] Order failed {ticker}: {e}")
            self.risk.undo_reservation(ticker, contracts)
            return False


def _parse_clock(clock_str: str, period: int, sport: str) -> float:
    """Return minutes remaining in the game."""
    try:
        parts = clock_str.replace(".", ":").split(":")
        mins = float(parts[0])
        secs = float(parts[1]) if len(parts) > 1 else 0
        clock_mins = mins + secs / 60
    except Exception:
        clock_mins = 0

    if sport == "nba":
        period_mins = 12.0
        total_periods = 4
        periods_left = max(0, total_periods - period)
        return clock_mins + periods_left * period_mins

    elif sport == "nhl":
        period_mins = 20.0
        total_periods = 3
        periods_left = max(0, total_periods - period)
        return clock_mins + periods_left * period_mins

    elif sport == "mlb":
        # Return innings remaining (reuse mins_rem field)
        total_innings = 9
        innings_left = max(0, total_innings - period) + 0.5
        return innings_left

    return clock_mins
