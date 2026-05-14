"""Risk management — position sizing, daily loss limits, exposure caps."""

import time
from dataclasses import dataclass, field
from typing import Optional
from utils.state import save_state, load_state


@dataclass
class RiskConfig:
    starting_balance: float = 500.0
    max_position_pct: float = 0.05      # 5% of balance on initial entry
    max_position_scale_pct: float = 0.12  # total exposure per ticker never exceeds this
    max_daily_loss_pct: float = 1.0
    max_open_positions: int = 999
    min_edge: float = 0.01


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.balance: float = cfg.starting_balance
        self.day_start_balance: float = cfg.starting_balance
        self._day_start_ts: float = time.time()
        self.halted: bool = False
        self.halt_reason: str = ""
        # Load persisted positions from last run
        state = load_state()
        self.open_positions: dict[str, int] = {
            k: v for k, v in state.get("open_positions", {}).items() if v > 0
        }
        if self.open_positions:
            from utils.logger import log
            log.info(f"[risk] Restored {len(self.open_positions)} open positions from state")

    # ── Balance updates ───────────────────────────────────────────────────────

    def update_balance(self, new_balance: float):
        self._reset_day_if_needed()
        self.balance = new_balance
        if self.cfg.max_daily_loss_pct < 1.0 and self._daily_loss() >= self.cfg.max_daily_loss_pct:
            self._halt(f"Daily loss limit hit ({self._daily_loss()*100:.1f}%)")

    def _daily_loss(self) -> float:
        if self.day_start_balance == 0:
            return 0.0
        return max(0.0, (self.day_start_balance - self.balance) / self.day_start_balance)

    def _reset_day_if_needed(self):
        now = time.time()
        # Reset at UTC midnight (86400s per day)
        if now - self._day_start_ts >= 86400:
            self.day_start_balance = self.balance
            self._day_start_ts = now
            self.halted = False
            self.halt_reason = ""

    def _halt(self, reason: str):
        self.halted = True
        self.halt_reason = reason

    # ── Trade approval ────────────────────────────────────────────────────────

    def approve_trade(
        self,
        ticker: str,
        yes_price_cents: int,  # 1-99
        contracts: int,
        edge_dollars: float,
    ) -> tuple[bool, str]:
        """Return (approved, reason). Modifies nothing."""
        if self.halted:
            return False, f"Bot halted: {self.halt_reason}"

        cost = (yes_price_cents / 100) * contracts
        existing_contracts = self.open_positions.get(ticker, 0)
        existing_cost = (yes_price_cents / 100) * existing_contracts
        total_cost = existing_cost + cost

        # Hard cap: total exposure on any ticker never exceeds max_position_scale_pct
        if total_cost > self.balance * self.cfg.max_position_scale_pct:
            return False, f"Would exceed {self.cfg.max_position_scale_pct*100:.0f}% max exposure on {ticker} (${total_cost:.2f})"

        # Initial entry: capped at max_position_pct
        if existing_contracts == 0 and cost > self.balance * self.cfg.max_position_pct:
            return False, f"Initial position too large (${cost:.2f} > {self.cfg.max_position_pct*100:.0f}% of balance)"

        if edge_dollars < self.cfg.min_edge:
            return False, f"Edge too small (${edge_dollars:.3f} < ${self.cfg.min_edge})"

        return True, "ok"

    def kelly_contracts(
        self,
        yes_price_cents: int,
        true_prob: float,
        max_contracts: int = 50,
    ) -> int:
        """
        Kelly criterion: f* = (bp - q) / b
        where b = odds (payout/cost), p = true_prob, q = 1 - p
        Returns conservative half-Kelly contract count.
        """
        price = yes_price_cents / 100
        if price <= 0 or price >= 1:
            return 0
        b = (1 - price) / price   # net odds on a $1 bet
        p = true_prob
        q = 1 - p
        f = (b * p - q) / b
        if f <= 0:
            return 0
        half_kelly = f * 0.5
        max_cost_per_contract = price
        max_spend = min(self.balance * half_kelly, self.balance * self.cfg.max_position_pct)
        contracts = int(max_spend / max_cost_per_contract)
        return max(0, min(contracts, max_contracts))

    def record_open(self, ticker: str, contracts: int):
        self.open_positions[ticker] = self.open_positions.get(ticker, 0) + contracts
        save_state(self.open_positions)

    def record_close(self, ticker: str, contracts: int):
        existing = self.open_positions.get(ticker, 0)
        remaining = existing - contracts
        if remaining <= 0:
            self.open_positions.pop(ticker, None)
        else:
            self.open_positions[ticker] = remaining
        save_state(self.open_positions)

    def status(self) -> dict:
        self._reset_day_if_needed()
        return {
            "balance": round(self.balance, 2),
            "day_start_balance": round(self.day_start_balance, 2),
            "daily_pnl": round(self.balance - self.day_start_balance, 2),
            "daily_loss_pct": round(self._daily_loss() * 100, 2),
            "open_positions": len(self.open_positions),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }
