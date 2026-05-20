"""
config.yaml loader with env-var override support.

Precedence (highest wins):
  1. .env variable (legacy compat — STRATEGY_ARBITRAGE etc. still works)
  2. config.yaml value
  3. Hardcoded default

Usage:
    from utils.config import config

    if config.strategies.arbitrage.enabled:
        ...
    max_pos = config.risk.max_position_pct
    risk_cfg = config.risk_config_dict()   # for RiskConfig kwargs

The single global `config` object is constructed at import time. To reload
(e.g. after editing the yaml without restarting the bot), call config.reload().

Validation is intentionally light — we want bot-friendly defaults rather than
hard crashes on unknown keys, so missing values fall back to safe defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "pyyaml is required for config.yaml support. "
        "Install with: pip install pyyaml"
    ) from e


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


# ── Dot-access wrapper around a dict ────────────────────────────────────────
class _DotDict(dict):
    """Dict that supports attribute access, recursively."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        if data:
            for k, v in data.items():
                self[k] = _DotDict(v) if isinstance(v, dict) else v

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"config has no key '{name}'. "
                f"Available: {sorted(self.keys())[:10]}..."
            )

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Safe lookup by dotted path: config.get_path('risk.min_edge', 0.02)."""
        cur: Any = self
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur


# ── Env-var overrides ───────────────────────────────────────────────────────
# Map env var → (yaml dotted path, type cast). Only the .env keys that
# users actually flip on the fly belong here — secrets stay env-only.
ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "STARTING_BALANCE":           ("risk.starting_balance",         float),
    "MAX_POSITION_SIZE_PCT":      ("risk.max_position_pct",         float),
    "MAX_POSITION_SCALE_PCT":     ("risk.max_position_scale_pct",   float),
    "MAX_DAILY_LOSS_PCT":         ("risk.max_daily_loss_pct",       float),
    "MAX_OPEN_POSITIONS":         ("risk.max_open_positions",       int),
    "MIN_EDGE_THRESHOLD":         ("risk.min_edge",                 float),
    "DEMO_MODE":                  ("runtime.demo_mode",             lambda s: s.lower() == "true"),
    # Strategy enable flags — STRATEGY_X env vars from before the yaml existed
    "STRATEGY_ARBITRAGE":         ("strategies.arbitrage.enabled",         lambda s: s.lower() == "true"),
    "STRATEGY_MARKET_MAKER":      ("strategies.market_maker.enabled",      lambda s: s.lower() == "true"),
    "STRATEGY_MISPRICING":        ("strategies.mispricing.enabled",        lambda s: s.lower() == "true"),
    "STRATEGY_SMART_MONEY":       ("strategies.smart_money.enabled",       lambda s: s.lower() == "true"),
    "STRATEGY_WEATHER":           ("strategies.weather.enabled",           lambda s: s.lower() == "true"),
    "STRATEGY_SPORTS":            ("strategies.sports.enabled",            lambda s: s.lower() == "true"),
    "STRATEGY_CRYPTO":            ("strategies.crypto.enabled",            lambda s: s.lower() == "true"),
    "STRATEGY_INTRADAY":          ("strategies.intraday.enabled",          lambda s: s.lower() == "true"),
    "STRATEGY_ORDERBOOK":         ("strategies.orderbook.enabled",         lambda s: s.lower() == "true"),
    "STRATEGY_MOMENTUM":          ("strategies.momentum.enabled",          lambda s: s.lower() == "true"),
    "STRATEGY_NEWS":              ("strategies.news.enabled",              lambda s: s.lower() == "true"),
    "STRATEGY_CALENDAR":          ("strategies.calendar.enabled",          lambda s: s.lower() == "true"),
    "STRATEGY_SETTLEMENT":        ("strategies.settlement_weather.enabled",    lambda s: s.lower() == "true"),
    "STRATEGY_SETTLEMENT_CRYPTO": ("strategies.settlement_crypto.enabled", lambda s: s.lower() == "true"),
    "STRATEGY_SETTLEMENT_STOCKS": ("strategies.settlement_stocks.enabled", lambda s: s.lower() == "true"),
    "STRATEGY_SETTLEMENT_SPORTS": ("strategies.settlement_sports.enabled", lambda s: s.lower() == "true"),
    "STRATEGY_SETTLEMENT_COMM":   ("strategies.settlement_commodities.enabled", lambda s: s.lower() == "true"),
    "STRATEGY_SETTLEMENT_FX":     ("strategies.settlement_forex.enabled",  lambda s: s.lower() == "true"),
}


def _apply_env_overrides(data: _DotDict) -> _DotDict:
    """Apply known env-var overrides on top of yaml-loaded values."""
    for env_key, (dotted, cast) in ENV_OVERRIDES.items():
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        try:
            value = cast(raw)
        except (TypeError, ValueError):
            continue
        # Walk to the parent and set
        parts = dotted.split(".")
        cur: Any = data
        for part in parts[:-1]:
            if part not in cur:
                cur[part] = _DotDict()
            cur = cur[part]
        cur[parts[-1]] = value
    return data


# ── Main config object ──────────────────────────────────────────────────────
class _Config(_DotDict):
    """The single config singleton. Attribute access throughout."""

    def __init__(self):
        super().__init__()
        self.reload()

    def reload(self) -> None:
        """Re-read config.yaml + env overrides. Safe to call at runtime."""
        self.clear()
        loaded: dict = {}
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH) as f:
                    loaded = yaml.safe_load(f) or {}
        except Exception as e:
            # Don't crash the bot if yaml is malformed — log and fall back to env-only.
            print(f"[config] WARNING: failed to load {CONFIG_PATH}: {e}")
            loaded = {}

        # Populate self
        for k, v in loaded.items():
            self[k] = _DotDict(v) if isinstance(v, dict) else v

        _apply_env_overrides(self)

    # ── Convenience accessors used by the rest of the codebase ─────────────

    def strategy_enabled(self, name: str) -> bool:
        """Is `name` (e.g. 'arbitrage', 'settlement_crypto') turned on?"""
        block = self.get_path(f"strategies.{name}")
        if not block:
            return False
        return bool(block.get("enabled", False))

    def strategy(self, name: str) -> _DotDict:
        """Get the full block for a strategy. Empty DotDict if missing."""
        return self.get_path(f"strategies.{name}", _DotDict())

    def risk_config_dict(self) -> dict:
        """RiskConfig kwargs dict, ready to splat into RiskConfig(**...)."""
        r = self.get("risk", _DotDict())
        return {
            "starting_balance":       float(r.get("starting_balance", 250.0)),
            "max_position_pct":       float(r.get("max_position_pct", 0.05)),
            "max_position_scale_pct": float(r.get("max_position_scale_pct", 0.12)),
            "max_daily_loss_pct":     float(r.get("max_daily_loss_pct", 1.0)),
            "max_open_positions":     int(r.get("max_open_positions", 999)),
            "min_edge":               float(r.get("min_edge", 0.02)),
        }


# Singleton — imported throughout the codebase
config = _Config()
