"""Persistent bot state — survives restarts."""

import json
import os

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "bot_state.json")


def save_state(open_positions: dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"open_positions": open_positions}, f)
    except Exception as e:
        print(f"[state] save failed: {e}")


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"open_positions": {}}
