"""
Momentum Engine — decides BUY / SELL / WAIT based on recent ticks.

Signal logic (tunable via .env):
  1. Take last MOMENTUM_WINDOW ticks.
  2. Count tick-to-tick UP vs DOWN moves.
  3. Signal BUY  if >= MIN_DIRECTION_PCT fraction go UP   AND total move >= MIN_MOVE_POINTS.
  4. Signal SELL if >= MIN_DIRECTION_PCT fraction go DOWN AND total move >= MIN_MOVE_POINTS.
  5. Skip if spread > MAX_SPREAD_POINTS.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Tuple

import config
from tick_feed import Tick


class Signal(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


def _tick_moves(ticks: list) -> Tuple[int, int, float]:
    """Returns (up_count, down_count, total_abs_move) for consecutive tick pairs."""
    up = down = 0
    for a, b in zip(ticks, ticks[1:]):
        delta = b.mid - a.mid
        if delta > 0:
            up += 1
        elif delta < 0:
            down += 1
    total_move = abs(ticks[-1].mid - ticks[0].mid) if len(ticks) > 1 else 0.0
    return up, down, total_move


class MomentumEngine:

    def evaluate(self, ticks: deque[Tick]) -> Signal:
        window = config.MOMENTUM_WINDOW

        if len(ticks) < window + 1:
            return Signal.WAIT

        recent: list[Tick] = list(ticks)[-window - 1:]
        latest = recent[-1]

        if latest.spread > config.MAX_SPREAD_POINTS:
            return Signal.WAIT

        up, down, total_move = _tick_moves(recent)
        total_pairs = up + down

        if total_pairs == 0 or total_move < config.MIN_MOVE_POINTS:
            return Signal.WAIT

        up_pct   = up   / total_pairs
        down_pct = down / total_pairs

        if up_pct >= config.MIN_DIRECTION_PCT:
            return Signal.BUY
        if down_pct >= config.MIN_DIRECTION_PCT:
            return Signal.SELL

        return Signal.WAIT

    def debug_snapshot(self, ticks: deque[Tick]) -> dict:
        """Returns signal internals for the dashboard."""
        window = config.MOMENTUM_WINDOW
        if len(ticks) < 2:
            return {"up": 0, "down": 0, "move": 0.0, "spread": 0.0}

        recent = list(ticks)[-window - 1:]
        up, down, total_move = _tick_moves(recent)
        latest = recent[-1] if recent else None
        return {
            "up":     up,
            "down":   down,
            "move":   round(total_move, 3),
            "spread": round(latest.spread, 3) if latest else 0.0,
        }
