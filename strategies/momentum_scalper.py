"""
Momentum Scalper — Backtest Adapter
─────────────────────────────────────────────────────────────────────────────
Bridges the live tick-momentum engine to the OHLCV backtest framework.

Each OHLCV bar is converted to a synthetic 'tick':
  mid    = bar close price
  bid    = close - BACKTEST_SPREAD / 2
  ask    = close + BACKTEST_SPREAD / 2

8 consecutive bars → 8-tick window → MomentumEngine.evaluate()

Trade timing:
  Entry : next-bar open (as simulated by BacktestEngine + slippage)
  SL/TP : fixed distance from setup entry price (config values)

Timeframe:
  Default "5M" — 8 bars × 5min = 40-minute momentum window.
  Set ENTRY_TF = "1M" for tighter 8-minute window (requires 1M data).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import config
from momentum import MomentumEngine, Signal
from strategies.base import BaseStrategy, StrategySignal, TradeSetup


# ── Minimal tick proxy (avoids importing TickFeed / httpx chain) ──────────────

@dataclass
class _BarTick:
    """Synthetic tick built from a single OHLCV bar close. Read-only after init."""
    bid:      float
    ask:      float
    time_ms:  int   = 0
    mid:      float = field(init=False)
    spread:   float = field(init=False)

    def __post_init__(self):
        self.mid    = (self.bid + self.ask) / 2
        self.spread = round(self.ask - self.bid, 5)


# ── Strategy ──────────────────────────────────────────────────────────────────

class MomentumScalper(BaseStrategy):
    """
    Backtest adapter for the live tick-momentum scalper.

    Uses OHLCV bar closes as synthetic ticks. Inherits all signal parameters
    from config (SL_POINTS, TP_POINTS, MOMENTUM_WINDOW, MIN_DIRECTION_PCT,
    MIN_MOVE_POINTS, MAX_SPREAD_POINTS).
    """

    name        = "momentum_scalper"
    version     = "1.0"
    asset       = "XAUUSD"
    description = (
        "Tick momentum scalper backtested on OHLCV bars. "
        "Signal: 75%+ of 8 consecutive bar-closes move same direction + 0.5pt total move."
    )

    ENTRY_TF         = "5M"       # primary timeframe key in bars dict
    BACKTEST_SPREAD  = 0.30       # constant spread applied to all synthetic ticks (points)
    FALLBACK_TFS     = ["1M", "1H"]

    def __init__(self):
        self._engine = MomentumEngine()

    def evaluate(self, bars: dict[str, pd.DataFrame]) -> Optional[TradeSetup]:
        # Resolve which timeframe to use
        df = bars.get(self.ENTRY_TF)
        if df is None:
            for tf in self.FALLBACK_TFS:
                df = bars.get(tf)
                if df is not None:
                    break
        if df is None or df.empty:
            return None

        needed = config.MOMENTUM_WINDOW + 1   # window + 1 for pair-wise deltas
        if len(df) < needed:
            return None

        recent_bars = df.iloc[-needed:]
        ticks = deque(self._build_ticks(recent_bars))

        signal = self._engine.evaluate(ticks)
        if signal == Signal.WAIT:
            return None

        close       = float(df.iloc[-1]["close"])
        half_spread = self.BACKTEST_SPREAD / 2

        if signal == Signal.BUY:
            entry = round(close + half_spread, 2)          # simulated ask
            sl    = round(entry - config.SL_POINTS, 2)
            tp    = round(entry + config.TP_POINTS, 2)
            sig   = StrategySignal.BUY
        else:
            entry = round(close - half_spread, 2)          # simulated bid
            sl    = round(entry + config.SL_POINTS, 2)
            tp    = round(entry - config.TP_POINTS, 2)
            sig   = StrategySignal.SELL

        # Spread-guard: reject if spread eats the entire SL (mirrors live BUG-5 fix)
        if self.BACKTEST_SPREAD >= config.SL_POINTS:
            return None

        rr = round(config.TP_POINTS / config.SL_POINTS, 1)
        return TradeSetup(
            signal = sig,
            entry  = entry,
            sl     = sl,
            tp     = tp,
            rr     = rr,
            reason = (
                f"Momentum {sig.value} | window={config.MOMENTUM_WINDOW} bars | "
                f"SL={config.SL_POINTS}pt TP={config.TP_POINTS}pt RR=1:{rr}"
            ),
        )

    def _build_ticks(self, df: pd.DataFrame) -> list:
        """One synthetic tick per bar, using bar close as mid price."""
        half = self.BACKTEST_SPREAD / 2
        return [
            _BarTick(
                bid=float(row["close"]) - half,
                ask=float(row["close"]) + half,
                time_ms=idx,
            )
            for idx, (_, row) in enumerate(df.iterrows())
        ]
