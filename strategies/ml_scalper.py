"""
ML Scalper Strategy — REGISTRY plug-in
─────────────────────────────────────────────────────────────────────────────
Bridges the ML probability engine into the existing BaseStrategy interface.
Works identically to any other strategy: receives OHLCV bars dict, returns
TradeSetup or None.  Can be selected via:

  python scalper.py --strategy ml_scalper
  python scalper.py --auto-strategy   (if ML ranks highest)

ATR-based dynamic SL/TP:
  SL = ATR(14, entry_tf) × ATR_SL_MULTIPLIER   (env: ML_ATR_SL_MULT, default 0.5)
  TP = SL × RR_RATIO                            (env: ML_RR_RATIO,    default 2.0)

Minimum SL floor:
  SL is clamped to at least spread × 1.5 so the trade is never immediately
  stopped out by the spread alone.

This file intentionally has no hardcoded instrument logic.  All thresholds,
timeframes, and parameters come from InstrumentConfig.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd

import config
from strategies.base import BaseStrategy, StrategySignal, TradeSetup


# Read ATR multipliers from env so they can be overridden without code changes
_ATR_SL_MULT: float = float(os.getenv("ML_ATR_SL_MULT", "0.5"))
_RR_RATIO:    float = float(os.getenv("ML_RR_RATIO",    "2.0"))


class MLScalper(BaseStrategy):
    """
    ML-driven strategy.  Probability engine decides signal; ATR drives SL/TP.

    Requirements:
      1. A trained model must exist in models/<SYMBOL>_BUY_*.pkl
      2. Candles for the entry TF and context TFs must be available
         (they come from LiveStrategyRunner via the standard bars dict)
    """

    name        = "ml_scalper"
    version     = "1.0"
    asset       = "multi"   # instrument-agnostic
    description = (
        "ML probability engine (XGBoost/LightGBM). "
        "Signal: P(TP before SL) >= per-instrument threshold. "
        "ATR-dynamic SL/TP."
    )

    def __init__(self) -> None:
        # Lazy imports: heavy modules loaded only when the strategy is actually used
        self._decision_engine = None
        self._instrument      = None
        self._last_signal_ts  = 0.0
        self._cooldown_sec    = 60.0   # minimum seconds between signals on same bar

    # ── Lazy initialiser ──────────────────────────────────────────────────────

    def _ensure_init(self, symbol: str | None = None) -> None:
        sym = (symbol or config.SYMBOL).upper()
        if self._instrument is not None and self._instrument.symbol == sym:
            return
        from ml.instrument_config import get_instrument
        from ml.decision_engine import DecisionEngine
        self._instrument      = get_instrument(sym)
        self._decision_engine = DecisionEngine(self._instrument)

    # ── BaseStrategy interface ────────────────────────────────────────────────

    def evaluate(self, bars: dict[str, pd.DataFrame]) -> Optional[TradeSetup]:
        """
        bars : multi-TF OHLCV dict, e.g. {"M5": df, "M15": df, "H1": df}
        Returns TradeSetup if ML probability exceeds threshold, else None.
        """
        self._ensure_init()

        entry_tf = self._instrument.entry_tf
        df = bars.get(entry_tf)
        if df is None or len(df) < 50:
            return None

        # ── Build feature vector for the latest completed bar ─────────────────
        from ml.feature_engine import build_feature_vector
        ts_utc = int(df.index[-1].timestamp())

        # Cooldown: don't re-evaluate the same bar multiple times
        if ts_utc == self._last_signal_ts:
            return None
        self._last_signal_ts = float(ts_utc)

        features = build_feature_vector(bars, entry_tf, ts_utc)
        if not features:
            return None

        # ── Get decision ──────────────────────────────────────────────────────
        decision = self._decision_engine.decide(features)
        if decision.signal.value == "WAIT":
            return None

        # ── ATR-based SL/TP ───────────────────────────────────────────────────
        atr_val = self._current_atr(df)
        spread  = self._instrument.spread_typical
        min_sl  = max(spread * 1.5, 0.1)   # never less than 1.5× spread

        sl_points = max(round(atr_val * _ATR_SL_MULT, 5), min_sl)
        tp_points = round(sl_points * _RR_RATIO, 5)

        close = float(df["close"].iloc[-1])
        if decision.signal.value == "BUY":
            entry = close
            sl    = round(entry - sl_points, 5)
            tp    = round(entry + tp_points, 5)
        else:
            entry = close
            sl    = round(entry + sl_points, 5)
            tp    = round(entry - tp_points, 5)

        rr = round(tp_points / sl_points, 2) if sl_points > 0 else _RR_RATIO

        return TradeSetup(
            signal=StrategySignal(decision.signal.value),
            entry=entry,
            sl=sl,
            tp=tp,
            rr=rr,
            reason=decision.explanation,
        )

    def _current_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """ATR of the last `period` bars."""
        if len(df) < period + 2:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1])
        atr_series = self.atr(df, period)
        val = atr_series.iloc[-1]
        return float(val) if pd.notna(val) and val > 0 else 1.0

    def debug_snapshot(self, bars: dict[str, pd.DataFrame]) -> dict:
        """Dashboard-compatible snapshot."""
        entry_tf = getattr(self._instrument, "entry_tf", "M5") if self._instrument else "M5"
        df = bars.get(entry_tf)
        if df is None or not self._decision_engine:
            return {"signal": "WAIT", "probability": 0.0, "model": "no_model"}
        from ml.feature_engine import build_feature_vector
        ts_utc = int(df.index[-1].timestamp()) if len(df) > 0 else 0
        features = build_feature_vector(bars, entry_tf, ts_utc)
        if not features:
            return {"signal": "WAIT", "probability": 0.0, "model": "no_model"}
        d = self._decision_engine.decide(features)
        return {
            "signal":      d.signal.value,
            "probability": d.probability,
            "explanation": d.explanation,
            "model":       self._decision_engine.model_version,
        }
