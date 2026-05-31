"""
Base Strategy — all strategies inherit from this.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class StrategySignal(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


@dataclass
class TradeSetup:
    signal:    StrategySignal
    entry:     float
    sl:        float
    tp:        float
    rr:        float        # risk:reward
    reason:    str = ""     # human-readable reason for the signal


class BaseStrategy(ABC):
    """
    All strategies implement evaluate().
    Input:  multi-timeframe OHLCV DataFrames
    Output: TradeSetup or None
    """

    name:        str = "base"
    version:     str = "1.0"
    asset:       str = "XAUUSD"
    description: str = ""

    @abstractmethod
    def evaluate(self, bars: dict[str, pd.DataFrame]) -> Optional[TradeSetup]:
        """
        bars: dict of timeframe -> OHLCV DataFrame
              e.g. {"1H": df_1h, "4H": df_4h, "D1": df_daily}
        Each DataFrame has columns: open, high, low, close, volume
        Index is datetime.
        Returns TradeSetup if signal found, else None.
        """
        ...

    # ── Shared indicator helpers ─────────────────────────────────────────────

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def swing_low(df: pd.DataFrame, lookback: int = 5) -> float:
        return df["low"].iloc[-lookback:].min()

    @staticmethod
    def swing_high(df: pd.DataFrame, lookback: int = 5) -> float:
        return df["high"].iloc[-lookback:].max()

    @staticmethod
    def macd(series: pd.Series, fast=12, slow=26, signal=9):
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def in_session(ts, start_h: int, end_h: int) -> bool:
        """Check if timestamp is within trading session (UTC hours)."""
        h = ts.hour
        return start_h <= h < end_h

    @staticmethod
    def dynamic_rr(sl_pct: float) -> float:
        """Dynamic RR based on SL size (smaller SL = more aggressive target)."""
        if sl_pct < 0.002:   # SL < 0.2%
            return 5.0
        elif sl_pct < 0.004: # SL < 0.4%
            return 4.0
        else:
            return 3.0
