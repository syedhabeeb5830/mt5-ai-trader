"""
Backtest Engine
─────────────────────────────────────────────────────────────────────────────
Fetches historical OHLCV data from MT5 and runs a strategy against it.
Works with any strategy that extends BaseStrategy.
Data source:
  - MetaTrader5 Python package (Windows, MT5 must be running)
  - Falls back to CSV files in backtest/data/ if MT5 not available
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal, TradeSetup


# ── MT5 data fetcher ─────────────────────────────────────────────────────────

TIMEFRAME_MAP = {
    "1M":  1,
    "5M":  5,
    "15M": 15,
    "1H":  60,
    "4H":  240,
    "D1":  1440,
}

def fetch_bars(symbol: str, timeframe: str, bars: int = 5000) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV bars from MT5.
    Returns None if MT5 is not available — use load_csv() instead.
    """
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None

        tf_const = getattr(mt5, f"TIMEFRAME_M{TIMEFRAME_MAP[timeframe]}", None)
        if tf_const is None:
            # Handle D1 and H4 specially
            if timeframe == "D1":  tf_const = mt5.TIMEFRAME_D1
            elif timeframe == "4H": tf_const = mt5.TIMEFRAME_H4
            else: return None

        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    except Exception:
        return None


def load_csv(filepath: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data from a CSV file (fallback when MT5 not available)."""
    path = Path(filepath)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]]


# ── Trade simulation ──────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    entry_time:  datetime
    exit_time:   Optional[datetime]
    direction:   str
    entry:       float
    sl:          float
    tp:          float
    exit_price:  float = 0.0
    pnl:         float = 0.0
    pnl_r:       float = 0.0   # PnL in R (risk units)
    result:      str   = ""    # WIN | LOSS | TIMEOUT
    reason:      str   = ""


@dataclass
class BacktestResult:
    strategy_name:  str
    symbol:         str
    start_date:     str
    end_date:       str
    total_trades:   int   = 0
    winning_trades: int   = 0
    losing_trades:  int   = 0
    total_pnl:      float = 0.0
    max_drawdown:   float = 0.0
    win_rate:       float = 0.0
    profit_factor:  float = 0.0
    avg_win:        float = 0.0
    avg_loss:       float = 0.0
    largest_win:    float = 0.0
    largest_loss:   float = 0.0
    trades:         list  = field(default_factory=list)


# ── Core engine ───────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Simulates a strategy against historical data.

    Usage:
        engine = BacktestEngine(symbol="XAUUSD", volume=0.01)
        result = engine.run(strategy, bars_data, entry_tf="1H")
    """

    def __init__(
        self,
        symbol:       str   = "XAUUSD",
        volume:       float = 0.01,
        commission:   float = 0.10,   # per trade in account currency
        slippage_pts: float = 0.3,    # entry slippage in points
        initial_equity: float = 1000.0,
    ):
        self.symbol         = symbol
        self.volume         = volume
        self.commission     = commission
        self.slippage_pts   = slippage_pts
        self.initial_equity = initial_equity

    def run(
        self,
        strategy:  BaseStrategy,
        all_bars:  dict[str, pd.DataFrame],
        entry_tf:  str = "1H",
        warmup:    int = 60,
    ) -> BacktestResult:
        """
        all_bars: dict of timeframe -> full historical DataFrame
        entry_tf: the timeframe we iterate bar-by-bar for signals
        warmup:   bars to skip at start (let indicators stabilize)
        """
        entry_df = all_bars[entry_tf]
        n_bars   = len(entry_df)
        trades   = []
        equity   = self.initial_equity
        peak     = equity
        max_dd   = 0.0

        open_trade: Optional[BacktestTrade] = None

        for i in range(warmup, n_bars - 1):
            bar_time = entry_df.index[i]

            # ── Check if open trade hit SL or TP ────────────────────────────
            if open_trade is not None:
                next_bar = entry_df.iloc[i + 1]
                pnl, result = self._check_exit(open_trade, next_bar)
                if result:
                    open_trade.pnl        = pnl - self.commission
                    open_trade.result     = result
                    open_trade.exit_time  = entry_df.index[i + 1]
                    risk = abs(open_trade.entry - open_trade.sl)
                    open_trade.pnl_r = open_trade.pnl / (risk * self.volume * 100) if risk > 0 else 0
                    trades.append(open_trade)
                    equity += open_trade.pnl
                    peak    = max(peak, equity)
                    max_dd  = max(max_dd, peak - equity)
                    open_trade = None
                continue  # one trade at a time

            # ── Slice bars up to current bar ─────────────────────────────────
            sliced = {}
            for tf, df in all_bars.items():
                mask = df.index <= bar_time
                sliced[tf] = df[mask]

            if any(len(df) < 30 for df in sliced.values()):
                continue

            # ── Evaluate strategy ─────────────────────────────────────────────
            setup = strategy.evaluate(sliced)
            if setup is None or setup.signal == StrategySignal.WAIT:
                continue

            # ── Open trade ────────────────────────────────────────────────────
            entry_price = setup.entry + (
                self.slippage_pts if setup.signal == StrategySignal.BUY else -self.slippage_pts
            )
            open_trade = BacktestTrade(
                entry_time=bar_time,
                exit_time=None,
                direction=setup.signal.value,
                entry=entry_price,
                sl=setup.sl,
                tp=setup.tp,
                reason=setup.reason,
            )

        result = self._compile_result(strategy, entry_df, trades, max_dd)
        return result

    def _check_exit(self, trade: BacktestTrade, next_bar: pd.Series) -> tuple[float, str]:
        """Check if a trade exits on the next bar. Returns (pnl, result_str)."""
        multiplier = 1 if trade.direction == "BUY" else -1

        if trade.direction == "BUY":
            if next_bar["low"] <= trade.sl:
                pnl = (trade.sl - trade.entry) * self.volume * 100
                return pnl, "LOSS"
            if next_bar["high"] >= trade.tp:
                pnl = (trade.tp - trade.entry) * self.volume * 100
                return pnl, "WIN"
        else:
            if next_bar["high"] >= trade.sl:
                pnl = (trade.entry - trade.sl) * self.volume * 100
                return pnl, "LOSS"
            if next_bar["low"] <= trade.tp:
                pnl = (trade.entry - trade.tp) * self.volume * 100
                return pnl, "WIN"

        return 0.0, ""

    def _compile_result(
        self,
        strategy: BaseStrategy,
        entry_df:  pd.DataFrame,
        trades:    list,
        max_dd:    float,
    ) -> BacktestResult:
        wins   = [t for t in trades if t.result == "WIN"]
        losses = [t for t in trades if t.result == "LOSS"]

        gross_profit = sum(t.pnl for t in wins)
        gross_loss   = abs(sum(t.pnl for t in losses))

        result = BacktestResult(
            strategy_name  = strategy.name,
            symbol         = self.symbol,
            start_date     = str(entry_df.index[0].date()),
            end_date       = str(entry_df.index[-1].date()),
            total_trades   = len(trades),
            winning_trades = len(wins),
            losing_trades  = len(losses),
            total_pnl      = round(sum(t.pnl for t in trades), 2),
            max_drawdown   = round(max_dd, 2),
            win_rate       = round(len(wins) / len(trades) * 100, 1) if trades else 0,
            profit_factor  = round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0,
            avg_win        = round(sum(t.pnl for t in wins) / len(wins), 2) if wins else 0,
            avg_loss       = round(sum(t.pnl for t in losses) / len(losses), 2) if losses else 0,
            largest_win    = round(max((t.pnl for t in wins), default=0), 2),
            largest_loss   = round(min((t.pnl for t in losses), default=0), 2),
            trades         = trades,
        )
        return result
