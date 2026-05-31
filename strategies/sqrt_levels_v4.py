"""
SQRT Levels v4 — Trail Machine
─────────────────────────────────────────────────────────────────────────────
Backtested: 65% win rate | Profit Factor: 15.6 | 20 trades (limited sample)
Asset:      XAUUSD
Timeframe:  Daily levels + 1H entry

IMPORTANT: Only 20 trades in backtest. Promising results but verify with
more data before going live. Run backtest over more periods first.

Concept:
  Price naturally respects square-root-based levels calculated from the
  daily open. Formula: level_n = (sqrt(daily_open) + n)^2
  These levels are ~$35-40 apart on Gold and act as S/R.
  System enters near a level with MACD + RSI + volume confirmation.
  No fixed TP — trailing stop captures the full move.

Entry Logic:
  1. Daily: Close > EMA50 (direction filter)
  2. Price is within 0.3% of a SQRT level
  3. Candle body > 50% of range (conviction candle)
  4. MACD histogram is accelerating in signal direction
  5. RSI is in 35–68 range (not overbought/oversold)
  6. Volume > 1.2x 20-bar average
  7. Session: 08:00–20:00 UTC

Stop Loss:  Trailing — 3 SQRT levels below price (moves up, never down)
Take Profit: None — trail rides the full move. Average: 11+ levels ($400+)

Source: ZeroOne D.O.T.S AI — strategy-dotsai-cloud
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import math
from typing import Optional, List
import pandas as pd
from strategies.base import BaseStrategy, StrategySignal, TradeSetup


class SQRTLevelsV4(BaseStrategy):

    name        = "sqrt_levels_v4"
    version     = "4.0"
    asset       = "XAUUSD"
    description = "SQRT Levels Trail Machine. PF 15.6, WR 65%. 20 trades — verify with more data."

    LEVEL_PROXIMITY_PCT  = 0.003   # enter if within 0.3% of a SQRT level
    MIN_BODY_PCT         = 0.50    # candle body must be > 50% of range
    RSI_LOW              = 35
    RSI_HIGH             = 68
    VOL_MULTIPLIER       = 1.2     # volume must be 1.2x average
    VOL_LOOKBACK         = 20
    TRAIL_LEVELS         = 3       # SL sits 3 levels behind price
    SESSION_START        = 8
    SESSION_END          = 20
    MAX_CONCURRENT       = 3       # max open positions at once

    @staticmethod
    def calc_sqrt_levels(daily_open: float, count: int = 20) -> List[float]:
        """
        Calculate SQRT price levels from daily open.
        Formula: level_n = (sqrt(daily_open) + n)^2
        Also computes midpoints between main levels.
        """
        root = math.sqrt(daily_open)
        main_levels = [(root + n) ** 2 for n in range(-3, count)]
        # Add midpoints between consecutive levels
        all_levels = []
        for i in range(len(main_levels) - 1):
            all_levels.append(main_levels[i])
            all_levels.append((main_levels[i] + main_levels[i + 1]) / 2)
        all_levels.append(main_levels[-1])
        return sorted(all_levels)

    @staticmethod
    def nearest_level(price: float, levels: List[float]) -> tuple[float, float]:
        """Returns (nearest_level, distance_pct)."""
        nearest = min(levels, key=lambda l: abs(l - price))
        dist_pct = abs(nearest - price) / price
        return nearest, dist_pct

    def evaluate(self, bars: dict[str, pd.DataFrame]) -> Optional[TradeSetup]:
        df_d  = bars.get("D1")
        df_1h = bars.get("1H")

        if df_d is None or df_1h is None:
            return None
        if len(df_d) < 55 or len(df_1h) < 30:
            return None

        # ── Session filter ───────────────────────────────────────────────────
        if not self.in_session(df_1h.index[-1], self.SESSION_START, self.SESSION_END):
            return None

        # ── Daily EMA50 direction ─────────────────────────────────────────────
        ema_d = self.ema(df_d["close"], 50)
        daily_close = df_d["close"].iloc[-1]
        daily_long  = daily_close > ema_d.iloc[-1]
        daily_short = daily_close < ema_d.iloc[-1]

        # ── SQRT levels from today's daily open ───────────────────────────────
        daily_open = df_d["open"].iloc[-1]
        levels = self.calc_sqrt_levels(daily_open)

        curr_bar  = df_1h.iloc[-1]
        curr_price = curr_bar["close"]

        nearest_lvl, dist_pct = self.nearest_level(curr_price, levels)
        if dist_pct > self.LEVEL_PROXIMITY_PCT:
            return None   # not near a SQRT level

        # ── Candle body conviction ────────────────────────────────────────────
        body   = abs(curr_bar["close"] - curr_bar["open"])
        candle_range = curr_bar["high"] - curr_bar["low"]
        if candle_range < 1e-6 or body / candle_range < self.MIN_BODY_PCT:
            return None

        bullish_candle = curr_bar["close"] > curr_bar["open"]
        bearish_candle = curr_bar["close"] < curr_bar["open"]

        # ── MACD acceleration ─────────────────────────────────────────────────
        _, _, hist = self.macd(df_1h["close"])
        macd_accel_up   = hist.iloc[-1] > hist.iloc[-2] > 0   # accelerating up
        macd_accel_down = hist.iloc[-1] < hist.iloc[-2] < 0   # accelerating down

        # ── RSI filter ────────────────────────────────────────────────────────
        rsi_val = self.rsi(df_1h["close"]).iloc[-1]
        rsi_ok = self.RSI_LOW <= rsi_val <= self.RSI_HIGH

        if not rsi_ok:
            return None

        # ── Volume filter ─────────────────────────────────────────────────────
        vol_avg = df_1h["volume"].iloc[-self.VOL_LOOKBACK:].mean()
        if df_1h["volume"].iloc[-1] < vol_avg * self.VOL_MULTIPLIER:
            return None

        # ── SL = 3 SQRT levels below/above entry ─────────────────────────────
        level_idx = min(range(len(levels)), key=lambda i: abs(levels[i] - curr_price))

        # ── Long setup ────────────────────────────────────────────────────────
        if daily_long and bullish_candle and macd_accel_up:
            sl_level_idx = max(0, level_idx - self.TRAIL_LEVELS)
            sl = levels[sl_level_idx]
            sl_usd = curr_price - sl
            # No fixed TP — use trailing stop (trail_tp = current price + 11 levels as initial target)
            tp_idx = min(len(levels) - 1, level_idx + 11)
            tp = levels[tp_idx]  # initial target — bot trails dynamically

            return TradeSetup(
                signal=StrategySignal.BUY,
                entry=round(curr_price, 2),
                sl=round(sl, 2),
                tp=round(tp, 2),
                rr=round((tp - curr_price) / sl_usd, 1) if sl_usd > 0 else 0,
                reason=(
                    f"SQRT level {nearest_lvl:.2f} (dist {dist_pct*100:.2f}%). "
                    f"MACD accel up. RSI {rsi_val:.1f}. Trail SL at level [{level_idx-3}]."
                ),
            )

        # ── Short setup ───────────────────────────────────────────────────────
        if daily_short and bearish_candle and macd_accel_down:
            sl_level_idx = min(len(levels) - 1, level_idx + self.TRAIL_LEVELS)
            sl = levels[sl_level_idx]
            sl_usd = sl - curr_price
            tp_idx = max(0, level_idx - 11)
            tp = levels[tp_idx]

            return TradeSetup(
                signal=StrategySignal.SELL,
                entry=round(curr_price, 2),
                sl=round(sl, 2),
                tp=round(tp, 2),
                rr=round((curr_price - tp) / sl_usd, 1) if sl_usd > 0 else 0,
                reason=(
                    f"SQRT level {nearest_lvl:.2f} (dist {dist_pct*100:.2f}%). "
                    f"MACD accel down. RSI {rsi_val:.1f}. Trail SL at level [{level_idx+3}]."
                ),
            )

        return None
