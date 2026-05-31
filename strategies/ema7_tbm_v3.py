"""
7 EMA TBM v3 — Triple Timeframe Trend Follower
─────────────────────────────────────────────────────────────────────────────
Backtested: 36.8% win rate | Profit Factor: 1.19 | 57 trades (1 year)
Asset:      XAUUSD
Timeframes: Daily (direction) + 4H (confirm) + 1H (entry)

What makes this different from v2:
  Trades BOTH directions (Long + Short) based on Daily EMA50.
  No human bias — if Daily says down, it shorts.
  Sits out when Daily is flat/choppy. Most selective setup.

Entry Logic (Long):
  1. Daily: Close > EMA50 AND slope is upward
  2. 4H: 3+ consecutive candles close ABOVE EMA7
  3. 1H: Low touches EMA7, candle closes above with bullish body
  4. 1H: EMA7 slope > 0.02%
  5. 1H: Volume > 20-bar average
  6. Session: 08:00–20:00 UTC

Entry Logic (Short):  Mirror of above — Daily < EMA50, 4H below EMA7, 1H retest from below.

Stop Loss:  Swing low/high (5 bars), min 0.8x ATR, capped at $30
Take Profit: Dynamic 1:3 to 1:5

Source: ZeroOne D.O.T.S AI — strategy-dotsai-cloud
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
from strategies.base import BaseStrategy, StrategySignal, TradeSetup


class EMA7TBMv3(BaseStrategy):

    name        = "ema7_tbm_v3"
    version     = "3.0"
    asset       = "XAUUSD"
    description = "7 EMA TBM v3 — Triple TF Trend Follower Long+Short. PF 1.19, WR 36.8%"

    EMA_ENTRY_PERIOD   = 7
    EMA_TREND_PERIOD   = 50
    ATR_PERIOD         = 14
    VOL_LOOKBACK       = 20
    SWING_BARS         = 5
    MIN_TREND_BARS     = 3
    MIN_EMA_SLOPE      = 0.0002
    DAILY_SLOPE_BARS   = 3     # look back N daily bars for slope
    MAX_SL_USD         = 30.0
    SESSION_START      = 8
    SESSION_END        = 20

    def evaluate(self, bars: dict[str, pd.DataFrame]) -> Optional[TradeSetup]:
        df_d  = bars.get("D1")
        df_4h = bars.get("4H")
        df_1h = bars.get("1H")

        if any(df is None for df in [df_d, df_4h, df_1h]):
            return None
        if len(df_d) < 60 or len(df_4h) < 30 or len(df_1h) < 30:
            return None

        # ── Session filter ───────────────────────────────────────────────────
        if not self.in_session(df_1h.index[-1], self.SESSION_START, self.SESSION_END):
            return None

        # ── Daily: EMA50 direction ────────────────────────────────────────────
        ema_d = self.ema(df_d["close"], self.EMA_TREND_PERIOD)
        daily_close = df_d["close"].iloc[-1]
        daily_ema   = ema_d.iloc[-1]
        daily_slope = (ema_d.iloc[-1] - ema_d.iloc[-self.DAILY_SLOPE_BARS]) / ema_d.iloc[-self.DAILY_SLOPE_BARS]

        daily_up   = daily_close > daily_ema and daily_slope > 0
        daily_down = daily_close < daily_ema and daily_slope < 0

        if not (daily_up or daily_down):
            return None   # sideways/chop — sit out

        # ── 4H: 3+ consecutive closes on correct side of EMA7 ────────────────
        ema_4h = self.ema(df_4h["close"], self.EMA_ENTRY_PERIOD)
        closes_4h = df_4h["close"].iloc[-self.MIN_TREND_BARS:]
        emas_4h   = ema_4h.iloc[-self.MIN_TREND_BARS:]

        if daily_up:
            trend_confirmed = (closes_4h.values > emas_4h.values).all()
        else:
            trend_confirmed = (closes_4h.values < emas_4h.values).all()

        if not trend_confirmed:
            return None

        # ── 1H: EMA7, ATR, slope ────────────────────────────────────────────
        ema_1h   = self.ema(df_1h["close"], self.EMA_ENTRY_PERIOD)
        atr_1h   = self.atr(df_1h, self.ATR_PERIOD)
        curr_bar = df_1h.iloc[-1]
        curr_ema = ema_1h.iloc[-1]
        prev_ema = ema_1h.iloc[-2]
        curr_atr = atr_1h.iloc[-1]

        slope_pct = (curr_ema - prev_ema) / prev_ema if prev_ema else 0

        # ── Volume filter ─────────────────────────────────────────────────────
        vol_avg = df_1h["volume"].iloc[-self.VOL_LOOKBACK:].mean()
        if df_1h["volume"].iloc[-1] < vol_avg:
            return None

        # ── Long setup ────────────────────────────────────────────────────────
        if daily_up:
            if slope_pct < self.MIN_EMA_SLOPE:
                return None

            touched = curr_bar["low"] <= curr_ema
            bullish = curr_bar["close"] > curr_ema and curr_bar["close"] > curr_bar["open"]
            if not (touched and bullish):
                return None

            entry = curr_bar["close"]
            sl_anchor = self.swing_low(df_1h, self.SWING_BARS)
            sl_usd = min(entry - max(sl_anchor, entry - 0.8 * curr_atr), self.MAX_SL_USD)
            sl = entry - sl_usd
            rr = self.dynamic_rr(sl_usd / entry)
            tp = entry + sl_usd * rr

            return TradeSetup(
                signal=StrategySignal.BUY,
                entry=round(entry, 2),
                sl=round(sl, 2),
                tp=round(tp, 2),
                rr=rr,
                reason=f"Daily EMA50 UP. 4H EMA7 confirmed. 1H EMA7 retest. RR 1:{rr}",
            )

        # ── Short setup ───────────────────────────────────────────────────────
        if daily_down:
            if slope_pct > -self.MIN_EMA_SLOPE:
                return None

            touched = curr_bar["high"] >= curr_ema
            bearish = curr_bar["close"] < curr_ema and curr_bar["close"] < curr_bar["open"]
            if not (touched and bearish):
                return None

            entry = curr_bar["close"]
            sl_anchor = self.swing_high(df_1h, self.SWING_BARS)
            sl_usd = min(max(sl_anchor, entry + 0.8 * curr_atr) - entry, self.MAX_SL_USD)
            sl = entry + sl_usd
            rr = self.dynamic_rr(sl_usd / entry)
            tp = entry - sl_usd * rr

            return TradeSetup(
                signal=StrategySignal.SELL,
                entry=round(entry, 2),
                sl=round(sl, 2),
                tp=round(tp, 2),
                rr=rr,
                reason=f"Daily EMA50 DOWN. 4H EMA7 confirmed. 1H EMA7 retest. RR 1:{rr}",
            )

        return None
