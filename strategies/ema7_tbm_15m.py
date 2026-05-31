"""
7 EMA TBM 15M — STUDY / REFERENCE ONLY
─────────────────────────────────────────────────────────────────────────────
WARNING: This strategy LOSES MONEY in backtesting.
  Win Rate: 30.7% | Profit Factor: 0.55 | 806 trades

DO NOT USE THIS LIVE. It is included for educational purposes only.
Compare it to ema7_tbm_v2.py to understand what improvements were made.

Why it loses:
  - 15M timeframe generates too many false signals (noise)
  - No volume filter — enters on weak moves
  - No session filter — trades Asian session (low quality)
  - Shorts on Gold are unprofitable structurally
  - Dynamic TP is not aggressive enough for 15M timeframe

Study this alongside v2 to understand how filtering improves a strategy.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
from strategies.base import BaseStrategy, StrategySignal, TradeSetup


class EMA7TBM15M(BaseStrategy):

    name        = "ema7_tbm_15m"
    version     = "1.0"
    asset       = "XAUUSD"
    description = "STUDY ONLY. PF 0.55 — this strategy loses money. Do not use live."

    EMA_PERIOD     = 7
    ATR_PERIOD     = 14
    SWING_BARS     = 10
    MIN_TREND_BARS = 2      # 1H bars consecutively above/below EMA
    MAX_SL_USD     = 20.0   # tighter cap than v2

    def evaluate(self, bars: dict[str, pd.DataFrame]) -> Optional[TradeSetup]:
        df_1h  = bars.get("1H")
        df_15m = bars.get("15M")

        if df_1h is None or df_15m is None:
            return None

        # ── 1H trend: 2+ consecutive closes above EMA7 ───────────────────────
        ema_1h   = self.ema(df_1h["close"], self.EMA_PERIOD)
        closes   = df_1h["close"].iloc[-self.MIN_TREND_BARS:]
        emas     = ema_1h.iloc[-self.MIN_TREND_BARS:]
        trend_up   = (closes.values > emas.values).all()
        trend_down = (closes.values < emas.values).all()

        if not (trend_up or trend_down):
            return None

        # ── 15M: EMA7 retest ──────────────────────────────────────────────────
        ema_15m  = self.ema(df_15m["close"], self.EMA_PERIOD)
        atr_15m  = self.atr(df_15m, self.ATR_PERIOD)
        curr     = df_15m.iloc[-1]
        curr_ema = ema_15m.iloc[-1]
        curr_atr = atr_15m.iloc[-1]

        if trend_up:
            touched = curr["low"] <= curr_ema
            bullish = curr["close"] > curr_ema and curr["close"] > curr["open"]
            ema_rising = ema_15m.iloc[-1] > ema_15m.iloc[-2]

            if not (touched and bullish and ema_rising):
                return None

            entry = curr["close"]
            sl_raw = entry - max(self.swing_low(df_15m, self.SWING_BARS), entry - curr_atr)
            sl_usd = min(sl_raw, self.MAX_SL_USD)
            sl = entry - sl_usd

            sl_pct = sl_usd / entry
            rr = 4.0 if sl_pct < 0.0015 else (3.0 if sl_pct < 0.003 else 2.0)
            tp = entry + sl_usd * rr

            return TradeSetup(
                signal=StrategySignal.BUY,
                entry=round(entry, 2),
                sl=round(sl, 2),
                tp=round(tp, 2),
                rr=rr,
                reason=f"[STUDY] 1H trend up. 15M EMA7 retest. RR 1:{rr}",
            )

        if trend_down:
            touched = curr["high"] >= curr_ema
            bearish = curr["close"] < curr_ema and curr["close"] < curr["open"]
            ema_falling = ema_15m.iloc[-1] < ema_15m.iloc[-2]

            if not (touched and bearish and ema_falling):
                return None

            entry = curr["close"]
            sl_raw = max(self.swing_high(df_15m, self.SWING_BARS), entry + curr_atr) - entry
            sl_usd = min(sl_raw, self.MAX_SL_USD)
            sl = entry + sl_usd

            sl_pct = sl_usd / entry
            rr = 4.0 if sl_pct < 0.0015 else (3.0 if sl_pct < 0.003 else 2.0)
            tp = entry - sl_usd * rr

            return TradeSetup(
                signal=StrategySignal.SELL,
                entry=round(entry, 2),
                sl=round(sl, 2),
                tp=round(tp, 2),
                rr=rr,
                reason=f"[STUDY] 1H trend down. 15M EMA7 retest. RR 1:{rr}",
            )

        return None
