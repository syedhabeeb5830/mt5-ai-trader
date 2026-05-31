"""
7 EMA TBM v2 — Long Only
─────────────────────────────────────────────────────────────────────────────
Backtested: 39.7% win rate | Profit Factor: 1.33 | 58 trades (1 year)
Asset:      XAUUSD
Timeframes: 4H trend filter + 1H entry

Why Long Only:
  Gold is in a structural uptrend. Backtesting showed shorts lost $1,067 in v1.
  Asian session also removed — it added noise, not profit.

Entry Logic:
  1. 4H: 3+ consecutive candles close ABOVE EMA7  (strong trend)
  2. 1H: Price LOW touches EMA7 (retest)
  3. 1H: Candle CLOSES above EMA7 with bullish body
  4. 1H: EMA7 slope is positive (> 0.02%)
  5. 1H: Volume is above 20-bar average
  6. Time is 09:00–21:00 UTC (London + NY session only)

Stop Loss:  Swing low of last 5 bars, minimum 0.8x ATR, capped at $30
Take Profit: Dynamic RR (1:5 if SL < 0.2%, 1:4 if SL < 0.4%, else 1:3)

Source: ZeroOne D.O.T.S AI — strategy-dotsai-cloud
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
from strategies.base import BaseStrategy, StrategySignal, TradeSetup


class EMA7TBMv2(BaseStrategy):

    name        = "ema7_tbm_v2"
    version     = "2.0"
    asset       = "XAUUSD"
    description = "7 EMA TBM v2 — Long Only with session filter. PF 1.33, WR 39.7%"

    EMA_PERIOD     = 7
    ATR_PERIOD     = 14
    VOL_LOOKBACK   = 20
    SWING_BARS     = 5
    MIN_TREND_BARS = 3        # 4H bars consecutively above EMA
    MIN_EMA_SLOPE  = 0.0002   # 0.02% per bar minimum slope
    MAX_SL_USD     = 30.0
    SESSION_START  = 9        # UTC
    SESSION_END    = 21       # UTC

    def evaluate(self, bars: dict[str, pd.DataFrame]) -> Optional[TradeSetup]:
        df_4h = bars.get("4H")
        df_1h = bars.get("1H")

        if df_4h is None or df_1h is None:
            return None
        if len(df_4h) < 30 or len(df_1h) < 30:
            return None

        # ── Session filter ───────────────────────────────────────────────────
        last_1h = df_1h.index[-1]
        if not self.in_session(last_1h, self.SESSION_START, self.SESSION_END):
            return None

        # ── 4H: 3+ consecutive closes above EMA7 ─────────────────────────────
        ema_4h = self.ema(df_4h["close"], self.EMA_PERIOD)
        last_n = df_4h["close"].iloc[-self.MIN_TREND_BARS:]
        ema_n  = ema_4h.iloc[-self.MIN_TREND_BARS:]
        if not (last_n.values > ema_n.values).all():
            return None

        # ── 1H: EMA7 and ATR ────────────────────────────────────────────────
        ema_1h  = self.ema(df_1h["close"], self.EMA_PERIOD)
        atr_1h  = self.atr(df_1h, self.ATR_PERIOD)

        curr_bar  = df_1h.iloc[-1]
        prev_ema  = ema_1h.iloc[-2]
        curr_ema  = ema_1h.iloc[-1]
        curr_atr  = atr_1h.iloc[-1]

        # EMA slope check
        slope_pct = (curr_ema - prev_ema) / prev_ema if prev_ema else 0
        if slope_pct < self.MIN_EMA_SLOPE:
            return None

        # ── 1H: Retest — low touched EMA7 this bar ───────────────────────────
        touched_ema = curr_bar["low"] <= curr_ema

        # ── 1H: Bullish close above EMA7 ─────────────────────────────────────
        bullish_close = (
            curr_bar["close"] > curr_ema and
            curr_bar["close"] > curr_bar["open"]    # green candle
        )

        if not (touched_ema and bullish_close):
            return None

        # ── 1H: Volume above 20-bar average ──────────────────────────────────
        vol_avg = df_1h["volume"].iloc[-self.VOL_LOOKBACK:].mean()
        if df_1h["volume"].iloc[-1] < vol_avg:
            return None

        # ── Calculate SL / TP ─────────────────────────────────────────────────
        entry = curr_bar["close"]
        swing_low = self.swing_low(df_1h, self.SWING_BARS)
        sl_raw = entry - max(swing_low, entry - 0.8 * curr_atr)
        sl_usd = min(sl_raw, self.MAX_SL_USD)
        sl     = entry - sl_usd

        sl_pct = sl_usd / entry
        rr     = self.dynamic_rr(sl_pct)
        tp     = entry + sl_usd * rr

        return TradeSetup(
            signal=StrategySignal.BUY,
            entry=round(entry, 2),
            sl=round(sl, 2),
            tp=round(tp, 2),
            rr=rr,
            reason=(
                f"4H EMA7 trend confirmed ({self.MIN_TREND_BARS}+ bars). "
                f"1H retest with bullish close. EMA slope {slope_pct*100:.3f}%. "
                f"RR 1:{rr}"
            ),
        )
