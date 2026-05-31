# Strategies

Community-tested strategies for XAUUSD (Gold). All backtested on 1 year of real data.

> **Before going live:** Backtest first (`backtest/run.py`), then paper trade for at least 1 week.

---

## Strategy Scoreboard

| Strategy | Status | Win Rate | Profit Factor | Trades | Verdict |
|---|---|---|---|---|---|
| `ema7_tbm_v2.py` | Active | 39.7% | **1.33** | 58 | Use this |
| `ema7_tbm_v3.py` | Active | 36.8% | **1.19** | 57 | Use this |
| `sqrt_levels_v4.py` | Active | 65.0% | **15.6** | 20 | Promising — limited data |
| `ema7_tbm_15m.py` | Study | 30.7% | 0.55 | 806 | Do NOT use live — study only |

**Profit Factor > 1.0 = profitable. < 1.0 = losing money.**

---

## Strategy Details

### 1. 7 EMA TBM v2 — Long Only ✅
**File:** `ema7_tbm_v2.py`
**Timeframe:** 4H trend + 1H entry
**Win Rate:** 39.7% | **Profit Factor:** 1.33

The cleanest version of the EMA retest strategy. Long only — because Gold is in a structural uptrend and shorts statistically lose money. Waits for a strong 4H trend (3+ consecutive closes above EMA7), then enters when price retests the 1H EMA7 with a confirming bounce. Session filter removes the low-quality Asian session.

**Entry:** 4H has 3+ closes above EMA7 → 1H low touches EMA7 → 1H closes above EMA7 with bullish candle → EMA slope positive → volume above average → session 09:00–21:00 UTC
**Stop Loss:** Swing low (5 bars back), min 0.8×ATR, capped at $30
**Take Profit:** Dynamic — 1:5 RR if SL < 0.2%, 1:4 if SL < 0.4%, else 1:3

---

### 2. 7 EMA TBM v3 — Trend Follower ✅
**File:** `ema7_tbm_v3.py`
**Timeframe:** Daily + 4H + 1H (triple timeframe)
**Win Rate:** 36.8% | **Profit Factor:** 1.19

Triple-timeframe confirmation. Uses the Daily EMA50 to decide overall direction (no bias — trades longs in uptrends, shorts in downtrends, sits out in chop). Requires 4H EMA7 confirmation, then 1H entry. Most selective filter — sits out most days, strikes when the setup is clean.

**Entry (Long):** Daily close > EMA50 + slope up → 4H: 3+ closes > EMA7 → 1H: Low touches EMA7, close above, bullish candle, slope > 0.02%, volume > average, session 08–20 UTC
**Entry (Short):** Mirror of above, Daily EMA50 trending down
**Stop Loss:** Swing low/high (5 bars), min 0.8×ATR, capped at $30
**Take Profit:** Dynamic 1:3 to 1:5

---

### 3. SQRT Levels v4 — Trail Machine ⚡
**File:** `sqrt_levels_v4.py`
**Timeframe:** Daily levels + 1H entry
**Win Rate:** 65.0% | **Profit Factor:** 15.6 (only 20 trades — verify with more data)

Unique. Uses square root math to find natural price levels from the daily open. Formula: `level_n = (sqrt(daily_open) + n)^2`. These levels act as natural support/resistance. Enters with MACD + RSI + volume confirmation. No fixed TP — trailing stop rides the move. Average winner crosses 11+ levels ($400+). **Promising but small sample — backtest before using.**

**Entry:** Daily EMA50 direction → price approaches SQRT level → bullish/bearish candle (body > 50%) → MACD histogram accelerating → RSI in range → volume > 1.2× average → session 08–20 UTC
**Stop Loss:** Trailing — sits 3 SQRT levels behind the price, moves up only
**Take Profit:** No fixed TP — trail captures the full move

---

### 4. 7 EMA TBM 15M — Study Only ⚠️
**File:** `ema7_tbm_15m.py`
**Timeframe:** 1H trend + 15M entry
**Win Rate:** 30.7% | **Profit Factor:** 0.55 ← LOSING STRATEGY

Included for **educational purposes only**. This is the original version of the EMA TBM strategy — it loses money over 806 trades. Study it to understand why: too many false signals on 15M, no volume filter, no session filter, shorts on Gold lose money. Compare it to v2 to see what the improvements did.

**DO NOT USE THIS LIVE.**

---

## How to Run a Strategy Live

Connect any strategy to the main bot by setting it in `.env`:

```env
STRATEGY=ema7_tbm_v2
```

Then run:
```bash
python scalper.py
```

The bot will use the strategy's signal instead of the default momentum signal.

## How to Backtest

```bash
# Backtest one strategy
python backtest/run.py --strategy ema7_tbm_v2 --days 365

# Compare all strategies
python backtest/run.py --all --days 90
```

See `backtest/README.md` for full instructions.
