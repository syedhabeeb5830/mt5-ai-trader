# Backtesting

Test any strategy against historical data before risking real money.

---

## Quick Start

```bash
# Backtest the best strategy (1 year)
python backtest/run.py --strategy ema7_tbm_v2 --days 365

# Compare all strategies side by side
python backtest/run.py --all --days 365

# Save trade list to CSV
python backtest/run.py --strategy ema7_tbm_v3 --days 180 --save

# Study the losing strategy (educational)
python backtest/run.py --strategy ema7_tbm_15m --days 365
```

---

## Data Sources

The backtest engine tries two sources in order:

### 1. MT5 (Automatic — recommended)
If MetaTrader5 is installed and `mt5_server.py` is running, data loads automatically.
No extra steps needed.

### 2. CSV Files (Fallback)
If MT5 is not available, place CSV files in `backtest/data/`:

```
backtest/data/
├── XAUUSD_1H.csv
├── XAUUSD_4H.csv
├── XAUUSD_D1.csv
└── XAUUSD_15M.csv
```

**CSV Format:**
```
time,open,high,low,close,volume
2025-01-01 00:00:00+00:00,2620.5,2625.0,2618.0,2622.5,1200
2025-01-01 01:00:00+00:00,2622.5,2628.0,2620.0,2626.0,980
...
```

**Export from MT5:**
1. Open MT5 → History Center (F2)
2. Select symbol + timeframe
3. Export → CSV

---

## Understanding Results

```
Backtest: EMA7_TBM_V2  |  XAUUSD  |  2025-01-01 → 2026-01-01

Total Trades    58        Win Rate      39.7%
Winning         23        Losing        35
Profit Factor   1.33      Total PnL     +$187.40
Avg Win         +$18.50   Avg Loss      -$8.90
Max Drawdown    $62.00

Verdict: PROFITABLE — Verify with more data
```

| Metric | What it means | Good range |
|---|---|---|
| **Win Rate** | % of trades that profit | 30–60% (depends on RR) |
| **Profit Factor** | Total wins / Total losses | > 1.0 (higher = better) |
| **Max Drawdown** | Biggest losing streak | < 20% of account |
| **Avg Win / Avg Loss** | Average win vs loss size | Win > Loss |

**A strategy with 30% win rate can still be profitable** if the average win is 3× the average loss.

---

## Important Warnings

**Overfitting:** If you keep tweaking a strategy until the backtest looks great, it will likely fail live. A strategy that works "okay" on historical data often works better live than a "perfect" one.

**Small sample:** Less than 30 trades = unreliable results. Always backtest at least 100 trades worth of data.

**Look-ahead bias:** The engine processes bars sequentially so there is no look-ahead. But be careful if you manually adjust strategy parameters after seeing results.

**Spread:** Default slippage is 0.3 points. Set this to match your actual broker spread for accurate results.

---

## Running Custom Backtests

```python
from backtest.engine import BacktestEngine, fetch_bars
from strategies.ema7_tbm_v2 import EMA7TBMv2

# Load data
bars = {
    "4H": fetch_bars("XAUUSD", "4H", 2000),
    "1H": fetch_bars("XAUUSD", "1H", 8000),
}

# Run backtest
engine = BacktestEngine(symbol="XAUUSD", volume=0.01)
result = engine.run(EMA7TBMv2(), bars, entry_tf="1H")

print(f"Win Rate: {result.win_rate}%")
print(f"Profit Factor: {result.profit_factor}")
print(f"Total PnL: ${result.total_pnl}")
```
