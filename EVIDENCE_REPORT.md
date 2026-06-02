# EVIDENCE_REPORT.md — Phase 2 Gate Decision

**Strategy:** `momentum_scalper`  
**Symbol:** XAUUSD (GC=F proxy via yfinance, 5-minute bars)  
**Period:** 2026-03-23 → 2026-06-02 (60 days)  
**Bars processed:** 13,493 (5M)  
**Backtest spread:** 0.30 pts (conservative proxy for live spread)  
**Generated:** 2026-06-02

---

## 1. Gate Condition Checklist

| Condition                | Threshold | Actual           | Pass? |
| ------------------------ | --------- | ---------------- | ----- |
| Profit Factor ≥ 1.0      | ≥ 1.000   | **1.068**        | ✅    |
| Expectancy per trade > 0 | > $0.00   | **+$0.07/trade** | ✅    |
| Sample size ≥ 100 trades | ≥ 100     | **1,785 trades** | ✅    |

**GATE RESULT: PASSED — Preliminary edge exists. Proceed to Phase 3 with caveats.**

---

## 2. Full Metrics

| Metric                   | Value            |
| ------------------------ | ---------------- |
| Total Trades             | 1,785            |
| Win Rate                 | 24.5%            |
| Avg Win                  | +$4.60           |
| Avg Loss                 | −$1.40           |
| Gross Profit             | +$2,014.80       |
| Gross Loss               | −$1,885.80       |
| **Net PnL**              | **+$129.00**     |
| **Profit Factor**        | **1.068**        |
| **Expectancy**           | **+$0.07/trade** |
| Largest Win              | +$4.60           |
| Largest Loss             | −$1.40           |
| Max Drawdown             | **$111.60**      |
| Drawdown / Net PnL ratio | **86.5%**        |
| Sharpe Ratio             | 2.571            |
| Sortino Ratio            | 2.664            |
| Avg Holding Time         | 17.8 min         |
| Max Consecutive Wins     | 5                |
| Max Consecutive Losses   | **19**           |

---

## 3. Monthly Breakdown

| Month   | Net PnL | Signal       |
| ------- | ------- | ------------ |
| 2026-03 | +$45.20 | Positive     |
| 2026-04 | −$18.00 | **Negative** |
| 2026-05 | +$65.40 | Positive     |
| 2026-06 | +$36.40 | Positive     |

One of four months was negative. The strategy is not monotonically profitable.

---

## 4. Best and Worst Hours (UTC)

**Top 3 profitable hours:** 16:00 (+$34.20), 06:00 (+$35.80), 23:00 (+$22.40)  
**Worst 3 hours:** 09:00 (−$39.80), 20:00 (−$16.80), 03:00 (−$10.60)

**Critical finding:** The 09:00 UTC hour (London open) is the single most destructive
hour (−$39.80 across 67 trades). European afternoon (14:00–18:00 UTC) is the most
consistently profitable window (+$97.00 combined, 4 hours).

---

## 5. Risk Concerns — Do NOT Ignore

### 5a. Marginal Profit Factor

PF = 1.068 is 6.8% above breakeven. A small increase in live spread, commissions, or
slippage would erase this edge entirely. In live trading, add:

- Broker commission per trade: ~$0.07–$0.15 per 0.01 lot round-trip
- Real spread variance: can be 2–3× backtest spread during news events
- Slippage on exits: typically 0.1–0.5 pts on 5M bars

**Estimated breakeven at 0.01 lot:** live costs of ~$0.10/trade would reduce expectancy
from +$0.07 to −$0.03, flipping the strategy to a net loser.

### 5b. Severe Drawdown Ratio

Max Drawdown ($111.60) represents **86.5% of total Net PnL ($129.00)**.  
This means the strategy earned $129 over 60 days but required surviving a $111.60
drawdown to collect it. At 0.01 lot, a $200 mini-account would have been wiped.

**Minimum recommended account for live test (0.01 lot):** ≥ $500 to survive the
observed drawdown with a 2× safety buffer.

### 5c. 19 Consecutive Losses

The strategy's loss rate is 75.5%. The maximum consecutive losing streak of 19 trades
means the system went 19 trades without a win. At SL=1.0pt per trade, this is
a −19pt run. Psychologically and mechanically, this requires discipline and pre-funded
capital to weather.

### 5d. Uniform Win/Loss Values (Structural Observation)

Every win is exactly $4.60 and every loss is exactly $1.40. This occurs because the
backtest uses fixed bar-close prices for entries; in live trading, the entry price
varies and fills are not guaranteed at the signal bar's close. Actual win/loss
distribution will be less uniform.

---

## 6. Time-Filter Opportunity

Based on hourly PnL data, restricting the strategy to the **14:00–18:00 UTC** window
may improve the Profit Factor significantly:

| Window          | Trades | Net PnL  | Notes                      |
| --------------- | ------ | -------- | -------------------------- |
| All hours       | 1,785  | +$129.00 | PF 1.068                   |
| 14:00–18:00 UTC | ~418   | +$97.00  | European afternoon session |
| Excluding 09:00 | ~1,718 | +$168.80 | Avoids London open spike   |

A filtered backtest (Phase 3 candidate) should test these sub-windows.

---

## 7. Strategy Comparison Note

The codebase already contains `ema7_tbm_v2` with a documented PF of 1.33 in earlier
backtests — significantly higher than momentum_scalper's 1.068. If Phase 3 confirms
the EMA7 TBM strategy maintains its edge on current data, consider switching the
**live strategy** from momentum_scalper to ema7_tbm_v2 for the first real-money test.

To verify: `python -m backtest.run --strategy ema7_tbm_v2 --days 180`

---

## 8. Phase 3 Recommendations

Given the gate is passed (marginally), the following must be done before any live
deployment:

1. **Out-of-sample test**: Download 2024 data (1H timeframe, ~365 days) and run
   momentum_scalper on it. If PF ≥ 1.0 holds, the edge is not curve-fit to 2026 data.

2. **Time filter**: Re-backtest with trades restricted to 14:00–18:00 UTC only.
   Target PF ≥ 1.2 in this window before enabling live.

3. **Commission sensitivity test**: Re-run with `TP_POINTS=5.0` and `SL_POINTS=1.0`
   but subtract $0.10/trade from every PnL. If still profitable → live-ready.

4. **Compare vs EMA7 TBM v2**: Run `--strategy ema7_tbm_v2 --days 180` and compare
   PF, drawdown, and Sharpe. Use the better strategy for live.

5. **Paper trade 2 weeks first**: Even if all above pass, paper trade before real money.
   The PAPER=true mode now has correct SL/TP enforcement (BUG-1 fix in Phase 1).

---

## 9. Decision Summary

```
GATE:    PASSED (PF 1.068, Expectancy +$0.07, N=1785)
VERDICT: Preliminary statistical edge exists.
WARNING: Edge is thin — live costs likely reduce PF below 1.0.
ACTION:  Phase 3 — time-filter test + out-of-sample validation before live.
DO NOT:  Trade real money until Phase 3 gate is passed.
```

---

_Generated from: `logs/analytics/momentum_scalper_20260602_214038.json`_  
_Backtest trades: `logs/backtest/momentum_scalper_20260602_214038.csv`_
