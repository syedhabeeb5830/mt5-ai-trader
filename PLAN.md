# MT5 AI Trader — Compressed Implementation Plan

**Objective:** Determine whether this strategy has a measurable statistical edge. Fix what is broken. Build evidence.

---

## Code Audit Summary (Pre-Plan)

Before planning anything, every source file was read. The following critical issues were found. These must be understood before any build begins.

---

### CRITICAL BUGS FOUND

#### BUG-1 — Paper Mode Never Enforces SL or TP _(SEVERITY: CRITICAL)_

**File:** `order_manager.py` → `check_timeouts()` + `close()`

**Problem:**
In paper mode, trades are only closed by the 120-second timeout (`POSITION_TIMEOUT_SEC = 120`). The SL and TP fields are stored on the trade object but **are never checked** in the main loop against the current price in paper mode. The main loop calls `manager.check_timeouts(tick.mid)` — that's the only close trigger in paper mode.

**Impact:** This is the direct cause of losses exceeding intended SL. A trade with SL_POINTS=1 ($1 risk) can lose -$4.1 or -$5.2 because Gold can move $4–$5 in 120 seconds on high-volatility periods. The SL is cosmetic in paper mode.

**Evidence:** `close()` paper branch:

```python
if config.PAPER:
    trade.closed = True
    if trade.direction == Signal.BUY:
        trade.closed_pnl = (current_price - trade.entry) * trade.volume * 100
```

`current_price` at timeout may be far below SL. Nothing stops it.

---

#### BUG-2 — Live Mode PnL Is Never Fetched _(SEVERITY: CRITICAL)_

**File:** `order_manager.py` → `sync_closed_from_api()`

**Problem:**

```python
async def sync_closed_from_api(self) -> None:
    live_tickets = {p["ticket"] for p in await self.open_positions()}
    for trade in self.trades:
        if not trade.closed and trade.ticket not in live_tickets:
            trade.closed = True
            # closed_pnl stays at 0.0 — never fetched from MT5
```

When MT5 closes a trade via SL or TP broker-side, the system marks it closed but `closed_pnl` remains 0.0. All live-mode PnL reported by RiskGuard and the dashboard is **wrong**.

**Impact:** `DAILY_LOSS_LIMIT` check is based on 0.0 PnL for broker-closed trades. The risk guard is effectively disabled for SL/TP exits. The 122 trades and -$48.27 PnL figure may be inaccurate.

---

#### BUG-3 — Daily Trade Count Never Resets _(SEVERITY: HIGH)_

**File:** `order_manager.py` → `total_today` property

**Problem:**

```python
@property
def total_today(self) -> int:
    return len(self.trades)  # ALL trades in session, not today's
```

If the bot runs continuously past midnight, `MAX_DAILY_TRADES = 30` is never reset. A 48-hour session counts all trades against the daily limit. Also, the daily loss limit uses `realized_pnl` which has the same problem — it's a session total, not a daily total.

---

#### BUG-4 — Momentum Signal Has No Predictive Timeframe _(SEVERITY: HIGH)_

**File:** `momentum.py`

**Problem:**
The signal is evaluated on 8 ticks at 200ms polling intervals = approximately 1.6 seconds of price data. For XAUUSD (Gold):

- Normal tick frequency: 5–50 ticks/second during London/NY session
- 200ms poll captures at most 1 tick per poll if deduplication works
- Result: 8 "ticks" may span 1.6 seconds or 80 seconds depending on market activity
- Gold moves $0.3–$1.0 routinely in 2 seconds purely from bid/ask microstructure fluctuation

This is pure noise. The signal has no structural reason to predict direction. The 38.5% win rate is consistent with random entry.

---

#### BUG-5 — SL Placement Does Not Account for Spread _(SEVERITY: HIGH)_

**File:** `order_manager.py` → `enter()`

**Problem:**

```python
# BUY entry
entry_price = ask
sl = round(entry_price - config.SL_POINTS, 2)  # 1 point below ask
```

A BUY opens at ASK. Price is immediately quoted at BID = ASK - spread. The effective loss from entry is already `spread` points. With `MAX_SPREAD_POINTS = 0.8` and `SL_POINTS = 1.0`, the trade starts 0.8 points in the hole against a 1.0 point SL. The effective SL from bid perspective is only 0.2 points. The trade is triggered instantly on any minor reversal.

---

#### BUG-6 — Backtest Engine Exists But Scalper Is Not Backtest-Compatible _(SEVERITY: MEDIUM)_

**Files:** `backtest/engine.py`, `scalper.py`, `momentum.py`

**Problem:**
The backtest framework (`backtest/engine.py`) uses OHLCV bar-by-bar simulation and the `BaseStrategy` interface. The live scalper uses real-time tick data with `MomentumEngine`. These are completely separate systems with no connection. You cannot backtest the live scalper strategy using the existing engine. The strategies in `strategies/` (EMA7, SQRT Levels) are completely different from the tick-based scalper being traded live.

**Impact:** There is no historical evidence for the scalper strategy at all. The 122 trades are the only data, and that data is contaminated by BUG-1 and BUG-2.

---

#### BUG-7 — PnL Multiplier Is Hardcoded and Unverified _(SEVERITY: MEDIUM)_

**File:** `order_manager.py` + `backtest/engine.py`

**Problem:**
Both paper mode and backtest use `* volume * 100` as the PnL multiplier. For XAUUSD this assumes contract_size = 100 oz/lot, which is standard. However this is never validated against actual MT5 contract specifications and is not configurable. For non-XAUUSD symbols this will be wrong.

---

#### BUG-8 — RiskGuard `halted` Flag Never Resets _(SEVERITY: MEDIUM)_

**File:** `risk_guard.py`

**Problem:**
Once `halted = True`, there is no mechanism to reset it within a session. If the daily limit is hit at 10am, the bot stays halted for the rest of the session even if restarted. The halt is checked but never unset. This is by design for session safety, but combined with BUG-2 (PnL tracking), the guard may halt incorrectly.

---

#### BUG-9 — Tick Deduplication May Drop Valid Signals _(SEVERITY: LOW)_

**File:** `tick_feed.py`

**Problem:**

```python
if not self.ticks or tick.time_ms > self.ticks[-1].time_ms:
    self.ticks.append(tick)
```

This deduplication is correct (avoids duplicate timestamps) but at 200ms polling, if the broker sends the same timestamp twice (common in slow markets), the tick is dropped. During low-volatility periods, the momentum window may be stale by 10–30 seconds.

---

## Architecture Disconnect

The repository has **two parallel but disconnected systems**:

| System                           | Strategy                       | Signal Source   | Backtest? |
| -------------------------------- | ------------------------------ | --------------- | --------- |
| Live scalper (`scalper.py`)      | Tick momentum (8 ticks, 200ms) | Real-time ticks | ❌ None   |
| Backtest framework (`backtest/`) | EMA7 TBM, SQRT Levels          | OHLCV bars      | ✅ Exists |

The live system is running a strategy that has **zero backtesting evidence**. The strategies with backtesting evidence (EMA7 v2: PF 1.33, SQRT v4: PF 15.6) are **never executed live**.

This is the single most important structural problem in the repository.

---

## Compressed Phase Plan

Original 11 phases compressed into **3 phases**. Each phase has a clear gate — you do not proceed to the next phase without passing the gate condition.

---

### PHASE 1 — DIAGNOSE & FIX

_Compresses original Phases 1 + 2_

**Duration estimate:** 1–2 sessions

**Objective:** Make the system produce accurate data. Right now, reported PnL is wrong, SL is not enforced, and the live strategy has never been backtested. Nothing can be concluded from the 122 trades until this is fixed.

**Deliverables:**

1. **`AUDIT_REPORT.md`** — All 9 bugs documented with severity, root cause, and fix.

2. **Fix BUG-1 (Paper SL/TP):** Add SL/TP enforcement in the main scalper loop for paper mode:
   - Every poll: check if current tick crosses SL or TP
   - Close at SL/TP price, not current market
   - Validate: paper trade max loss must never exceed `SL_POINTS * volume * contract_size`

3. **Fix BUG-2 (Live PnL fetch):** After `sync_closed_from_api()` marks a trade closed, fetch the actual deal from MT5 history API and populate `closed_pnl`.

4. **Fix BUG-3 (Daily reset):** Add date tracking to `OrderManager`. Reset `trades` list (or filter by date) at midnight.

5. **Fix BUG-5 (Spread-aware SL):** For BUY: `effective_sl_distance = SL_POINTS - spread`. Reject entry if `effective_sl_distance <= 0`. Log when this triggers.

6. **Add validation test:** `tests/test_sl_integrity.py` — paper mode simulation proving SL is enforced within tolerance.

7. **`STOP_LOSS_VALIDATION.md`** — Before/after comparison of loss distribution.

**Gate Condition:** All paper-mode losses must be ≤ `SL_POINTS * volume * 100 + 0.1` (tolerance for tick granularity).

---

### PHASE 2 — MEASURE

_Compresses original Phases 3 + 4_

**Duration estimate:** 2–3 sessions

**Objective:** Generate the first statistically valid evidence. Answer: _Does this tick-momentum approach have any edge at all?_

**Deliverables:**

1. **Analytics Engine** (`analytics/engine.py`):
   - Reads `logs/closed_trades.csv` or DB
   - Computes: Win Rate, Avg Win, Avg Loss, Expectancy, Profit Factor, Largest W/L, Median W/L, Max Drawdown, Sharpe (if enough trades), Consecutive W/L, Trade distribution by hour
   - Outputs: `logs/analytics/report_YYYYMMDD.json` + `logs/analytics/report_YYYYMMDD.txt`
   - CLI: `python -m analytics.engine`

2. **Scalper Backtest Adapter** (`strategies/momentum_scalper.py`):
   - Wraps `MomentumEngine` logic into a `BaseStrategy`-compatible interface
   - Uses M1 OHLCV data (1-minute bars) as tick proxy (lower bound of realism)
   - Maps bar OHLCV → synthetic ticks (open, high, low, close prices as tick sequence)
   - SL/TP enforced at next-bar open + max adverse excursion within bar
   - This is the **honest** way to backtest a tick strategy without real tick data

3. **Run 6-month backtest** on `XAUUSD M1`:
   - Requires MT5 running or M1 CSV export
   - Output: `backtest/results/momentum_scalper_6m.json`
   - Report: total trades, win rate, profit factor, expectancy, max drawdown, equity curve CSV

4. **Honest Assessment Report** (`EVIDENCE_REPORT.md`):
   - State the result clearly
   - If PF < 1.0: "Edge not proven. Do not trade with real money."
   - If PF ≥ 1.0 and ≥ 100 trades: "Preliminary edge. Proceed to Phase 3."

**Gate Condition:**

- Profit Factor ≥ 1.0 **AND**
- ≥ 100 backtest trades **AND**
- Expectancy > 0 after realistic spread (0.3 points) and commission ($0.10/trade)

If gate fails: document clearly. Proceed to Phase 3 only to test alternative strategies (EMA7 v2, SQRT v4), not the tick scalper.

---

### PHASE 3 — IMPROVE ✅ COMPLETE (2026-06-02)

_Compresses original Phases 5 + 6 + 7 + 8 + 9 + 10 + 11_

**Status:** COMPLETE — all Phase 3 core deliverables built and validated.

---

#### Phase 3 Deliverables — COMPLETED

**3-Core — Dynamic Strategy Selector** ✅

- `analytics/strategy_selector.py` — batch-runs all REGISTRY strategies, ranks by composite score
  (0.35×PF + 0.30×WR + 0.20×expectancy + 0.15×sharpe), prints leaderboard, saves recommendation
- `analytics/live_strategy.py` — bridges any `BaseStrategy.evaluate(bars)` to the live scalper loop;
  `LiveStrategyRunner` fetches OHLCV bars from MT5 API every `OHLCV_REFRESH_SEC` seconds;
  `within_trade_hours()` enforces `TRADE_HOURS_UTC` filter; `auto_select_strategy()` reads recommendation
- `backtest/run.py --recommend` — runs all strategies, prints ranked leaderboard, saves `logs/recommendation.json`
- `backtest/run.py --oos` — out-of-sample split (IS 70% / OOS 30%), compares PF/WR to detect curve-fitting
- `backtest/data_loader.py` — extended: now downloads 5M + 1H + 4H (resampled) + D1 for all strategy TFs
- `scalper.py` — updated: `--strategy NAME`, `--auto-strategy`, `ACTIVE_STRATEGY` config, hour filter
- `config.py` — added: `ACTIVE_STRATEGY`, `TRADE_HOURS_UTC`, `OHLCV_REFRESH_SEC`
- `tests/test_strategy_selector.py` — 31 tests, all passing

**3-OOS — Out-of-Sample Validation** ✅ (2026-06-02)

| Split   | Period                  | Trades | PF    | WR    | Expectancy |
| ------- | ----------------------- | ------ | ----- | ----- | ---------- |
| IS 70%  | 2026-03-23 → 2026-05-11 | 1265   | 1.048 | 24.2% | +$0.051    |
| OOS 30% | 2026-05-11 → 2026-06-02 | 516    | 1.129 | 25.6% | +$0.135    |

**OOS gate PASSED** — OOS PF (1.129) = 107.7% of IS PF (1.048). Edge is not curve-fit.

**3-Leaderboard — Strategy Comparison** ✅ (60-day data)

| Rank | Strategy         | Trades | WR    | PF    | Score | Tag       |
| ---- | ---------------- | ------ | ----- | ----- | ----- | --------- |
| 1 ★  | momentum_scalper | 1785   | 24.5% | 1.068 | 0.240 | MARGINAL  |
| 2    | ema7_tbm_v2      | 16     | 18.8% | 0.737 | 0.056 | THIN DATA |
| 3    | ema7_tbm_v3      | 0      | —     | —     | —     | THIN DATA |
| 4    | sqrt_levels_v4   | 0      | —     | —     | —     | THIN DATA |

Only `momentum_scalper` has sufficient 5M data. EMA strategies require longer backtest windows
(they trade infrequently on 1H/4H bars). Re-run with `--days 365` after more 5M data accumulates.

**3-TimeFilter — Hour Analysis** ✅ (insight from Phase 2 backtest)

Best session: 14:00–18:00 UTC (+$97.00, ~23% of trades). Worst hour: 09:00 UTC (−$39.80).
Set `TRADE_HOURS_UTC=14-18` in `.env` to restrict to European afternoon.

**CLI Quick Reference:**

```bash
# Run leaderboard and auto-save recommendation
python -m backtest.run --recommend --days 60 --save

# Out-of-sample validation
python -m backtest.run --strategy momentum_scalper --days 60 --oos

# Auto-select best strategy from recommendation
python scalper.py --auto-strategy

# Pin a specific strategy
python scalper.py --strategy momentum_scalper

# Time-filtered live trading (14:00-18:00 UTC only, via .env)
# TRADE_HOURS_UTC=14-18  in .env, then:
python scalper.py --paper

# Re-run comparison with selector module directly
python -m analytics.strategy_selector --days 60 --metric composite --save
```

---

#### Phase 3 Sub-phases — PENDING (not yet started)

**3A — Trend Filter** — Add EMA50 on 5M as configurable trend filter
**3B — Volatility Filter** — ATR-based entry gate  
**3C — Dynamic Risk** — ATR-based SL/TP replacement  
**3D — Session Filter** — Trade session restriction (London / NY)  
**3E — Walk-Forward** — Rolling parameter optimization  
**3F — AI Analyst** — Wire `ai_loop/analyst.py` to analytics JSON

These are the next steps. Each requires passing a comparison gate before the next sub-phase begins.

---

## Current Status Summary

```
Phase 1 : ✅ COMPLETE — 4 bugs fixed, 11/11 tests pass
Phase 2 : ✅ COMPLETE — analytics engine, backtest, EVIDENCE_REPORT.md
Phase 3 : ✅ CORE COMPLETE — dynamic selector, OOS validation, live integration
           ⏳ Sub-phases 3A-3F pending
```

**Gate to live deployment:** Phase 3A gate (trend filter adds measurable edge improvement)

**Objective:** Systematically improve the strategy's edge through validated, additive filters. Each filter must be justified by before/after metrics.

**Sub-phases (sequential, each requires evidence before next):**

**3A — Trend Filter (original Phase 5)**

- Add EMA50 on M15 or M30 as configurable trend filter
- Config: `TREND_FILTER_ENABLED=true`, `TREND_EMA_PERIOD=50`, `TREND_TIMEFRAME=M15`
- Backtest with filter ON vs OFF
- Output: comparison table

**3B — Volatility Filter (original Phase 6)**

- ATR-based entry gate: `ATR_MIN_THRESHOLD`, `ATR_MAX_THRESHOLD`
- Do not enter when ATR < min (dead market) or > max (erratic market)
- Config: `VOLATILITY_FILTER_ENABLED=true`
- Backtest comparison

**3C — Dynamic Risk (original Phase 7)**

- Replace `SL_POINTS` with `SL = ATR_MULTIPLIER × ATR`
- Replace `TP_POINTS` with `TP = RR_RATIO × SL`
- Config: `DYNAMIC_RISK_ENABLED=true`, `ATR_SL_MULTIPLIER=0.5`, `RR_RATIO=2.0`
- Backtest comparison: fixed vs dynamic risk

**3D — Market Activity + Session Filter (original Phases 8 + 9)**

- Track tick rate per minute (rolling average)
- Only trade when current rate > average rate × `MIN_ACTIVITY_RATIO`
- Session filter: `TRADE_SESSIONS=london,newyork` (disable asian by default)
- Unified config for both filters
- Output: win rate / PnL breakdown by session

**3E — Walk-Forward Optimization (original Phase 10)**

- Grid search on: `MOMENTUM_WINDOW`, `MIN_DIRECTION_PCT`, `MIN_MOVE_POINTS`, `ATR_SL_MULTIPLIER`, `RR_RATIO`
- Structure: 4-month train, 2-month validate, repeat rolling
- Report best parameter sets + parameter stability score (how consistent across windows)
- Warning: any result with < 30 trades in the validation window is flagged as unreliable

**3F — AI Analyst (original Phase 11)**

- Wire existing `ai_loop/analyst.py` to read `logs/analytics/` output
- AI reads the JSON analytics report, NOT raw trades
- AI suggests: which filters to enable/disable, which parameters to investigate
- AI does NOT generate trade signals, does NOT override risk controls
- Gate: only enabled after 3E shows robust parameters

---

## What Gets Built in What Order

```
Phase 1:  Fix bugs → Validate SL integrity → AUDIT_REPORT.md
            ↓
Phase 2:  Analytics engine → Scalper backtest adapter → 6-month backtest
            ↓ (if PF ≥ 1.0 and expectancy > 0)
Phase 3A: EMA50 trend filter → backtest comparison
            ↓ (if improvement confirmed)
Phase 3B: ATR volatility filter → backtest comparison
            ↓ (if improvement confirmed)
Phase 3C: Dynamic SL/TP (ATR-based) → backtest comparison
            ↓ (if improvement confirmed)
Phase 3D: Activity + session filter → session stats
            ↓ (after all filters validated)
Phase 3E: Walk-forward optimization → robust parameter set
            ↓ (after stable parameters found)
Phase 3F: AI analyst → reads analytics, suggests experiments
```

---

## What Is NOT Being Built

- No UI improvements
- No dashboard redesign
- No new AI features (beyond Phase 3F)
- No architecture refactoring for its own sake
- No new strategies invented without evidence

---

## Files to Be Created

| File                                   | Phase | Purpose                            |
| -------------------------------------- | ----- | ---------------------------------- |
| `AUDIT_REPORT.md`                      | 1     | Full bug documentation             |
| `STOP_LOSS_VALIDATION.md`              | 1     | SL fix before/after proof          |
| `tests/test_sl_integrity.py`           | 1     | SL validation test                 |
| `analytics/engine.py`                  | 2     | Statistics calculator              |
| `analytics/__init__.py`                | 2     | Module init                        |
| `strategies/momentum_scalper.py`       | 2     | Backtest adapter for live scalper  |
| `backtest/results/`                    | 2     | Directory for result files         |
| `EVIDENCE_REPORT.md`                   | 2     | Go/no-go decision document         |
| `config.py` (additions)                | 3A–3D | New filter config keys             |
| Filter modules (inline in momentum.py) | 3A–3D | Trend/vol/session/activity filters |
| `backtest/optimizer.py`                | 3E    | Walk-forward optimizer             |
| `OPTIMIZATION_REPORT.md`               | 3E    | Parameter stability results        |

---

## Risk Assessment of Each Phase

| Phase                    | Risk of Breaking Live Trading                    | Reversibility                                 |
| ------------------------ | ------------------------------------------------ | --------------------------------------------- |
| 1 (bug fixes)            | Low — paper mode fixes, PnL fetch is additive    | High — all changes are additive or paper-only |
| 2 (analytics + backtest) | Zero — read-only and offline                     | N/A — no live changes                         |
| 3A–3D (filters)          | Medium — new conditions can suppress all signals | High — all behind feature flags, default OFF  |
| 3E (optimizer)           | Zero — offline only                              | N/A                                           |
| 3F (AI)                  | Low — analyst only, no trade execution           | High                                          |

All Phase 3 filters are implemented as **feature flags** (`FILTER_ENABLED=false` by default). The live system behavior is unchanged until a flag is explicitly enabled.

---

## Decision Gate Summary

| Gate              | Condition to Proceed                                                               |
| ----------------- | ---------------------------------------------------------------------------------- |
| Phase 1 → Phase 2 | All paper-mode losses ≤ SL tolerance, PnL tracking verified                        |
| Phase 2 → Phase 3 | Backtest PF ≥ 1.0, expectancy > 0, ≥ 100 trades                                    |
| Phase 2 FAIL      | Document clearly. Consider switching live strategy to EMA7 v2 (PF 1.33 backtested) |
| Phase 3A → 3B     | Filter improves or is neutral. Never proceed if filter makes PF worse              |
| Phase 3E → 3F     | At least one walk-forward window shows PF > 1.2 consistently                       |

---

## Honest Prior Assessment

Based on the code audit alone (before any backtest):

**The tick-momentum scalper is likely not viable** for the following reasons:

1. Signal window = 8 ticks × 200ms = ~1.6 seconds. Gold microstructure noise exceeds this signal.
2. SL = $1 with spread up to $0.8 = effective SL of $0.2. Near-zero edge needed.
3. TP = $5 requires Gold to move 5× the SL before reversing. Achievable but at only 38.5% rate in 122 trades, the math does not work. Expectancy = (0.385 × $5) − (0.615 × $1) = $1.925 − $0.615 = $1.31 per trade. **But this is before costs and assuming SL/TP are properly enforced.** With BUG-1 active, the actual loss distribution is unknown.
4. The only honest baseline will come from Phase 2 with bugs fixed.

**The EMA7 TBM v2 strategy (already in the codebase) has demonstrable backtested results:**

- Win Rate: 39.7%
- Profit Factor: 1.33
- 58 trades over 1 year
- This is a reasonable candidate for live deployment after independent verification.

---

_Ready to build Phase 1 on confirmation._
