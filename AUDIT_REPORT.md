# AUDIT REPORT — MT5 AI Trader Phase 1

**Date:** 2026-06-02  
**Scope:** Full codebase audit — signal generation, execution, risk management, logging, position management, SL/TP handling, spread filtering

---

## Summary

| #     | Issue                                             | Severity | Status     |
| ----- | ------------------------------------------------- | -------- | ---------- |
| BUG-1 | Paper mode never enforces SL or TP                | CRITICAL | FIXED      |
| BUG-2 | Live mode PnL never fetched from MT5              | CRITICAL | FIXED      |
| BUG-3 | Daily trade count and PnL never reset at midnight | HIGH     | FIXED      |
| BUG-4 | Tick-momentum signal has no predictive basis      | HIGH     | DOCUMENTED |
| BUG-5 | SL placement ignores spread cost                  | HIGH     | FIXED      |
| BUG-6 | Live scalper cannot be backtested                 | MEDIUM   | PHASE 2    |
| BUG-7 | PnL multiplier hardcoded and unvalidated          | MEDIUM   | DOCUMENTED |
| BUG-8 | RiskGuard `halted` flag never resets              | MEDIUM   | DOCUMENTED |
| BUG-9 | Tick deduplication may stale-read momentum window | LOW      | DOCUMENTED |

---

## BUG-1 — Paper Mode Never Enforces SL or TP

**Severity:** CRITICAL  
**File:** `order_manager.py` → `close()`, `check_timeouts()`  
**Status:** FIXED

### Root Cause

In paper mode, the only mechanism that closes an open trade is `check_timeouts()`, which fires after `POSITION_TIMEOUT_SEC = 120` seconds. The SL and TP values are stored on the trade object but are never checked against the current price during the 120-second window. The close method computes PnL using whatever price exists at the timeout moment — which can be far beyond the intended SL.

```python
# BEFORE (broken): trade closed at random market price after 120s
await manager.check_timeouts(tick.mid)

# AFTER (fixed): SL/TP enforced every tick, timeout is a fallback only
await manager.check_paper_exits(tick.bid, tick.ask)
await manager.check_timeouts(tick.mid)
```

### Impact

This is the direct and complete explanation for why reported losses (-4.1, -5.2) exceeded the configured `SL_POINTS = 1.0`. Gold (XAUUSD) can and does move $4–5 in 120 seconds during high-volatility periods. The SL was cosmetic.

### Fix

Added `OrderManager.check_paper_exits(bid, ask)` which runs on every tick before `check_timeouts`. It checks bid against SL/TP for BUY trades and ask against SL/TP for SELL trades (realistic spread simulation). Closes at exact SL or TP price.

### Validation

`tests/test_sl_integrity.py`:

- `test_buy_sl_enforced` — PASS
- `test_sell_sl_enforced` — PASS
- `test_buy_tp_enforced` — PASS
- `test_sell_tp_enforced` — PASS
- `test_trade_not_closed_twice` — PASS

---

## BUG-2 — Live Mode PnL Never Fetched from MT5

**Severity:** CRITICAL  
**File:** `order_manager.py` → `sync_closed_from_api()`  
**Status:** FIXED

### Root Cause

When MT5 closes a trade broker-side (via SL or TP trigger), `sync_closed_from_api()` detected the disappearance from open positions and set `trade.closed = True` — but never fetched the actual realised PnL. The `closed_pnl` field remained at its default value of `0.0`.

```python
# BEFORE (broken): PnL stays 0.0 for all broker-closed trades
for trade in self.trades:
    if not trade.closed and trade.ticket not in live_tickets:
        trade.closed = True  # closed_pnl = 0.0 always

# AFTER (fixed): fetch actual PnL from MT5 deal history
for trade in self.trades:
    if not trade.closed and trade.ticket not in live_tickets:
        trade.closed_pnl = await self._fetch_deal_pnl(trade.ticket)
        trade.closed = True
```

### Impact

- `RiskGuard.check()` uses `manager.realized_pnl` to enforce the daily loss limit. With all broker-closed PnLs at 0.0, the risk guard was always reading near-zero losses and never halting correctly.
- All session summaries and dashboard displays showed incorrect total PnL.
- The reported -$48.27 figure may be inaccurate; the true figure requires re-reading logs combined with MT5 deal history.

### Fix

Added `OrderManager._fetch_deal_pnl(ticket)` which calls the new `/deals/{ticket}` endpoint on the MT5 server. Added corresponding `GET /deals/{ticket}` endpoint to `mt5_server.py` using `mt5.history_deals_get()` with a 30-day lookback.

### Note

This fix requires MT5 connection to verify. Will be validated on personal laptop during live testing.

---

## BUG-3 — Daily Trade Count and PnL Never Reset

**Severity:** HIGH  
**File:** `order_manager.py` → `total_today`, `realized_pnl`  
**Status:** FIXED

### Root Cause

```python
# BEFORE (broken): counts ALL trades in the session, not today's
@property
def total_today(self) -> int:
    return len(self.trades)   # grows forever

@property
def realized_pnl(self) -> float:
    return sum(t.closed_pnl for t in self.trades if t.closed)  # session total
```

If the bot runs continuously past midnight (e.g. from Sunday evening through Monday), `MAX_DAILY_TRADES = 30` and `DAILY_LOSS_LIMIT = -$50` apply to all accumulated trades and PnL since startup, not to the current trading day.

### Fix

Both properties now filter using `datetime.fromtimestamp(t.opened_at, tz=timezone.utc).date()` compared against `datetime.now(timezone.utc).date()`. Trades from previous calendar days are automatically excluded without any explicit reset mechanism.

### Validation

- `test_total_today_excludes_yesterday` — PASS
- `test_realized_pnl_excludes_yesterday` — PASS

---

## BUG-4 — Tick-Momentum Signal Has No Predictive Basis

**Severity:** HIGH  
**File:** `momentum.py`  
**Status:** DOCUMENTED — strategy-level issue, addressed in Phase 2

### Root Cause

The momentum signal uses 8 consecutive ticks polled at 200ms intervals (~1.6 seconds of data). For XAUUSD:

- Normal bid/ask microstructure noise on Gold is $0.2–$0.8 per tick
- A $0.5 move in 1.6 seconds is normal market noise, not directional signal
- The threshold `MIN_MOVE_POINTS = 0.5` is within the noise floor

The 38.5% win rate on 122 trades is statistically consistent with random entry (expected ~50% on random, skewed lower by the spread cost on every trade). A 38.5% rate with TP=5 and SL=1 would yield positive expectancy _if_ the SL were actually enforced — but BUG-1 invalidated all historical data.

### Impact

Until Phase 2 generates a properly backtested baseline with BUG-1 fixed, no conclusion about strategy viability can be drawn from the 122 trades.

### Required Action (Phase 2)

Build `strategies/momentum_scalper.py` as a backtest-compatible wrapper of `MomentumEngine` and run it against 6 months of XAUUSD M1 data with realistic spread and commission.

---

## BUG-5 — SL Placement Does Not Guard Against Spread Cost

**Severity:** HIGH  
**File:** `order_manager.py` → `enter()`  
**Status:** FIXED

### Root Cause

With `SL_POINTS = 1.0` and `MAX_SPREAD_POINTS = 0.8`, the system could enter when spread = 0.75. A BUY at ask with spread = 0.75 immediately puts the position 0.75 points in the hole at bid. The effective distance to SL from the bid perspective is only 0.25 points — an almost instant stop-out.

### Fix

Added a spread guard at the top of `enter()`:

```python
spread = round(ask - bid, 5)
if spread >= config.SL_POINTS:
    return None
```

Entries are now rejected if the spread consumes the entire SL distance. The existing `MAX_SPREAD_POINTS` filter in `scalper.py` remains as the first filter; this adds a secondary safety check inside the order placement logic itself.

### Validation

- `test_entry_rejected_when_spread_equals_sl` — PASS
- `test_entry_rejected_when_spread_exceeds_sl` — PASS
- `test_entry_accepted_when_spread_below_sl` — PASS

---

## BUG-6 — Live Scalper Cannot Be Backtested

**Severity:** MEDIUM  
**File:** `backtest/engine.py`, `scalper.py`, `momentum.py`  
**Status:** PHASE 2

### Root Cause

The repository contains two completely disconnected systems:

| System                           | Strategy                       | Signal Source   | Backtest? |
| -------------------------------- | ------------------------------ | --------------- | --------- |
| Live scalper (`scalper.py`)      | Tick momentum (8 ticks, 200ms) | Real-time ticks | None      |
| Backtest framework (`backtest/`) | EMA7 TBM, SQRT Levels          | OHLCV bars      | Exists    |

The strategies with backtested evidence (EMA7 v2: PF 1.33, SQRT v4: PF 15.6) are never executed live. The strategy being traded live has zero backtesting evidence.

### Required Action (Phase 2)

Create `strategies/momentum_scalper.py` implementing `BaseStrategy` that wraps the tick-momentum logic using M1 OHLCV bars as a tick proxy.

---

## BUG-7 — PnL Multiplier Hardcoded

**Severity:** MEDIUM  
**File:** `order_manager.py` → `close()`, `backtest/engine.py`  
**Status:** DOCUMENTED

### Root Cause

Both paper mode and backtest use `* volume * 100` as the PnL multiplier. This assumes XAUUSD contract size = 100 oz/lot, which is standard for most brokers. However, this is never validated against the actual MT5 contract specification and is not configurable.

### Risk

For XAUUSD this is almost certainly correct. For any other symbol this will be wrong. If broker uses a non-standard contract size (some ECN brokers use different specs), all PnL calculations are wrong.

### Recommended Fix (Phase 2)

Add `SYMBOL_CONTRACT_SIZE = float(os.getenv("CONTRACT_SIZE", "100"))` to config and use it in both places.

---

## BUG-8 — RiskGuard `halted` Flag Never Resets

**Severity:** MEDIUM  
**File:** `risk_guard.py`  
**Status:** DOCUMENTED — by-design behavior, but interacts with BUG-2

### Root Cause

Once `halted = True`, no mechanism exists to reset within a session. This is intentional (circuit breaker behavior). However, combined with BUG-2 (PnL = 0.0 for broker-closed trades), the guard could fire incorrectly if the few paper-mode PnLs that were tracked happened to hit the -$50 limit before actual drawdown occurred.

### Status

BUG-2 is now fixed. The halt behavior itself is correct as a safety feature.

---

## BUG-9 — Tick Deduplication May Stale Momentum Window

**Severity:** LOW  
**File:** `tick_feed.py`  
**Status:** DOCUMENTED

### Root Cause

```python
if not self.ticks or tick.time_ms > self.ticks[-1].time_ms:
    self.ticks.append(tick)
```

During quiet markets, the broker may send the same timestamp for multiple successive polls. These ticks are correctly deduplicated. However, the momentum window then operates on ticks that may span a much longer actual time period than the 200ms polling interval suggests. An 8-tick window with deduplication active could represent 30–60 seconds of real time rather than 1.6 seconds.

### Impact

Low — this actually makes the signal slightly less noisy (fewer duplicate ticks), but the timestamps stored in `Tick` do not reflect this accurately, making debugging harder.

---

## Architecture Note

The most impactful finding is the disconnect between the live system and the backtesting framework. The two EMA7 strategies and SQRT Levels strategy in `strategies/` have documented backtested performance. The tick scalper being traded live has no backtesting history and has been running with a broken SL mechanism. **No valid performance conclusions can be drawn from the 122-trade history.**

Phase 2 will produce the first statistically valid evidence.
