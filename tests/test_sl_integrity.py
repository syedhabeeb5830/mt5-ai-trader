"""
SL Integrity Tests — Phase 1 Validation
─────────────────────────────────────────────────────────────────────────────
Verifies that:
  1. Paper-mode losses never exceed SL_POINTS × volume × contract_size
  2. TP closes at exactly TP_POINTS profit
  3. Trades stay open when price is between SL and TP
  4. Entry is rejected when spread >= SL_POINTS
  5. daily total_today and realized_pnl scope to UTC today only

Runs fully offline — no MT5 or network connection required.

Usage:
  python tests/test_sl_integrity.py          # standalone runner
  pytest tests/test_sl_integrity.py -v       # via pytest
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

# ── Bootstrap: set env vars before any config import ─────────────────────────
os.environ.setdefault("PAPER",     "true")
os.environ.setdefault("SL_POINTS", "1.0")
os.environ.setdefault("TP_POINTS", "5.0")
os.environ.setdefault("VOLUME",    "0.01")
os.environ.setdefault("MT5_API_URL", "http://localhost:8000")

# Add repo root to path so imports work from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

config.PAPER     = True
config.SL_POINTS = 1.0
config.TP_POINTS = 5.0
config.VOLUME    = 0.01

from order_manager import OpenTrade, OrderManager
from momentum import Signal

# ── Helpers ───────────────────────────────────────────────────────────────────

_loop = asyncio.new_event_loop()


def run(coro):
    """Run a coroutine synchronously using the module-level loop."""
    return _loop.run_until_complete(coro)


# Max acceptable loss per trade: SL_POINTS × volume × contract_size (100 oz/lot for XAUUSD)
MAX_LOSS = config.SL_POINTS * config.VOLUME * 100   # = $1.00
TOLERANCE = 0.01                                     # float rounding tolerance


# ── Test Cases ────────────────────────────────────────────────────────────────

class TestSLIntegrity:

    # ── BUY SL ────────────────────────────────────────────────────────────────

    def test_buy_sl_enforced(self):
        """BUY trade: loss must equal exactly SL_POINTS when bid crashes below SL."""
        mgr = OrderManager()
        # spread = 0.30 < SL_POINTS=1.0 → entry allowed
        trade = run(mgr.enter(Signal.BUY, ask=3000.50, bid=3000.20))
        assert trade is not None, "Entry should be accepted (spread 0.30 < SL 1.0)"

        expected_sl = round(3000.50 - config.SL_POINTS, 2)   # 2999.50
        assert trade.sl == expected_sl, f"SL placed wrong: {trade.sl} != {expected_sl}"

        # Price crashes far below SL
        run(mgr.check_paper_exits(bid=2995.00, ask=2995.30))

        assert trade.closed, "Trade must be closed when bid <= SL"
        expected_pnl = round((trade.sl - trade.entry) * config.VOLUME * 100, 4)
        assert abs(trade.closed_pnl - expected_pnl) < TOLERANCE, (
            f"PnL {trade.closed_pnl} should equal SL loss {expected_pnl}"
        )
        assert trade.closed_pnl >= -(MAX_LOSS + TOLERANCE), (
            f"Loss {trade.closed_pnl} exceeds max allowed -{MAX_LOSS}"
        )

    # ── BUY TP ────────────────────────────────────────────────────────────────

    def test_buy_tp_enforced(self):
        """BUY trade: profit must equal TP_POINTS when bid rises to TP."""
        mgr = OrderManager()
        trade = run(mgr.enter(Signal.BUY, ask=3000.50, bid=3000.20))
        assert trade is not None

        expected_tp = round(3000.50 + config.TP_POINTS, 2)   # 3005.50
        assert trade.tp == expected_tp, f"TP placed wrong: {trade.tp} != {expected_tp}"

        # Bid rises well past TP
        run(mgr.check_paper_exits(bid=3006.00, ask=3006.30))

        assert trade.closed, "Trade must be closed when bid >= TP"
        expected_pnl = round((trade.tp - trade.entry) * config.VOLUME * 100, 4)
        assert abs(trade.closed_pnl - expected_pnl) < TOLERANCE, (
            f"TP PnL {trade.closed_pnl} should equal {expected_pnl}"
        )
        assert trade.closed_pnl > 0, "TP close must be profitable"

    # ── SELL SL ───────────────────────────────────────────────────────────────

    def test_sell_sl_enforced(self):
        """SELL trade: loss must equal exactly SL_POINTS when ask spikes above SL."""
        mgr = OrderManager()
        trade = run(mgr.enter(Signal.SELL, ask=3000.50, bid=3000.20))
        assert trade is not None

        expected_sl = round(3000.20 + config.SL_POINTS, 2)   # 3001.20
        assert trade.sl == expected_sl, f"SELL SL placed wrong: {trade.sl} != {expected_sl}"

        # Ask spikes far above SL
        run(mgr.check_paper_exits(bid=3005.00, ask=3005.30))

        assert trade.closed, "SELL trade must close when ask >= SL"
        expected_pnl = round((trade.entry - trade.sl) * config.VOLUME * 100, 4)
        assert abs(trade.closed_pnl - expected_pnl) < TOLERANCE, (
            f"SELL SL PnL {trade.closed_pnl} should equal {expected_pnl}"
        )
        assert trade.closed_pnl >= -(MAX_LOSS + TOLERANCE), (
            f"Loss {trade.closed_pnl} exceeds max allowed -{MAX_LOSS}"
        )

    # ── SELL TP ───────────────────────────────────────────────────────────────

    def test_sell_tp_enforced(self):
        """SELL trade: profit must equal TP_POINTS when ask drops to TP."""
        mgr = OrderManager()
        trade = run(mgr.enter(Signal.SELL, ask=3000.50, bid=3000.20))
        assert trade is not None

        expected_tp = round(3000.20 - config.TP_POINTS, 2)   # 2995.20
        assert trade.tp == expected_tp, f"SELL TP placed wrong: {trade.tp} != {expected_tp}"

        # Ask drops well past TP
        run(mgr.check_paper_exits(bid=2994.50, ask=2994.80))

        assert trade.closed, "SELL trade must close when ask <= TP"
        expected_pnl = round((trade.entry - trade.tp) * config.VOLUME * 100, 4)
        assert abs(trade.closed_pnl - expected_pnl) < TOLERANCE, (
            f"SELL TP PnL {trade.closed_pnl} should equal {expected_pnl}"
        )
        assert trade.closed_pnl > 0, "TP close must be profitable"

    # ── No premature close ─────────────────────────────────────────────────────

    def test_no_close_between_sl_and_tp(self):
        """Trade must NOT close while price stays between SL and TP."""
        mgr = OrderManager()
        trade = run(mgr.enter(Signal.BUY, ask=3000.50, bid=3000.20))
        assert trade is not None

        # Price moves around but never crosses SL (2999.50) or TP (3005.50)
        for bid in [3000.10, 2999.60, 3001.00, 3004.90, 3002.00]:
            run(mgr.check_paper_exits(bid=bid, ask=bid + 0.30))
            assert not trade.closed, (
                f"Trade must stay open at bid={bid} (SL={trade.sl}, TP={trade.tp})"
            )

    # ── Spread guard ──────────────────────────────────────────────────────────

    def test_entry_rejected_when_spread_equals_sl(self):
        """Entry must be rejected when spread = SL_POINTS (cost = entire SL)."""
        mgr = OrderManager()
        # spread = ask - bid = 3001.00 - 3000.00 = 1.0 = SL_POINTS
        trade = run(mgr.enter(Signal.BUY, ask=3001.00, bid=3000.00))
        assert trade is None, "Must reject when spread >= SL_POINTS"

    def test_entry_rejected_when_spread_exceeds_sl(self):
        """Entry must be rejected when spread > SL_POINTS."""
        mgr = OrderManager()
        trade = run(mgr.enter(Signal.BUY, ask=3001.50, bid=3000.00))
        assert trade is None, "Must reject when spread > SL_POINTS"

    def test_entry_accepted_when_spread_below_sl(self):
        """Entry must be accepted when spread < SL_POINTS."""
        mgr = OrderManager()
        # spread = 0.50 < 1.0
        trade = run(mgr.enter(Signal.BUY, ask=3000.70, bid=3000.20))
        assert trade is not None, "Must accept when spread < SL_POINTS"

    # ── Daily scoping (BUG-3) ─────────────────────────────────────────────────

    def test_total_today_excludes_yesterday(self):
        """total_today must ignore trades opened before today (UTC)."""
        mgr = OrderManager()

        # Inject a synthetic "yesterday" trade (opened 25h ago)
        yesterday = OpenTrade(
            ticket=99901,
            direction=Signal.BUY,
            entry=3000.0, sl=2999.0, tp=3005.0, volume=0.01,
            opened_at=time.time() - 90_000,   # 25 hours ago
        )
        mgr.trades.append(yesterday)

        # Open a trade today
        run(mgr.enter(Signal.BUY, ask=3000.50, bid=3000.20))

        assert mgr.total_today == 1, (
            f"total_today={mgr.total_today}, expected 1 (yesterday's trade must be excluded)"
        )

    def test_realized_pnl_excludes_yesterday(self):
        """realized_pnl must exclude closed trades from before today (UTC)."""
        mgr = OrderManager()

        # Inject a closed "yesterday" trade with a large loss
        yesterday = OpenTrade(
            ticket=99902,
            direction=Signal.BUY,
            entry=3000.0, sl=2999.0, tp=3005.0, volume=0.01,
            opened_at=time.time() - 90_000,
            closed=True,
            closed_pnl=-100.0,
        )
        mgr.trades.append(yesterday)

        # Open and close a winning trade today
        trade = run(mgr.enter(Signal.BUY, ask=3000.50, bid=3000.20))
        assert trade is not None
        run(mgr.check_paper_exits(bid=3006.00, ask=3006.30))  # TP hit
        assert trade.closed

        pnl = mgr.realized_pnl
        assert pnl > 0, (
            f"realized_pnl={pnl}, should be positive (yesterday's -$100 must be excluded)"
        )
        assert pnl < 10, f"realized_pnl={pnl} is unexpectedly large"

    # ── Double-close guard ────────────────────────────────────────────────────

    def test_trade_not_closed_twice(self):
        """A closed trade must not be re-closed and PnL must not change."""
        mgr = OrderManager()
        trade = run(mgr.enter(Signal.BUY, ask=3000.50, bid=3000.20))
        assert trade is not None

        # Hit SL
        run(mgr.check_paper_exits(bid=2999.00, ask=2999.30))
        assert trade.closed
        first_pnl = trade.closed_pnl

        # Call again with a different price — PnL must not change
        run(mgr.check_paper_exits(bid=2990.00, ask=2990.30))
        assert trade.closed_pnl == first_pnl, (
            f"PnL changed after double-close: {first_pnl} -> {trade.closed_pnl}"
        )


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    suite  = TestSLIntegrity()
    tests  = sorted(m for m in dir(suite) if m.startswith("test_"))
    passed = 0
    failed = 0

    print("=" * 60)
    print("SL Integrity Tests — Phase 1 Validation")
    print(f"Config: SL={config.SL_POINTS}pt  TP={config.TP_POINTS}pt  "
          f"VOL={config.VOLUME}  MAX_LOSS=${MAX_LOSS:.2f}")
    print("=" * 60)

    for name in tests:
        try:
            getattr(suite, name)()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed  "
          f"({'ALL PASS' if failed == 0 else 'FAILURES DETECTED'})")
    print("=" * 60)

    _loop.close()
    sys.exit(0 if failed == 0 else 1)
