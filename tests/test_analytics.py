"""
Analytics Engine Unit Tests — Phase 2 Validation
─────────────────────────────────────────────────────────────────────────────
Tests all core metric computations with known data.
No I/O, no MT5, no network required.

Usage:
  python tests/test_analytics.py          # standalone
  pytest tests/test_analytics.py -v       # via pytest
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.engine import (
    TradeRecord, analyze,
    _median, _max_drawdown, _consecutive_streaks, _sharpe, _sortino,
)

TOLERANCE = 1e-6


def _make_trade(pnl: float, hour: int = 10, day_offset: int = 0) -> TradeRecord:
    """Helper: create a trade record with controllable timing."""
    base = datetime(2026, 1, 2, hour, 0, 0, tzinfo=timezone.utc)
    opened = base + timedelta(days=day_offset)
    closed = opened + timedelta(minutes=30)
    return TradeRecord(pnl=pnl, direction="BUY", entry=3000.0,
                       sl=2999.0, tp=3005.0,
                       opened_at=opened, closed_at=closed)


# ── Core function tests ───────────────────────────────────────────────────────

class TestCoreMetrics:

    def test_median_odd(self):
        assert _median([1.0, 3.0, 5.0]) == 3.0

    def test_median_even(self):
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_median_single(self):
        assert _median([7.5]) == 7.5

    def test_median_empty(self):
        assert _median([]) == 0.0

    def test_max_drawdown_basic(self):
        # equity: 5, 10, 4, 7, 3 → peak=10, dd from 10→3 = 7
        pnls = [5.0, 5.0, -6.0, 3.0, -4.0]
        dd = _max_drawdown(pnls)
        assert dd == 7.0, f"Expected 7.0, got {dd}"

    def test_max_drawdown_no_loss(self):
        assert _max_drawdown([1.0, 2.0, 3.0]) == 0.0

    def test_max_drawdown_all_loss(self):
        # equity goes -1, -2, -3 — peak stays 0 so dd=3
        assert _max_drawdown([-1.0, -1.0, -1.0]) == 3.0

    def test_max_drawdown_empty(self):
        assert _max_drawdown([]) == 0.0

    def test_consecutive_streaks_basic(self):
        # W L W W W L L L L
        wins = [True, False, True, True, True, False, False, False, False]
        max_w, max_l = _consecutive_streaks(wins)
        assert max_w == 3, f"max_w={max_w}"
        assert max_l == 4, f"max_l={max_l}"

    def test_consecutive_streaks_all_wins(self):
        max_w, max_l = _consecutive_streaks([True, True, True])
        assert max_w == 3
        assert max_l == 0

    def test_consecutive_streaks_empty(self):
        assert _consecutive_streaks([]) == (0, 0)

    def test_sharpe_positive_returns(self):
        # All positive equal daily PnL → Sharpe should be inf or very large
        daily = [1.0] * 20
        s = _sharpe(daily)
        # std=0 → returns None
        assert s is None

    def test_sharpe_mixed_returns(self):
        daily = [1.0, -0.5, 2.0, -1.0, 0.5, 1.5, -0.3, 0.8, 1.2, -0.4,
                 0.9, -0.6, 1.1, 0.3, -0.2, 1.4, 0.7, -0.8, 0.6, 0.4]
        s = _sharpe(daily)
        assert s is not None
        assert isinstance(s, float)
        assert not math.isnan(s)

    def test_sharpe_insufficient_data(self):
        assert _sharpe([1.0, 2.0]) is None

    def test_sortino_positive_mean_no_downside(self):
        daily = [1.0, 2.0, 3.0, 1.5, 2.5, 1.0, 2.0, 0.5, 1.0, 2.0, 1.5]
        # Mean > 0, no downside days → None (insufficient downside)
        s = _sortino(daily)
        assert s is None


# ── analyze() function tests ──────────────────────────────────────────────────

class TestAnalyze:

    def _known_trades(self):
        """Known set: 5 wins at $5.0, 8 losses at -$1.0 — all on different days."""
        trades = []
        day = 0
        for _ in range(5):
            trades.append(_make_trade(5.0, day_offset=day))
            day += 1
        for _ in range(8):
            trades.append(_make_trade(-1.0, day_offset=day))
            day += 1
        return trades

    def test_trade_counts(self):
        r = analyze(self._known_trades())
        assert r.total_trades   == 13
        assert r.winning_trades == 5
        assert r.losing_trades  == 8

    def test_win_rate(self):
        r = analyze(self._known_trades())
        expected = round(5 / 13 * 100, 2)
        assert abs(r.win_rate - expected) < 0.01, f"win_rate={r.win_rate}"

    def test_avg_win(self):
        r = analyze(self._known_trades())
        assert abs(r.avg_win - 5.0) < TOLERANCE

    def test_avg_loss(self):
        r = analyze(self._known_trades())
        assert abs(r.avg_loss - (-1.0)) < TOLERANCE

    def test_gross_profit(self):
        r = analyze(self._known_trades())
        assert abs(r.gross_profit - 25.0) < TOLERANCE

    def test_gross_loss(self):
        r = analyze(self._known_trades())
        assert abs(r.gross_loss - 8.0) < TOLERANCE

    def test_net_pnl(self):
        r = analyze(self._known_trades())
        assert abs(r.net_pnl - 17.0) < TOLERANCE

    def test_profit_factor(self):
        r = analyze(self._known_trades())
        expected = round(25.0 / 8.0, 3)
        assert abs(r.profit_factor - expected) < 0.001, f"pf={r.profit_factor}"

    def test_expectancy(self):
        r = analyze(self._known_trades())
        wr   = 5 / 13
        expected = round(wr * 5.0 + (1 - wr) * (-1.0), 4)
        assert abs(r.expectancy - expected) < 0.001, f"expectancy={r.expectancy}"

    def test_largest_win(self):
        r = analyze(self._known_trades())
        assert r.largest_win == 5.0

    def test_largest_loss(self):
        r = analyze(self._known_trades())
        assert r.largest_loss == -1.0

    def test_median_win(self):
        r = analyze(self._known_trades())
        assert r.median_win == 5.0

    def test_median_loss(self):
        r = analyze(self._known_trades())
        assert r.median_loss == -1.0

    def test_max_drawdown_nonzero(self):
        # 8 consecutive losses of -$1 = $8 max drawdown
        trades = [_make_trade(-1.0, day_offset=i) for i in range(8)]
        r = analyze(trades)
        assert r.max_drawdown == 8.0, f"max_drawdown={r.max_drawdown}"

    def test_max_drawdown_zero_for_all_wins(self):
        trades = [_make_trade(5.0, day_offset=i) for i in range(5)]
        r = analyze(trades)
        assert r.max_drawdown == 0.0

    def test_consecutive_streaks(self):
        # W W W L L L L L
        trades = (
            [_make_trade(5.0,  day_offset=i)     for i in range(3)] +
            [_make_trade(-1.0, day_offset=i + 3) for i in range(5)]
        )
        r = analyze(trades)
        assert r.max_consecutive_wins   == 3, f"max_w={r.max_consecutive_wins}"
        assert r.max_consecutive_losses == 5, f"max_l={r.max_consecutive_losses}"

    def test_holding_time(self):
        r = analyze(self._known_trades())
        # Each trade has 30-minute holding time
        assert r.avg_holding_sec is not None
        assert abs(r.avg_holding_sec - 1800.0) < 1.0

    def test_empty_input(self):
        r = analyze([])
        assert r.total_trades == 0
        assert "NO DATA" in r.verdict

    def test_all_losses(self):
        trades = [_make_trade(-1.0, day_offset=i) for i in range(10)]
        r = analyze(trades)
        assert r.winning_trades == 0
        assert r.profit_factor  == 0.0
        assert r.net_pnl < 0

    def test_all_wins(self):
        trades = [_make_trade(5.0, day_offset=i) for i in range(10)]
        r = analyze(trades)
        assert r.losing_trades == 0
        assert r.net_pnl > 0

    def test_by_hour_aggregation(self):
        # 5 trades at hour=9, 5 trades at hour=14
        t9  = [_make_trade(1.0,  hour=9,  day_offset=i)     for i in range(5)]
        t14 = [_make_trade(-1.0, hour=14, day_offset=i + 5) for i in range(5)]
        r   = analyze(t9 + t14)
        assert r.trades_by_hour.get(9,  0) == 5
        assert r.trades_by_hour.get(14, 0) == 5
        assert abs(r.pnl_by_hour.get(9,  0) - 5.0) < TOLERANCE
        assert abs(r.pnl_by_hour.get(14, 0) + 5.0) < TOLERANCE

    def test_monthly_aggregation(self):
        # 3 trades in Jan, 3 in Feb
        jan = [_make_trade(2.0, day_offset=i)      for i in range(3)]
        feb = [_make_trade(-1.0, day_offset=i + 31) for i in range(3)]
        r   = analyze(jan + feb)
        assert "2026-01" in r.monthly_pnl
        assert "2026-02" in r.monthly_pnl
        assert abs(r.monthly_pnl["2026-01"] - 6.0) < TOLERANCE
        assert abs(r.monthly_pnl["2026-02"] + 3.0) < TOLERANCE

    def test_verdict_profitable(self):
        # Many wins, high PF
        trades = (
            [_make_trade(5.0,  day_offset=i)      for i in range(40)] +
            [_make_trade(-1.0, day_offset=i + 40) for i in range(20)]
        )
        r = analyze(trades)
        assert r.profit_factor > 1.0
        assert "PROFITABLE" in r.verdict or "STRONG" in r.verdict

    def test_verdict_losing(self):
        trades = [_make_trade(-1.0, day_offset=i) for i in range(30)]
        r = analyze(trades)
        assert "LOSING" in r.verdict

    def test_to_text_contains_key_fields(self):
        r = analyze(self._known_trades())
        text = r.to_text()
        assert "Win Rate"      in text, "Win Rate missing"
        assert "Profit Factor" in text, "Profit Factor missing"
        assert "Expectancy"    in text, "Expectancy missing"
        assert "Max Drawdown"  in text, "Max Drawdown missing"
        assert "VERDICT"       in text, "VERDICT missing"

    def test_to_dict_serialisable(self):
        import json
        r = analyze(self._known_trades())
        d = r.to_dict()
        # Must serialise without error
        s = json.dumps(d)
        assert len(s) > 100


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    suites = [TestCoreMetrics(), TestAnalyze()]
    passed = failed = 0

    print("=" * 62)
    print("Analytics Engine Tests — Phase 2 Validation")
    print("=" * 62)

    for suite in suites:
        tests = sorted(m for m in dir(suite) if m.startswith("test_"))
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

    print("=" * 62)
    print(f"Results: {passed} passed, {failed} failed  "
          f"({'ALL PASS' if failed == 0 else 'FAILURES DETECTED'})")
    print("=" * 62)
    sys.exit(0 if failed == 0 else 1)
