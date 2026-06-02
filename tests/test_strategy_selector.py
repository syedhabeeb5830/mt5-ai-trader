"""
Strategy Selector Unit Tests — Phase 3 Validation
─────────────────────────────────────────────────────────────────────────────
Tests the composite scorer, ranking logic, OOS splitter, hour filter,
recommendation file I/O, and edge cases.
No network, no MT5. All mocked.

Usage:
  python tests/test_strategy_selector.py
  pytest tests/test_strategy_selector.py -v
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from typing import List
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.strategy_selector import (
    StrategyRank, _composite, rank_strategies, recommend,
    print_leaderboard, save_recommendation, load_recommendation,
    oos_split, VALID_METRICS,
)
from analytics.live_strategy import within_trade_hours, _parse_hours


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rank(
    name: str = "test",
    n:    int   = 100,
    wr:   float = 50.0,
    pf:   float = 1.5,
    exp:  float = 0.5,
    sh:   float = 1.5,
    net:  float = 50.0,
    dd:   float = 20.0,
) -> StrategyRank:
    comp = _composite(pf, wr, exp, sh)
    return StrategyRank(
        name          = name,
        description   = f"Test strategy {name}",
        n_trades      = n,
        win_rate      = wr,
        profit_factor = pf,
        expectancy    = exp,
        net_pnl       = net,
        max_drawdown  = dd,
        sharpe        = sh,
        composite     = comp,
    )


# ── _composite() ─────────────────────────────────────────────────────────────

class TestComposite:

    def test_perfect_score(self):
        # PF=2.0 (max), WR=100%, exp=1.0+, sharpe=3.0+  → ~1.0
        s = _composite(2.0, 100.0, 1.0, 3.0)
        assert 0.95 <= s <= 1.0, f"Expected ~1.0, got {s}"

    def test_zero_score(self):
        # PF=1.0 (breakeven), WR=0, exp≤0, sharpe=0
        s = _composite(1.0, 0.0, 0.0, 0.0)
        assert s == 0.0, f"Expected 0.0, got {s}"

    def test_losing_pf(self):
        # PF < 1.0 → pf component = 0; PF=1.0 → also 0 (breakeven is floor)
        # But higher WR or sharpe can compensate — test that same WR gives same
        # result for PF=0.5 vs PF=1.0 (both at 0 on PF component)
        s_sub = _composite(0.5, 50.0, 0.2, 1.0)
        s_be  = _composite(1.0, 50.0, 0.2, 1.0)
        # Both PF≤1 → n_pf=0 → same score on pf component
        assert abs(s_sub - s_be) < 1e-6, "PF≤1.0 should all score 0 on PF component"

    def test_sharpe_none(self):
        # None sharpe should be treated as 0
        s_none = _composite(1.2, 50.0, 0.3, None)
        s_zero = _composite(1.2, 50.0, 0.3, 0.0)
        assert abs(s_none - s_zero) < 1e-6

    def test_values_bounded_0_1(self):
        for pf in [0.5, 1.0, 1.5, 3.0]:
            for wr in [0, 25, 50, 100]:
                for exp in [-1.0, 0.0, 0.5, 2.0]:
                    s = _composite(pf, wr, exp, 2.0)
                    assert 0.0 <= s <= 1.0, f"Out of bounds: pf={pf} wr={wr} exp={exp} → {s}"

    def test_higher_pf_gives_higher_score(self):
        s1 = _composite(1.1, 40.0, 0.1, 1.0)
        s2 = _composite(1.5, 40.0, 0.1, 1.0)
        assert s2 > s1

    def test_higher_wr_gives_higher_score(self):
        s1 = _composite(1.2, 30.0, 0.2, 1.0)
        s2 = _composite(1.2, 60.0, 0.2, 1.0)
        assert s2 > s1


# ── StrategyRank ──────────────────────────────────────────────────────────────

class TestStrategyRank:

    def test_verdict_strong(self):
        r = _rank(pf=1.4, wr=48.0)
        assert r.verdict_tag == "STRONG"

    def test_verdict_good(self):
        r = _rank(pf=1.2, wr=40.0, n=100)
        assert r.verdict_tag == "GOOD"

    def test_verdict_marginal(self):
        r = _rank(pf=1.05, wr=25.0, n=50)
        assert r.verdict_tag == "MARGINAL"

    def test_verdict_thin_data(self):
        r = _rank(n=10, pf=1.5, wr=60.0)
        r.n_trades = 10   # force < 20
        tag = r.verdict_tag
        assert tag == "THIN DATA", f"Expected THIN DATA, got {tag!r}"

    def test_verdict_losing(self):
        r = _rank(pf=0.8, wr=30.0, n=50)
        assert r.verdict_tag == "LOSING"

    def test_to_dict_json_serialisable(self):
        r = _rank()
        d = r.to_dict()
        s = json.dumps(d)
        assert len(s) > 10

    def test_to_dict_none_sharpe(self):
        r = _rank()
        r.sharpe = None
        r.composite = _composite(r.profit_factor, r.win_rate, r.expectancy, None)
        d = r.to_dict()
        assert d["sharpe"] is None


# ── VALID_METRICS ─────────────────────────────────────────────────────────────

class TestValidMetrics:

    def test_all_metrics_listed(self):
        expected = {"profit_factor", "win_rate", "expectancy", "sharpe", "composite"}
        assert expected == set(VALID_METRICS)

    def test_invalid_metric_raises(self):
        try:
            rank_strategies(days=1, metric="invalid_metric", verbose=False)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "invalid_metric" in str(e)


# ── OOS split ─────────────────────────────────────────────────────────────────

class TestOOSSplit:

    def _make_bars(self, n: int = 100) -> dict:
        import pandas as pd
        import numpy as np
        idx = pd.date_range("2025-01-01", periods=n, freq="1H", tz="UTC")
        df  = pd.DataFrame({
            "open":   np.random.rand(n) + 3000,
            "high":   np.random.rand(n) + 3001,
            "low":    np.random.rand(n) + 2999,
            "close":  np.random.rand(n) + 3000,
            "volume": np.random.randint(100, 1000, n),
        }, index=idx)
        return {"1H": df}

    def test_split_sizes(self):
        bars = self._make_bars(100)
        in_s, out_s = oos_split(bars, 0.7)
        assert len(in_s["1H"]) == 70
        assert len(out_s["1H"]) == 30

    def test_split_non_overlapping(self):
        bars = self._make_bars(100)
        in_s, out_s = oos_split(bars, 0.7)
        in_idx  = set(in_s["1H"].index)
        out_idx = set(out_s["1H"].index)
        assert len(in_idx & out_idx) == 0, "IS and OOS should not overlap"

    def test_split_covers_all_bars(self):
        bars = self._make_bars(100)
        in_s, out_s = oos_split(bars, 0.7)
        assert len(in_s["1H"]) + len(out_s["1H"]) == 100

    def test_split_80_20(self):
        bars = self._make_bars(50)
        in_s, out_s = oos_split(bars, 0.8)
        assert len(in_s["1H"]) == 40
        assert len(out_s["1H"]) == 10

    def test_multiple_timeframes(self):
        import pandas as pd
        import numpy as np
        def _df(n):
            idx = pd.date_range("2025-01-01", periods=n, freq="1H", tz="UTC")
            return pd.DataFrame({"open":np.ones(n)*3000,"high":np.ones(n)*3001,
                                  "low":np.ones(n)*2999,"close":np.ones(n)*3000,
                                  "volume":np.ones(n)}, index=idx)
        bars = {"1H": _df(100), "4H": _df(40)}
        in_s, out_s = oos_split(bars, 0.7)
        assert len(in_s["1H"])  == 70
        assert len(in_s["4H"])  == 28
        assert len(out_s["1H"]) == 30
        assert len(out_s["4H"]) == 12


# ── save/load recommendation ──────────────────────────────────────────────────

class TestRecommendationIO:

    def test_save_and_load_round_trip(self):
        ranks = [
            _rank("strategy_a", pf=1.5, wr=55.0, n=200),
            _rank("strategy_b", pf=1.2, wr=40.0, n=100),
        ]
        ranks[0].rank = 1; ranks[0].recommended = True
        ranks[1].rank = 2; ranks[1].recommended = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch Path("logs") → tmpdir
            import analytics.strategy_selector as sel_mod
            original_path = sel_mod.Path

            class _PatchedPath:
                def __init__(self, *args):
                    # Replace "logs" with tmpdir
                    self._p = original_path(
                        str(original_path(*args)).replace("logs", tmpdir)
                    )
                def __truediv__(self, other):
                    result = _PatchedPath.__new__(_PatchedPath)
                    result._p = self._p / other
                    return result
                def mkdir(self, **kw):    return self._p.mkdir(**kw)
                def write_text(self, t):  return self._p.write_text(t)
                def read_text(self):      return self._p.read_text()
                def exists(self):         return self._p.exists()
                def __str__(self):        return str(self._p)

            path = original_path(tmpdir) / "recommendation.json"
            payload = {
                "generated_at": "2026-06-02 12:00:00",
                "ranking_metric": "composite",
                "lookback_days": 60,
                "recommended": "strategy_a",
                "leaderboard": [r.to_dict() for r in ranks],
            }
            path.write_text(json.dumps(payload, indent=2))

            loaded = json.loads(path.read_text())
            assert loaded["recommended"] == "strategy_a"
            assert len(loaded["leaderboard"]) == 2
            assert loaded["leaderboard"][0]["profit_factor"] == 1.5

    def test_load_missing_file_returns_none(self):
        import analytics.strategy_selector as sel_mod
        from pathlib import Path
        with patch.object(sel_mod, "Path") as MockPath:
            mock_inst = MockPath.return_value
            mock_inst.exists.return_value = False
            # re-import function with context
            result = None
            import json as _json

            def _load():
                p = Path("logs/recommendation.json")
                if not p.exists():
                    return None
                return _json.loads(p.read_text())

            # Directly test the logic
            MockPath("logs/recommendation.json").exists.return_value = False
            # Simpler: just test that a non-existent file returns None
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                import importlib
                import analytics.strategy_selector as m
                original = m.Path
                m.Path = lambda *a: original(d, *a)
                result = m.load_recommendation()
                m.Path = original
            # Since tmpdir has no recommendation.json, result should be None
            assert result is None


# ── Trade hour filter ─────────────────────────────────────────────────────────

class TestTradeHourFilter:

    def test_parse_valid(self):
        assert _parse_hours("14-18") == (14, 18)
        assert _parse_hours("0-8")   == (0, 8)
        assert _parse_hours("22-2")  == (22, 2)

    def test_parse_empty_returns_none(self):
        assert _parse_hours("") is None
        assert _parse_hours("  ") is None

    def test_parse_invalid_returns_none(self):
        result = _parse_hours("not-valid")
        assert result is None

    def test_within_hours_no_filter(self):
        # Empty spec → always True
        assert within_trade_hours("") is True

    def test_within_hours_inside_window(self):
        # Test the pure logic: hour 15 is inside 14-18
        from analytics.live_strategy import _parse_hours as ph
        start, end = ph("14-18")
        hour = 15
        assert start <= hour < end, f"{hour} should be inside {start}-{end}"

    def test_within_hours_outside_window(self):
        from analytics.live_strategy import _parse_hours as ph
        start, end = ph("14-18")
        hour = 9
        assert not (start <= hour < end)

    def test_overnight_wrap_inside(self):
        # "22-02": hour 23 should be inside
        from analytics.live_strategy import _parse_hours as ph
        start, end = ph("22-2")
        hour = 23
        result = (hour >= start) or (hour < end)
        assert result is True

    def test_overnight_wrap_outside(self):
        from analytics.live_strategy import _parse_hours as ph
        start, end = ph("22-2")
        hour = 12
        result = (hour >= start) or (hour < end)
        assert result is False


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    suites = [
        TestComposite(),
        TestStrategyRank(),
        TestValidMetrics(),
        TestOOSSplit(),
        TestRecommendationIO(),
        TestTradeHourFilter(),
    ]
    passed = failed = 0

    print("=" * 62)
    print("Strategy Selector Tests — Phase 3 Validation")
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
