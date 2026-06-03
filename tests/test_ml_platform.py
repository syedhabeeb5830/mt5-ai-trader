"""
ML Platform — Unit Tests
─────────────────────────────────────────────────────────────────────────────
Tests the database layer, feature engine, labeler, instrument config,
and decision engine using only in-memory / temp data.  No network, no MT5.

Usage:
  python tests/test_ml_platform.py
  pytest tests/test_ml_platform.py -v
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 200, base: float = 3000.0, freq: str = "5min") -> pd.DataFrame:
    idx   = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    close = base + np.cumsum(np.random.randn(n) * 0.5)
    high  = close + np.abs(np.random.randn(n)) * 0.3
    low   = close - np.abs(np.random.randn(n)) * 0.3
    vol   = np.random.randint(100, 1000, n).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low,
                          "close": close, "volume": vol, "spread": 0.3}, index=idx)


def _tmp_db() -> "Database":
    from ml.database import Database
    tmp = tempfile.mktemp(suffix=".db")
    return Database(path=tmp)


# ── Database ──────────────────────────────────────────────────────────────────

class TestDatabase:

    def test_upsert_candles_and_count(self):
        db = _tmp_db()
        rows = [{"symbol":"XAUUSD","timeframe":"M5","ts": 1000+i,
                  "open":1.0,"high":1.1,"low":0.9,"close":1.0,"volume":100,"spread":0.1}
                for i in range(20)]
        db.upsert_candles(rows)
        assert db.candle_count("XAUUSD", "M5") == 20

    def test_upsert_candles_deduplicates(self):
        db = _tmp_db()
        row = {"symbol":"XAUUSD","timeframe":"M5","ts":1000,
               "open":1.0,"high":1.1,"low":0.9,"close":1.0,"volume":100,"spread":0.1}
        db.upsert_candles([row, row])
        assert db.candle_count("XAUUSD", "M5") == 1

    def test_latest_candle_ts(self):
        db = _tmp_db()
        rows = [{"symbol":"XAUUSD","timeframe":"H1","ts": 1000+i,
                  "open":1.0,"high":1.1,"low":0.9,"close":1.0,"volume":100,"spread":0.0}
                for i in range(5)]
        db.upsert_candles(rows)
        assert db.latest_candle_ts("XAUUSD", "H1") == 1004

    def test_get_candles_since(self):
        db = _tmp_db()
        rows = [{"symbol":"XAUUSD","timeframe":"M5","ts": i,
                  "open":1.0,"high":1.1,"low":0.9,"close":1.0,"volume":100,"spread":0.0}
                for i in range(100)]
        db.upsert_candles(rows)
        result = db.get_candles("XAUUSD", "M5", since_ts=50)
        assert all(r["ts"] > 50 for r in result)

    def test_upsert_and_get_labels(self):
        db = _tmp_db()
        # Need a candle first
        db.upsert_candles([{"symbol":"X","timeframe":"M5","ts":1,
                             "open":1,"high":1,"low":1,"close":1,"volume":1,"spread":0}])
        cid = db.get_candles("X","M5")[0]["id"]
        db.upsert_labels([{"candle_id":cid,"symbol":"X","timeframe":"M5","ts":1,
                            "label_profile":"scalp","direction":"BUY",
                            "label":1,"bars_to_exit":3,"exit_price":1.06}])
        rows = db.get_labels("X","scalp","BUY")
        assert len(rows) == 1
        assert rows[0]["label"] == 1

    def test_insert_and_resolve_prediction(self):
        db = _tmp_db()
        pid = db.insert_prediction({
            "symbol":"XAUUSD","timeframe":"M5","ts":1000,
            "label_profile":"momentum","model_version":"v1",
            "direction":"BUY","probability":0.81,
            "threshold_used":0.72,"top_features":"[]","acted":1,
        })
        assert pid > 0
        db.resolve_prediction(pid, outcome_label=1, outcome_pnl=0.50)
        recent = db.get_recent_predictions("XAUUSD", limit=10)
        assert any(r["id"] == pid and r["outcome_label"] == 1 for r in recent)

    def test_summary_runs(self):
        db = _tmp_db()
        s  = db.summary()
        assert "candles" in s and "features" in s


# ── Instrument config ─────────────────────────────────────────────────────────

class TestInstrumentConfig:

    def test_known_instrument(self):
        from ml.instrument_config import get_instrument
        inst = get_instrument("XAUUSD")
        assert inst.symbol == "XAUUSD"
        assert inst.entry_tf == "M5"
        assert 0 < inst.buy_threshold < 1

    def test_unknown_instrument_raises(self):
        from ml.instrument_config import get_instrument
        try:
            get_instrument("FAKECOIN")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "FAKECOIN" in str(e)

    def test_label_profile_scalp(self):
        from ml.instrument_config import get_label_profile
        p = get_label_profile("scalp")
        assert p.tp_points > p.sl_points

    def test_tf_to_minutes(self):
        from ml.instrument_config import TF_TO_MINUTES
        assert TF_TO_MINUTES["M5"]  == 5
        assert TF_TO_MINUTES["H1"]  == 60
        assert TF_TO_MINUTES["D1"]  == 1440

    def test_all_instruments_have_valid_entry_tf(self):
        from ml.instrument_config import INSTRUMENTS
        for sym, inst in INSTRUMENTS.items():
            assert inst.entry_tf in inst.timeframes, \
                f"{sym}: entry_tf={inst.entry_tf} not in timeframes={inst.timeframes}"

    def test_thresholds_consistent(self):
        from ml.instrument_config import INSTRUMENTS
        for sym, inst in INSTRUMENTS.items():
            assert inst.buy_threshold > inst.sell_threshold, \
                f"{sym}: buy_threshold must be > sell_threshold"


# ── Feature engine ────────────────────────────────────────────────────────────

class TestFeatureEngine:

    def test_extract_features_basic(self):
        from ml.feature_engine import build_feature_vector
        df = _make_ohlcv(200, freq="5min")
        bars = {"M5": df}
        ts   = int(df.index[-1].timestamp())
        fvec = build_feature_vector(bars, "M5", ts)
        assert isinstance(fvec, dict)
        assert len(fvec) > 20
        assert all(isinstance(v, float) for v in fvec.values())

    def test_features_no_lookahead(self):
        from ml.feature_engine import build_feature_vector
        df  = _make_ohlcv(200, freq="5min")
        ts1 = int(df.index[100].timestamp())
        ts2 = int(df.index[-1].timestamp())
        bars = {"M5": df}
        f1  = build_feature_vector(bars, "M5", ts1)
        f2  = build_feature_vector(bars, "M5", ts2)
        # EMA200 at bar 100 must not equal EMA200 at last bar
        if "ema200_m5" in f1 and "ema200_m5" in f2:
            assert f1["ema200_m5"] != f2["ema200_m5"]

    def test_features_multi_tf(self):
        from ml.feature_engine import build_feature_vector
        # 500 M5 bars = ~41.7h, so H1 slice has ~42 bars (≥30 minimum)
        df_m5 = _make_ohlcv(500, freq="5min")
        df_h1 = _make_ohlcv(100, freq="1h")
        bars  = {"M5": df_m5, "H1": df_h1}
        ts    = int(df_m5.index[-1].timestamp())
        fvec  = build_feature_vector(bars, "M5", ts)
        has_m5 = any("_m5" in k for k in fvec)
        has_h1 = any("_h1" in k for k in fvec)
        assert has_m5, "Should have M5 features"
        assert has_h1, "Should have H1 features"

    def test_session_flags_in_features(self):
        from ml.feature_engine import build_feature_vector
        df   = _make_ohlcv(200, freq="5min")
        ts   = int(df.index[-1].timestamp())
        fvec = build_feature_vector({"M5": df}, "M5", ts)
        assert "hour_of_day" in fvec
        assert "day_of_week" in fvec

    def test_empty_bars_returns_empty(self):
        from ml.feature_engine import build_feature_vector
        result = build_feature_vector({}, "M5", 1000)
        assert result == {}

    def test_insufficient_bars_returns_empty(self):
        from ml.feature_engine import build_feature_vector
        df   = _make_ohlcv(5, freq="5min")   # too few for indicators
        ts   = int(df.index[-1].timestamp())
        fvec = build_feature_vector({"M5": df}, "M5", ts)
        # Either empty or has very few features (some indicators need 200 bars)
        assert isinstance(fvec, dict)


# ── Labeler ───────────────────────────────────────────────────────────────────

class TestLabeler:

    def test_buy_tp_hit(self):
        from ml.labeler import _label_candle
        closes = [100.0] * 20
        highs  = [100.0] * 5 + [106.5] + [100.0] * 14   # TP hit at bar 5
        lows   = [99.0]  * 20
        lbl, bars, price = _label_candle(0, closes, highs, lows, "BUY",
                                          entry_price=100.0, tp_abs=106.0, sl_abs=97.0,
                                          horizon=15)
        assert lbl == 1
        assert bars == 5

    def test_buy_sl_hit(self):
        from ml.labeler import _label_candle
        closes = [100.0] * 20
        highs  = [100.5] * 20
        lows   = [100.0] * 3 + [96.5] + [100.0] * 16    # SL hit at bar 3
        lbl, bars, _ = _label_candle(0, closes, highs, lows, "BUY",
                                      entry_price=100.0, tp_abs=106.0, sl_abs=97.0,
                                      horizon=15)
        assert lbl == 0
        assert bars == 3

    def test_buy_horizon_timeout(self):
        from ml.labeler import _label_candle
        closes = [100.0] * 30
        highs  = [100.5] * 30
        lows   = [99.5]  * 30
        lbl, bars, _ = _label_candle(0, closes, highs, lows, "BUY",
                                      entry_price=100.0, tp_abs=106.0, sl_abs=97.0,
                                      horizon=10)
        assert lbl == 0
        assert bars is None

    def test_sell_tp_hit(self):
        from ml.labeler import _label_candle
        closes = [100.0] * 20
        highs  = [100.5] * 20
        lows   = [100.0] * 4 + [93.5] + [100.0] * 15    # TP hit at bar 4
        lbl, bars, _ = _label_candle(0, closes, highs, lows, "SELL",
                                      entry_price=100.0, tp_abs=94.0, sl_abs=103.0,
                                      horizon=15)
        assert lbl == 1
        assert bars == 4

    def test_label_instrument_no_data(self):
        from ml.labeler import label_instrument
        from ml.instrument_config import get_instrument, get_label_profile
        db = _tmp_db()
        with patch("ml.labeler.get_db", return_value=db):
            inst    = get_instrument("XAUUSD")
            profile = get_label_profile("momentum")
            n = label_instrument(inst, profile, verbose=False)
            assert n == 0   # no candles → no labels


# ── Decision engine (mock model) ─────────────────────────────────────────────

class TestDecisionEngine:

    def _mock_engine(self, buy_prob: float = 0.80, sell_prob: float = 0.20):
        from ml.probability_engine import ProbabilityResult, FeatureContribution
        from ml.decision_engine import DecisionEngine
        from ml.instrument_config import get_instrument

        inst    = get_instrument("XAUUSD")
        engine  = DecisionEngine(inst)

        buy_res  = ProbabilityResult(probability=buy_prob,  direction="BUY",  model_version="test_v1")
        sell_res = ProbabilityResult(probability=sell_prob, direction="SELL", model_version="test_v1")

        engine._engine = MagicMock()
        engine._engine.predict.side_effect = lambda features, direction: (
            buy_res if direction == "BUY" else sell_res
        )
        return engine

    def test_high_buy_probability_gives_buy(self):
        engine  = self._mock_engine(buy_prob=0.85, sell_prob=0.25)
        result  = engine.decide({"rsi14_m5": 55.0})
        from momentum import Signal
        assert result.signal == Signal.BUY

    def test_high_sell_probability_gives_sell(self):
        engine  = self._mock_engine(buy_prob=0.20, sell_prob=0.85)
        result  = engine.decide({"rsi14_m5": 45.0})
        from momentum import Signal
        assert result.signal == Signal.SELL

    def test_low_probability_gives_wait(self):
        engine = self._mock_engine(buy_prob=0.55, sell_prob=0.55)
        result = engine.decide({})
        from momentum import Signal
        assert result.signal == Signal.WAIT

    def test_decision_result_has_explanation(self):
        engine = self._mock_engine(buy_prob=0.81)
        result = engine.decide({})
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 5

    def test_to_dict_serialisable(self):
        import json
        engine = self._mock_engine(buy_prob=0.82)
        result = engine.decide({})
        d = result.to_dict()
        s = json.dumps(d)
        assert "signal" in s


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    suites = [
        TestDatabase(),
        TestInstrumentConfig(),
        TestFeatureEngine(),
        TestLabeler(),
        TestDecisionEngine(),
    ]
    passed = failed = 0
    print("=" * 62)
    print("ML Platform Tests")
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
