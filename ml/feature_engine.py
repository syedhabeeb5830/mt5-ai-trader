"""
ML Platform — Feature Engine
─────────────────────────────────────────────────────────────────────────────
Builds a flat feature vector for every candle in the entry timeframe.
Features are drawn from multiple timeframes simultaneously (multi-TF context).

Feature Registry:
  Features are registered as callables in FEATURE_REGISTRY.
  Adding a new feature requires only appending to that dict — nothing else changes.

Output:
  One dict of floats per candle, e.g.:
  {
    "ema9_m5":      3247.30,
    "rsi14_m5":     54.2,
    "atr14_m5":     2.1,
    "ema50_h1":     3240.0,
    "rsi14_h1":     61.0,
    "session_london": 1.0,
    ...
  }

All feature names are namespaced by timeframe (e.g. `_m5`, `_h1`) so
multi-TF features never collide in the vector.

Usage:
  python -m ml.feature_engine                        # active symbol
  python -m ml.feature_engine --symbol EURUSD
  python -m ml.feature_engine --symbol XAUUSD --all  # recompute all bars
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
import numpy as np

from ml.database import get_db
from ml.instrument_config import (
    INSTRUMENTS, InstrumentConfig, get_instrument, get_label_profile,
    TF_TO_MINUTES as _TF_TO_MINUTES,
)


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def _macd(s: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    fast_e = _ema(s, fast)
    slow_e = _ema(s, slow)
    line   = fast_e - slow_e
    signal = _ema(line, sig)
    hist   = line - signal
    return line, signal, hist

def _rolling_vol(s: pd.Series, p: int = 20) -> pd.Series:
    return s.pct_change().rolling(p).std() * math.sqrt(252 * 1440)

def _atr_percentile(atr: pd.Series, window: int = 100) -> pd.Series:
    return atr.rolling(window, min_periods=20).rank(pct=True) * 100


# ── Session classifier ────────────────────────────────────────────────────────

def _session_flags(ts_utc: int) -> dict[str, float]:
    """Return 0/1 flags for Asian/London/NY session overlap."""
    hour = datetime.fromtimestamp(ts_utc, tz=timezone.utc).hour
    return {
        "session_asian":  float(0 <= hour < 9),
        "session_london": float(8 <= hour < 16),
        "session_ny":     float(13 <= hour < 22),
        "hour_of_day":    float(hour),
        "day_of_week":    float(datetime.fromtimestamp(ts_utc, tz=timezone.utc).weekday()),
    }


# ── Per-timeframe feature extractor ──────────────────────────────────────────

def _extract_tf_features(df: pd.DataFrame, tf_label: str) -> dict[str, float]:
    """
    Compute all indicators for a single timeframe.
    Returns flat dict with keys namespaced like "ema9_m5", "rsi14_h1", etc.
    """
    sfx = f"_{tf_label.lower()}"  # e.g. "_m5"
    c   = df["close"]
    h   = df["high"]
    l   = df["low"]
    v   = df["volume"]

    ema9   = _ema(c, 9)
    ema20  = _ema(c, 20)
    ema50  = _ema(c, 50)
    ema200 = _ema(c, 200)
    rsi14  = _rsi(c, 14)
    rsi7   = _rsi(c, 7)
    macd_l, macd_s, macd_h = _macd(c)
    atr14  = _atr(df, 14)
    atr_p  = _atr_percentile(atr14)
    rvol20 = _rolling_vol(c, 20)
    vol20  = v.rolling(20).mean()
    mom5   = c - c.shift(5)
    mom10  = c - c.shift(10)

    last   = c.iloc[-1]
    feat: dict[str, float] = {}

    def _f(name: str, series: pd.Series) -> None:
        val = series.iloc[-1]
        feat[f"{name}{sfx}"] = float(val) if pd.notna(val) else 0.0

    _f("ema9",          ema9)
    _f("ema20",         ema20)
    _f("ema50",         ema50)
    _f("ema200",        ema200)
    _f("rsi14",         rsi14)
    _f("rsi7",          rsi7)
    _f("macd_line",     macd_l)
    _f("macd_signal",   macd_s)
    _f("macd_hist",     macd_h)
    _f("atr14",         atr14)
    _f("atr_pct",       atr_p)
    _f("rolling_vol",   rvol20)
    _f("rel_volume",    v / vol20.replace(0, np.nan))
    _f("mom5",          mom5)
    _f("mom10",         mom10)

    # Price distance from EMAs (normalised by ATR)
    atr_val = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) and atr14.iloc[-1] != 0 else 1.0
    feat[f"dist_ema20{sfx}"]  = (last - float(ema20.iloc[-1]))  / atr_val
    feat[f"dist_ema50{sfx}"]  = (last - float(ema50.iloc[-1]))  / atr_val
    feat[f"dist_ema200{sfx}"] = (last - float(ema200.iloc[-1])) / atr_val

    # Candle body / range
    bar_range = float(h.iloc[-1] - l.iloc[-1])
    bar_body  = abs(float(c.iloc[-1]) - float(df["open"].iloc[-1]))
    feat[f"candle_range{sfx}"] = bar_range
    feat[f"candle_body{sfx}"]  = bar_body
    feat[f"body_ratio{sfx}"]   = (bar_body / bar_range) if bar_range > 0 else 0.0

    # Market regime: trending (1) vs ranging (0) based on ADX proxy
    ema50_slope = (float(ema50.iloc[-1]) - float(ema50.iloc[-5])) / atr_val if len(df) > 5 else 0.0
    feat[f"ema50_slope{sfx}"] = ema50_slope
    feat[f"trending{sfx}"]    = float(abs(ema50_slope) > 0.5)

    return feat


# ── Multi-timeframe feature builder ──────────────────────────────────────────

def build_feature_vector(
    bars: dict[str, pd.DataFrame],
    entry_tf: str,
    ts_utc: int,
) -> dict[str, float]:
    """
    Build the full feature vector for a single entry candle.

    bars     : {tf -> OHLCV DataFrame, index=DatetimeTZDtype(UTC)}
    entry_tf : primary entry timeframe key (e.g. "M5")
    ts_utc   : Unix timestamp of the entry candle's open

    Returns a flat dict of float features, or empty dict if insufficient data.
    """
    feat: dict[str, float] = {}

    for tf, df in bars.items():
        if df is None or len(df) < 30:
            continue
        # Slice to only bars UP TO (and including) ts_utc — prevent lookahead
        df_slice = df[df.index.astype("int64") // 10**9 <= ts_utc]
        if len(df_slice) < 30:
            continue
        feat.update(_extract_tf_features(df_slice, tf))

    if not feat:
        return {}

    # Add time features from entry bar
    feat.update(_session_flags(ts_utc))
    return feat


# ── Batch feature computation ──────────────────────────────────────────────────

def compute_features(
    instrument: InstrumentConfig,
    label_profile_name: str | None = None,
    recompute_all: bool = False,
    verbose: bool = True,
) -> int:
    """
    Compute and store feature vectors for all unlabelled entry-TF candles.
    Returns number of new feature rows written.
    """
    from ml.database import get_db, Database
    db = get_db()
    profile_name = label_profile_name or instrument.label_profile

    # Find latest already-computed ts to avoid recomputing
    since_ts: int | None = None
    if not recompute_all:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT MAX(ts) FROM features WHERE symbol=? AND label_profile=?",
                (instrument.symbol, profile_name),
            ).fetchone()
            since_ts = row[0] if row and row[0] else None

    # Load all candles for all timeframes into memory
    all_bars: dict[str, pd.DataFrame] = {}
    for tf in instrument.timeframes:
        rows = db.get_candles(instrument.symbol, tf, limit=100_000)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df.set_index("ts").sort_index()
        all_bars[tf] = df[["open", "high", "low", "close", "volume", "spread"]]

    entry_tf = instrument.entry_tf
    if entry_tf not in all_bars:
        if verbose:
            print(f"  [{instrument.symbol}] No data for entry TF {entry_tf}. Run data_collector first.")
        return 0

    entry_candles = db.get_candles(instrument.symbol, entry_tf, limit=200_000,
                                   since_ts=since_ts)
    if not entry_candles:
        if verbose:
            print(f"  [{instrument.symbol}] No new candles to process.")
        return 0

    if verbose:
        print(f"  [{instrument.symbol}] Computing features for {len(entry_candles)} candles...")

    feature_rows = []
    skipped      = 0

    for candle in entry_candles:
        ts  = candle["ts"]
        cid = candle["id"]

        # Slice every TF to bars up to this candle's open time
        bars_slice: dict[str, pd.DataFrame] = {}
        for tf, df in all_bars.items():
            cut = df[df.index.astype("int64") // 10**9 <= ts]
            bars_slice[tf] = cut

        fvec = build_feature_vector(bars_slice, entry_tf, ts)
        if not fvec:
            skipped += 1
            continue

        feature_rows.append({
            "candle_id":     cid,
            "symbol":        instrument.symbol,
            "timeframe":     entry_tf,
            "ts":            ts,
            "label_profile": profile_name,
            "feature_json":  json.dumps(fvec),
        })

        # Batch write every 1000 rows
        if len(feature_rows) >= 1000:
            db.upsert_features(feature_rows)
            feature_rows = []

    if feature_rows:
        db.upsert_features(feature_rows)

    total = db.feature_count(instrument.symbol, profile_name)
    if verbose:
        print(f"  [{instrument.symbol}] Features: {total} total (skipped {skipped} warmup bars)")
    return total


# ── Live single-candle feature builder ───────────────────────────────────────

def get_live_features(
    instrument: InstrumentConfig,
    bars: dict[str, pd.DataFrame],
    ts_utc: int,
) -> dict[str, float]:
    """
    Build a live feature vector for the current candle.
    Used by the probability engine during live trading.
    Thin wrapper around build_feature_vector.
    """
    return build_feature_vector(bars, instrument.entry_tf, ts_utc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Feature engine")
    parser.add_argument("--symbol",  default="", help="Instrument (default: config SYMBOL)")
    parser.add_argument("--all",     action="store_true", help="Process all instruments")
    parser.add_argument("--recompute", action="store_true", help="Recompute all (slow)")
    parser.add_argument("--profile", default="", help="Label profile name")
    args = parser.parse_args()

    if args.all:
        for sym, inst in INSTRUMENTS.items():
            print(f"\n[{sym}] Computing features...")
            compute_features(inst, args.profile or None, recompute_all=args.recompute)
    else:
        inst = get_instrument(args.symbol or None)
        compute_features(inst, args.profile or None, recompute_all=args.recompute)

    print("\nNext: python train.py --label")


if __name__ == "__main__":
    _cli()
