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
import time
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
import numpy as np

from ml.database import get_db
from ml.instrument_config import (
    INSTRUMENTS, InstrumentConfig, get_instrument, get_label_profile,
    TF_TO_MINUTES as _TF_TO_MINUTES,
)


MAX_FEATURE_ROWS = 1_000_000


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


# ── Vectorised per-TF feature matrix (batch mode) ───────────────────────────

def _vectorize_tf_features(df: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    """
    Compute ALL features for a full TF DataFrame in one pass.
    Returns a DataFrame indexed identically to `df` with one column per feature.
    All pandas rolling/ewm ops are causal — no lookahead.
    """
    sfx = f"_{tf_label.lower()}"
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

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
    rel_vol = v / vol20.replace(0, np.nan)
    mom5   = c - c.shift(5)
    mom10  = c - c.shift(10)

    atr_safe = atr14.replace(0, np.nan).fillna(1.0)
    bar_range = (h - l).replace(0, np.nan)
    bar_body  = (c - o).abs()
    ema50_slope = (ema50 - ema50.shift(5)) / atr_safe

    feat = pd.DataFrame({
        f"ema9{sfx}":          ema9,
        f"ema20{sfx}":         ema20,
        f"ema50{sfx}":         ema50,
        f"ema200{sfx}":        ema200,
        f"rsi14{sfx}":         rsi14,
        f"rsi7{sfx}":          rsi7,
        f"macd_line{sfx}":     macd_l,
        f"macd_signal{sfx}":   macd_s,
        f"macd_hist{sfx}":     macd_h,
        f"atr14{sfx}":         atr14,
        f"atr_pct{sfx}":       atr_p,
        f"rolling_vol{sfx}":   rvol20,
        f"rel_volume{sfx}":    rel_vol,
        f"mom5{sfx}":          mom5,
        f"mom10{sfx}":         mom10,
        f"dist_ema20{sfx}":    (c - ema20) / atr_safe,
        f"dist_ema50{sfx}":    (c - ema50) / atr_safe,
        f"dist_ema200{sfx}":   (c - ema200) / atr_safe,
        f"candle_range{sfx}":  h - l,
        f"candle_body{sfx}":   bar_body,
        f"body_ratio{sfx}":    bar_body / bar_range,
        f"ema50_slope{sfx}":   ema50_slope,
        f"trending{sfx}":      (ema50_slope.abs() > 0.5).astype(float),
    }, index=df.index)

    return feat.fillna(0.0)


def _vectorize_session_flags(ts_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return session flag columns for a DatetimeIndex (UTC)."""
    hour = ts_index.hour
    dow  = ts_index.dayofweek
    return pd.DataFrame({
        "session_asian":  (hour < 9).astype(float),
        "session_london": ((hour >= 8) & (hour < 16)).astype(float),
        "session_ny":     ((hour >= 13) & (hour < 22)).astype(float),
        "hour_of_day":    hour.astype(float),
        "day_of_week":    dow.astype(float),
    }, index=ts_index)


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


# ── Batch feature computation (vectorised) ───────────────────────────────────

def compute_features(
    instrument: InstrumentConfig,
    label_profile_name: str | None = None,
    recompute_all: bool = False,
    verbose: bool = True,
) -> int:
    """
    Compute and store feature vectors for all entry-TF candles.

    Indicators are computed ONCE per timeframe across the full DataFrame (O(n)
    per TF), then aligned to entry-TF timestamps via merge_asof.  The write
    phase uses a numpy-backed vectorised loop so JSON serialisation never
    becomes the bottleneck at 500k+ candles.

    Scales linearly:  500k candles across 4 TFs → ~30–60s on a laptop.
    Returns the total number of feature rows in DB after this run.
    """
    _BATCH_SIZE = 5_000   # rows per SQLite executemany call
    _LOG_EVERY  = 1_000   # progress line every N rows processed

    db           = get_db()
    profile_name = label_profile_name or instrument.label_profile
    entry_tf     = instrument.entry_tf
    sym          = instrument.symbol
    t_start      = time.perf_counter()

    # ── Find resume point ─────────────────────────────────────────────────────
    since_ts: int | None = None
    if not recompute_all:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT MAX(ts) FROM features WHERE symbol=? AND label_profile=?",
                (sym, profile_name),
            ).fetchone()
            since_ts = row[0] if row and row[0] else None

    # ── Load all TF candles ───────────────────────────────────────────────────
    all_bars: dict[str, pd.DataFrame] = {}
    for tf in instrument.timeframes:
        rows = db.get_candles(sym, tf, limit=MAX_FEATURE_ROWS)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df.set_index("ts").sort_index()
        all_bars[tf] = df[["open", "high", "low", "close", "volume", "spread"]]

    if entry_tf not in all_bars:
        if verbose:
            print(f"  [{sym}] No data for entry TF {entry_tf}. Run data_collector first.")
        return 0

    # ── Load entry-TF candles that need features ──────────────────────────────
    entry_candles = db.get_candles(sym, entry_tf, limit=MAX_FEATURE_ROWS, since_ts=since_ts)
    if not entry_candles:
        if verbose:
            print(f"  [{sym}] No new candles to process.")
        return 0

    n_entry = len(entry_candles)
    if verbose:
        print(f"  [{sym}] {n_entry:,} candles to process across "
              f"{len(all_bars)} timeframes — precomputing indicators...", flush=True)

    cid_by_ts = {int(c["ts"]): c["id"] for c in entry_candles}

    # ── Phase 1: Compute indicators ONCE per TF (vectorised, O(n) per TF) ────
    t_ind = time.perf_counter()
    tf_feat_dfs: list[pd.DataFrame] = []
    for tf, df in all_bars.items():
        if len(df) < 30:
            if verbose:
                print(f"    {tf:>4}: skipped — only {len(df):,} bars (<30 minimum)", flush=True)
            continue
        t_tf = time.perf_counter()
        feat_df = _vectorize_tf_features(df, tf)
        tf_feat_dfs.append(feat_df)
        if verbose:
            print(f"    {tf:>4}: {len(df):>8,} bars  →  {len(feat_df.columns):>3} features  "
                  f"({time.perf_counter() - t_tf:.2f}s)", flush=True)

    if not tf_feat_dfs:
        if verbose:
            print(f"  [{sym}] No feature data produced.")
        return 0

    if verbose:
        print(f"  [{sym}] Indicators done in {time.perf_counter()-t_ind:.2f}s. "
              f"Aligning timeframes...", flush=True)

    # ── Phase 2: Align all TF features to entry-TF timestamps (merge_asof) ───
    t_merge = time.perf_counter()
    entry_index = all_bars[entry_tf].index

    combined: pd.DataFrame | None = None
    for feat_df in tf_feat_dfs:
        feat_df_sorted = feat_df.sort_index()
        if combined is None:
            combined = pd.merge_asof(
                pd.DataFrame(index=entry_index),
                feat_df_sorted,
                left_index=True,
                right_index=True,
            )
        else:
            combined = pd.merge_asof(
                combined,
                feat_df_sorted,
                left_index=True,
                right_index=True,
            )

    if combined is None or combined.empty:
        return 0

    # Session flags
    session_df = _vectorize_session_flags(combined.index)
    combined   = pd.concat([combined, session_df], axis=1).fillna(0.0)

    if verbose:
        print(f"  [{sym}] Merge done in {time.perf_counter()-t_merge:.2f}s. "
              f"Filtering + writing {n_entry:,} rows...", flush=True)

    # ── Phase 3: Filter to needed rows, drop warmup rows ─────────────────────
    combined["_ts_int"] = (combined.index.astype("int64") // 10**9).astype(int)
    need_ts         = set(cid_by_ts.keys())
    matched_rows    = combined[combined["_ts_int"].isin(need_ts)]

    MIN_NONZERO    = 10
    nonzero_counts = (matched_rows.drop(columns=["_ts_int"]) != 0.0).sum(axis=1)
    rows_to_write  = matched_rows[nonzero_counts >= MIN_NONZERO]

    skipped      = n_entry - len(rows_to_write)
    feature_cols = [c for c in combined.columns if c != "_ts_int"]
    n_rows       = len(rows_to_write)

    if verbose:
        warmup_dropped = len(matched_rows) - n_rows
        min_nonzero = int(nonzero_counts.min()) if len(nonzero_counts) else 0
        max_nonzero = int(nonzero_counts.max()) if len(nonzero_counts) else 0
        print(
            f"  [{sym}] Filter detail: entry={n_entry:,}, timestamp_matches={len(matched_rows):,}, "
            f"warmup_or_sparse_dropped={warmup_dropped:,}, nonzero_range={min_nonzero}-{max_nonzero}",
            flush=True,
        )

    if n_rows == 0:
        combined_first = int(combined["_ts_int"].iloc[0]) if len(combined) else None
        combined_last = int(combined["_ts_int"].iloc[-1]) if len(combined) else None
        need_first = min(need_ts) if need_ts else None
        need_last = max(need_ts) if need_ts else None
        raise RuntimeError(
            f"[{sym}] Feature generation produced 0 writable rows. "
            f"entry_candles={n_entry}, combined_rows={len(combined)}, "
            f"timestamp_matches={len(matched_rows)}, "
            f"combined_ts_range={combined_first}..{combined_last}, "
            f"needed_ts_range={need_first}..{need_last}. "
            "Check timeframe alignment, stale feature resume state, and warmup filtering."
        )

    # ── Phase 4: Vectorised write — numpy array avoids Python row-iteration ───
    # Converting to numpy once means json.dumps is the only per-row Python work.
    feat_np  = rows_to_write[feature_cols].to_numpy(dtype=float, na_value=0.0)
    ts_arr   = rows_to_write["_ts_int"].to_numpy(dtype=int)

    written     = 0
    t_write     = time.perf_counter()
    batch: list[dict] = []

    for i in range(n_rows):
        processed = i + 1
        ts_int = int(ts_arr[i])
        fvec   = dict(zip(feature_cols, feat_np[i].tolist()))
        batch.append({
            "candle_id":     cid_by_ts[ts_int],
            "symbol":        sym,
            "timeframe":     entry_tf,
            "ts":            ts_int,
            "label_profile": profile_name,
            "feature_json":  json.dumps(fvec),
        })

        if len(batch) >= _BATCH_SIZE:
            db.upsert_features(batch)
            written += len(batch)
            batch = []

        # Progress + ETA every _LOG_EVERY processed rows, independent of DB batch size.
        if verbose and (processed % _LOG_EVERY == 0 or processed == n_rows):
            now     = time.perf_counter()
            elapsed = now - t_write
            rps     = processed / elapsed if elapsed > 0 else 1
            eta_s   = (n_rows - processed) / rps
            pct     = processed / n_rows * 100
            print(f"  [{sym}] Writing {processed:>{len(str(n_rows))},}/{n_rows:,} "
                  f"({pct:5.1f}%)  {rps:,.0f} rows/s  ETA {eta_s:.0f}s", flush=True)

    if batch:
        db.upsert_features(batch)
        written += len(batch)

    total   = db.feature_count(sym, profile_name)
    elapsed = time.perf_counter() - t_start
    if verbose:
        rps = written / (time.perf_counter() - t_write) if written else 0
        print(
            f"  [{sym}] Features: {total:,} total  "
            f"(wrote {written:,}, skipped {skipped:,} warmup)  "
            f"— {rps:,.0f} rows/s write  —  total {elapsed:.1f}s",
            flush=True,
        )
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
