"""
ML Platform — Data Collector
─────────────────────────────────────────────────────────────────────────────
Backfills and continuously appends OHLCV candles into ml/data/market.db.

Two data sources (tried in order):
  1. MetaTrader5 Python package — works when MT5 terminal is running (Windows)
  2. yfinance — offline fallback (symbol mapping: XAUUSD → GC=F, etc.)

Usage:
  python -m ml.data_collector                        # active symbol (config.py SYMBOL)
  python -m ml.data_collector --symbol EURUSD
  python -m ml.data_collector --symbol XAUUSD --days 365
  python -m ml.data_collector --all                  # all configured instruments
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

from ml.database import get_db
from ml.instrument_config import INSTRUMENTS, InstrumentConfig, get_instrument


# ── Timeframe helpers ─────────────────────────────────────────────────────────

TF_TO_MINUTES: dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}

# yfinance interval strings for each TF
_YF_INTERVAL: dict[str, str] = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "H4": "1h",   # H4 is resampled from H1
    "D1": "1d",
}

# yfinance max lookback per interval (API restriction)
_YF_MAX_DAYS: dict[str, int] = {
    "M1": 7, "M5": 60, "M15": 60, "M30": 60,
    "H1": 730, "H4": 730, "D1": 3650,
}

# MT5 symbol mappings for yfinance fallback
_YF_SYMBOL: dict[str, str] = {
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "BTCUSD": "BTC-USD",
}

# MT5 timeframe constants (set at runtime when mt5 is available)
_MT5_TF: dict[str, int] | None = None


def _mt5_tf_map() -> dict[str, int] | None:
    """Build MT5 timeframe constant dict. Returns None if MT5 not available."""
    global _MT5_TF
    if _MT5_TF is not None:
        return _MT5_TF
    try:
        import MetaTrader5 as mt5
        _MT5_TF = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
        }
        return _MT5_TF
    except Exception:
        return None


# ── MT5 source ────────────────────────────────────────────────────────────────

def _fetch_mt5(symbol: str, tf: str, since_ts: int, n_bars: int) -> Optional[pd.DataFrame]:
    tf_map = _mt5_tf_map()
    if tf_map is None or tf not in tf_map:
        return None
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None
        rates = mt5.copy_rates_from_pos(symbol, tf_map[tf], 0, n_bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df = df.rename(columns={"tick_volume": "volume", "spread": "spread_col"})
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df["spread"] = 0.0
        # Filter to only new bars
        if since_ts:
            df = df[df.index.astype("int64") // 10**9 > since_ts]
        return df
    except Exception:
        return None


# ── yfinance source ───────────────────────────────────────────────────────────

def _fetch_yfinance(symbol: str, tf: str, days: int) -> Optional[pd.DataFrame]:
    yf_sym = _YF_SYMBOL.get(symbol.upper())
    if yf_sym is None:
        return None
    try:
        import yfinance as yf
    except ImportError:
        return None

    actual_tf = tf
    resample_to_h4 = (tf == "H4")
    if resample_to_h4:
        actual_tf = "H1"

    interval = _YF_INTERVAL.get(actual_tf, "1h")
    max_days = min(days, _YF_MAX_DAYS.get(actual_tf, 60))

    try:
        df = yf.download(
            yf_sym,
            period=f"{max_days}d",
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None
    except Exception:
        return None

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    needed = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[needed].dropna()

    if resample_to_h4:
        df = df.resample("4h", closed="left", label="left").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna()

    df["spread"] = 0.0
    return df


# ── Core collector ────────────────────────────────────────────────────────────

def collect(
    instrument: InstrumentConfig,
    days: int = 365,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Collect candles for all timeframes of `instrument`.
    Returns {timeframe: rows_inserted}.
    """
    db = get_db()
    results: dict[str, int] = {}

    for tf in instrument.timeframes:
        since_ts = db.latest_candle_ts(instrument.symbol, tf) or 0
        bars_needed = max(
            int(days * 1440 / TF_TO_MINUTES.get(tf, 60)) + 200,
            500,
        )

        if verbose:
            existing = db.candle_count(instrument.symbol, tf)
            print(f"  [{instrument.symbol}/{tf}] existing={existing}, fetching up to {bars_needed} bars")

        # Try MT5 first
        df = _fetch_mt5(instrument.symbol, tf, since_ts, bars_needed)

        # Fall back to yfinance
        if df is None or df.empty:
            df = _fetch_yfinance(instrument.symbol, tf, days)
            if df is not None and not df.empty and verbose:
                print(f"  [{instrument.symbol}/{tf}] Using yfinance fallback")

        if df is None or df.empty:
            if verbose:
                print(f"  [{instrument.symbol}/{tf}] No data — MT5 not running and no yfinance mapping")
            results[tf] = 0
            continue

        # Convert to DB rows
        rows = []
        for ts_idx, row in df.iterrows():
            ts = int(ts_idx.timestamp()) if hasattr(ts_idx, "timestamp") else int(ts_idx)
            rows.append({
                "symbol":    instrument.symbol,
                "timeframe": tf,
                "ts":        ts,
                "open":      float(row["open"]),
                "high":      float(row["high"]),
                "low":       float(row["low"]),
                "close":     float(row["close"]),
                "volume":    float(row.get("volume", 0)),
                "spread":    float(row.get("spread", 0)),
            })

        inserted = db.upsert_candles(rows)
        results[tf] = inserted
        if verbose:
            total = db.candle_count(instrument.symbol, tf)
            print(f"  [{instrument.symbol}/{tf}] +{inserted} new → {total} total")

    return results


def collect_all(days: int = 365, verbose: bool = True) -> None:
    """Collect data for every instrument in INSTRUMENTS registry."""
    for symbol, inst in INSTRUMENTS.items():
        if verbose:
            print(f"\n{'─'*50}")
            print(f"Collecting: {inst.display_name} ({symbol})")
            print(f"{'─'*50}")
        collect(inst, days=days, verbose=verbose)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="ML data collector")
    parser.add_argument("--symbol",  default="",    help="Instrument symbol (default: config SYMBOL)")
    parser.add_argument("--days",    type=int, default=365, help="Lookback days")
    parser.add_argument("--all",     action="store_true",   help="Collect all instruments")
    parser.add_argument("--summary", action="store_true",   help="Print DB summary and exit")
    args = parser.parse_args()

    db = get_db()

    if args.summary:
        import json
        print(json.dumps(db.summary(), indent=2))
        return

    if args.all:
        collect_all(days=args.days)
    else:
        inst = get_instrument(args.symbol or None)
        collect(inst, days=args.days)

    print("\nCollection complete.")
    print("Next: python train.py --features   (compute feature vectors)")


if __name__ == "__main__":
    _cli()
