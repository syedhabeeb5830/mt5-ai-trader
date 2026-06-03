"""
ML Platform — Labeling Engine
─────────────────────────────────────────────────────────────────────────────
Forward-scans every entry-TF candle and determines:
  Did price hit TP before SL within the next N bars?

Label = 1  →  TP reached first (trade wins)
Label = 0  →  SL reached first OR horizon elapsed (trade loses / expires)

Spread is deducted from the label calculation:
  BUY  entry: effective entry = ask = close + spread/2
              TP hit when HIGH  >= entry + tp_points
              SL hit when LOW   <= entry - sl_points
  SELL entry: effective entry = bid = close - spread/2
              TP hit when LOW   <= entry - tp_points
              SL hit when HIGH  >= entry + sl_points

Both BUY and SELL labels are computed for every candle.
The model can then be trained separately per direction or jointly
with direction as a feature.

Usage:
  python -m ml.labeler                        # active symbol
  python -m ml.labeler --symbol EURUSD
  python -m ml.labeler --profile intraday
  python -m ml.labeler --all
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd

from ml.database import get_db
from ml.instrument_config import (
    INSTRUMENTS, InstrumentConfig, LabelProfile,
    get_instrument, get_label_profile,
)


# ── Core labeling logic ───────────────────────────────────────────────────────

def _label_candle(
    idx:         int,
    closes:      list[float],
    highs:       list[float],
    lows:        list[float],
    direction:   str,          # "BUY" | "SELL"
    entry_price: float,        # already includes half-spread offset
    tp_abs:      float,        # absolute price level (not points)
    sl_abs:      float,        # absolute price level (not points)
    horizon:     int,
) -> tuple[int, Optional[int], Optional[float]]:
    """
    Scan forward from `idx` up to `horizon` bars.
    Returns (label, bars_to_exit, exit_price).
    """
    n = len(closes)
    for offset in range(1, horizon + 1):
        i = idx + offset
        if i >= n:
            break
        high = highs[i]
        low  = lows[i]
        if direction == "BUY":
            if high >= tp_abs:
                return (1, offset, tp_abs)
            if low  <= sl_abs:
                return (0, offset, sl_abs)
        else:  # SELL
            if low  <= tp_abs:
                return (1, offset, tp_abs)
            if high >= sl_abs:
                return (0, offset, sl_abs)
    # Horizon elapsed without a hit
    return (0, None, None)


def label_instrument(
    instrument: InstrumentConfig,
    profile:    LabelProfile,
    recompute:  bool = False,
    verbose:    bool = True,
) -> int:
    """
    Label all unlabelled entry-TF candles for `instrument` using `profile`.
    Returns number of label rows written.
    """
    db = get_db()
    entry_tf = instrument.entry_tf
    spread   = instrument.spread_typical

    # Find latest already-labelled ts
    since_ts: int | None = None
    if not recompute:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT MAX(ts) FROM labels WHERE symbol=? AND label_profile=? AND direction='BUY'",
                (instrument.symbol, profile.name),
            ).fetchone()
            since_ts = row[0] if row and row[0] else None

    # Load entry-TF candles
    candles = db.get_candles(instrument.symbol, entry_tf, limit=200_000, since_ts=since_ts)
    # We need horizon extra bars beyond the last labellable candle, so load without since filter
    all_candles = db.get_candles(instrument.symbol, entry_tf, limit=200_000)
    if not all_candles:
        if verbose:
            print(f"  [{instrument.symbol}] No candles. Run data_collector first.")
        return 0

    if verbose:
        print(f"  [{instrument.symbol}/{entry_tf}] Labeling {len(all_candles)} candles "
              f"(profile={profile.name}, TP={profile.tp_points}, SL={profile.sl_points}, "
              f"horizon={profile.horizon_bars} bars)...")

    # Extract arrays for fast indexing
    ids    = [c["id"]    for c in all_candles]
    ts_arr = [c["ts"]    for c in all_candles]
    closes = [c["close"] for c in all_candles]
    highs  = [c["high"]  for c in all_candles]
    lows   = [c["low"]   for c in all_candles]

    # Build set of already-labelled candle ids (for incremental mode)
    labelled_ids: set[int] = set()
    if since_ts:
        with db._conn() as conn:
            rows = conn.execute(
                "SELECT candle_id FROM labels WHERE symbol=? AND label_profile=?",
                (instrument.symbol, profile.name),
            ).fetchall()
        labelled_ids = {r[0] for r in rows}

    label_rows = []
    labellable = len(all_candles) - profile.horizon_bars  # last N bars can't be labelled yet

    for i in range(labellable):
        cid = ids[i]
        if cid in labelled_ids:
            continue

        ts    = ts_arr[i]
        close = closes[i]
        half_spread = spread / 2.0

        for direction in ("BUY", "SELL"):
            if direction == "BUY":
                entry  = close + half_spread
                tp_abs = round(entry + profile.tp_points, 6)
                sl_abs = round(entry - profile.sl_points, 6)
            else:
                entry  = close - half_spread
                tp_abs = round(entry - profile.tp_points, 6)
                sl_abs = round(entry + profile.sl_points, 6)

            lbl, bars_to_exit, exit_price = _label_candle(
                i, closes, highs, lows, direction,
                entry, tp_abs, sl_abs, profile.horizon_bars,
            )
            label_rows.append({
                "candle_id":    cid,
                "symbol":       instrument.symbol,
                "timeframe":    entry_tf,
                "ts":           ts,
                "label_profile": profile.name,
                "direction":    direction,
                "label":        lbl,
                "bars_to_exit": bars_to_exit,
                "exit_price":   exit_price,
            })

        if len(label_rows) >= 2000:
            db.upsert_labels(label_rows)
            label_rows = []

    if label_rows:
        db.upsert_labels(label_rows)

    total = db.label_count(instrument.symbol, profile.name)
    tp_rate = _tp_rate(instrument.symbol, profile.name, db)
    if verbose:
        print(f"  [{instrument.symbol}] Labels: {total} total "
              f"(BUY TP-rate ≈ {tp_rate:.1%})")
    return total


def _tp_rate(symbol: str, profile_name: str, db) -> float:
    with db._conn() as conn:
        row = conn.execute(
            "SELECT AVG(label) FROM labels WHERE symbol=? AND label_profile=? AND direction='BUY'",
            (symbol, profile_name),
        ).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0


def label_all(
    profile_name: str | None = None,
    recompute: bool = False,
    verbose: bool = True,
) -> None:
    for sym, inst in INSTRUMENTS.items():
        profile = get_label_profile(profile_name or inst.label_profile)
        print(f"\n[{sym}] Labeling...")
        label_instrument(inst, profile, recompute=recompute, verbose=verbose)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Labeling engine")
    parser.add_argument("--symbol",    default="",    help="Instrument")
    parser.add_argument("--profile",   default="",    help="Label profile name")
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--recompute", action="store_true", help="Re-label everything")
    args = parser.parse_args()

    if args.all:
        label_all(args.profile or None, recompute=args.recompute)
    else:
        inst    = get_instrument(args.symbol or None)
        profile = get_label_profile(args.profile or inst.label_profile)
        label_instrument(inst, profile, recompute=args.recompute)

    print("\nNext: python train.py --train")


if __name__ == "__main__":
    _cli()
