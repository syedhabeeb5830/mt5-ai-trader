"""
ML Trading Platform — Unified Entry Point
─────────────────────────────────────────────────────────────────────────────
ONE COMMAND for everything.

  python train.py                  full pipeline: collect → features → label → train
  python train.py --collect        data collection only
  python train.py --features       feature computation only
  python train.py --label          labeling only
  python train.py --train          model training only
  python train.py --retrain        retrain with new live data included
  python train.py --monitor        check model health
  python train.py --status         DB summary + model inventory

  python train.py --symbol EURUSD  run for a specific instrument
  python train.py --all            run for all configured instruments
  python train.py --days 365       lookback for data collection
  python train.py --direction BOTH train BUY and SELL models
  python train.py --universal      train one model across all instruments

After running this command, run the bot:
  python scalper.py --strategy ml_scalper --paper
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _banner(msg: str) -> None:
    width = 56
    print(f"\n{'═'*width}")
    print(f"  {msg}")
    print(f"{'═'*width}")


def _step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}")
    print("─" * 50)


def run_collect(symbol: str | None, days: int, all_instruments: bool) -> None:
    from ml.data_collector import collect, collect_all
    from ml.instrument_config import get_instrument

    if all_instruments:
        collect_all(days=days)
    else:
        inst = get_instrument(symbol)
        collect(inst, days=days)


def run_features(symbol: str | None, all_instruments: bool, recompute: bool) -> None:
    from ml.feature_engine import compute_features
    from ml.instrument_config import INSTRUMENTS, get_instrument

    if all_instruments:
        for sym, inst in INSTRUMENTS.items():
            print(f"\n[{sym}] Computing features...")
            compute_features(inst, recompute_all=recompute)
    else:
        inst = get_instrument(symbol)
        compute_features(inst, recompute_all=recompute)


def run_label(symbol: str | None, all_instruments: bool, recompute: bool) -> None:
    from ml.labeler import label_instrument, label_all
    from ml.instrument_config import INSTRUMENTS, get_instrument, get_label_profile

    if all_instruments:
        label_all(recompute=recompute)
    else:
        inst    = get_instrument(symbol)
        profile = get_label_profile(inst.label_profile)
        label_instrument(inst, profile, recompute=recompute)


def run_train(
    symbol: str | None,
    all_instruments: bool,
    direction: str,
    universal: bool,
) -> None:
    from ml.trainer import train, train_all
    from ml.instrument_config import get_instrument

    dirs = ["BUY", "SELL"] if direction == "BOTH" else [direction]

    if all_instruments or universal:
        for d in dirs:
            train_all(direction=d, universal=universal)
    else:
        inst = get_instrument(symbol)
        for d in dirs:
            print(f"\n[{inst.symbol}] Training {d}...")
            train(inst, direction=d)


def run_monitor(symbol: str | None, all_instruments: bool, retrain: bool) -> None:
    from ml.monitor import check, check_all
    from ml.instrument_config import get_instrument

    if all_instruments:
        check_all(retrain_if_degraded=retrain)
    else:
        inst   = get_instrument(symbol)
        report = check(inst)
        if retrain and report.alert_level in ("ALERT", "HALT"):
            from ml.trainer import train
            print(f"\n[{inst.symbol}] Retraining triggered...")
            for d in ["BUY", "SELL"]:
                train(inst, direction=d)


def run_status() -> None:
    from ml.database import get_db
    from ml.instrument_config import MODELS_DIR

    db      = get_db()
    summary = db.summary()

    print("\n── Database ─────────────────────────────────────────────")
    print("Candles:")
    for r in summary["candles"]:
        print(f"  {r['symbol']:10} {r['timeframe']:5}  {r['n']:>8,} bars")

    print("\nFeatures:")
    for r in summary["features"]:
        print(f"  {r['symbol']:10} profile={r['label_profile']:12}  {r['n']:>8,} rows")

    print("\nLabels:")
    for r in summary["labels"]:
        print(f"  {r['symbol']:10} {r['direction']:5} profile={r['label_profile']:12}  "
              f"{r['n']:>6,} rows  TP-rate={r['wr']:.1%}" if r['wr'] else
              f"  {r['symbol']:10} {r['direction']:5} profile={r['label_profile']:12}  {r['n']:>6,} rows")

    print("\nPredictions (live):")
    for r in summary["predictions"]:
        print(f"  {r['symbol']:10}  {r['n']:>6,} predictions logged")

    print("\n── Models ───────────────────────────────────────────────")
    models = sorted(MODELS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if models:
        for m in models[:10]:
            mtime = datetime.fromtimestamp(m.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {m.name:55}  {mtime}")
    else:
        print("  No trained models found. Run: python train.py")

    print(f"\n── Config ───────────────────────────────────────────────")
    import config
    from ml.instrument_config import ACTIVE_SYMBOL, ACTIVE_LABEL_PROFILE, RETRAIN_SCHEDULE
    print(f"  Active symbol  : {ACTIVE_SYMBOL}")
    print(f"  Label profile  : {ACTIVE_LABEL_PROFILE}")
    print(f"  Retrain sched  : {RETRAIN_SCHEDULE}")
    print(f"  Model dir      : {MODELS_DIR.resolve()}")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline(
    symbol: str | None,
    days: int,
    all_instruments: bool,
    direction: str,
    universal: bool,
) -> None:
    steps = 4
    t0    = time.time()

    _step(1, steps, "Data Collection")
    run_collect(symbol, days, all_instruments)

    _step(2, steps, "Feature Engineering")
    run_features(symbol, all_instruments, recompute=False)

    _step(3, steps, "Labeling")
    run_label(symbol, all_instruments, recompute=False)

    _step(4, steps, "Model Training")
    run_train(symbol, all_instruments, direction, universal)

    elapsed = round(time.time() - t0, 1)
    _banner(f"Pipeline complete in {elapsed}s")
    print("\nTo start trading with ML signals:")
    print("  python scalper.py --strategy ml_scalper --paper")
    print("\nTo check model health:")
    print("  python train.py --monitor")


def run_retrain(symbol: str | None, all_instruments: bool, direction: str) -> None:
    """
    Retrain pipeline: features + label (incremental, using new live data) + train.
    Does NOT re-collect historical data.
    """
    steps = 3
    _step(1, steps, "Incremental Feature Update")
    run_features(symbol, all_instruments, recompute=False)

    _step(2, steps, "Incremental Labeling")
    run_label(symbol, all_instruments, recompute=False)

    _step(3, steps, "Model Retraining")
    run_train(symbol, all_instruments, direction, universal=False)

    _banner("Retrain complete")
    print("Models updated with latest live data.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML Trading Platform — unified pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Mode flags (mutually exclusive group)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--collect",  action="store_true", help="Data collection only")
    modes.add_argument("--features", action="store_true", help="Feature computation only")
    modes.add_argument("--label",    action="store_true", help="Labeling only")
    modes.add_argument("--train",    action="store_true", help="Training only")
    modes.add_argument("--retrain",  action="store_true", help="Retrain with new live data")
    modes.add_argument("--monitor",  action="store_true", help="Model health check")
    modes.add_argument("--status",   action="store_true", help="DB + model inventory")

    # Target
    parser.add_argument("--symbol",    default="",  help="Instrument (default: .env SYMBOL)")
    parser.add_argument("--all",       action="store_true", help="All configured instruments")
    parser.add_argument("--days",      type=int, default=365, help="Lookback for collection")
    parser.add_argument("--direction", default="BUY", choices=["BUY", "SELL", "BOTH"])
    parser.add_argument("--universal", action="store_true", help="Train universal model")
    parser.add_argument("--retrain-if-degraded", action="store_true",
                        help="Auto retrain when monitor detects degradation")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute features/labels from scratch")

    args = parser.parse_args()

    sym = args.symbol or None

    if args.status:
        run_status()

    elif args.collect:
        _banner("Data Collection")
        run_collect(sym, args.days, args.all)

    elif args.features:
        _banner("Feature Engineering")
        run_features(sym, args.all, args.recompute)

    elif args.label:
        _banner("Labeling")
        run_label(sym, args.all, args.recompute)

    elif args.train:
        _banner("Model Training")
        run_train(sym, args.all, args.direction, args.universal)

    elif args.retrain:
        _banner("Incremental Retrain")
        run_retrain(sym, args.all, args.direction)

    elif args.monitor:
        _banner("Model Health Monitor")
        run_monitor(sym, args.all, retrain=args.retrain_if_degraded)

    else:
        # Default: full pipeline
        _banner("Full ML Pipeline")
        run_full_pipeline(sym, args.days, args.all, args.direction, args.universal)


if __name__ == "__main__":
    main()
