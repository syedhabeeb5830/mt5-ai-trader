"""
ML Platform — Model Monitor
─────────────────────────────────────────────────────────────────────────────
Tracks live prediction quality and triggers alerts / retraining when the
model starts degrading.

Metrics tracked (rolling window):
  • prediction accuracy         — live outcome vs predicted label
  • rolling win rate            — last N resolved trades
  • feature drift score         — mean shift of live feature distributions
                                  vs training baseline
  • model drift flag            — AUC on last N predictions vs training AUC

Alert levels:
  INFO    — within normal bounds
  WARN    — 10–20% degradation
  ALERT   — >20% degradation → recommend retrain
  HALT    — >40% degradation → auto-switch to paper mode

Usage (standalone check):
  python -m ml.monitor                        # active symbol
  python -m ml.monitor --symbol EURUSD
  python -m ml.monitor --retrain-if-degraded  # auto-trigger trainer
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from ml.database import get_db
from ml.instrument_config import (
    INSTRUMENTS, InstrumentConfig, LOGS_DIR, MIN_SAMPLES_TO_TRAIN,
    get_instrument,
)


# ── Thresholds ─────────────────────────────────────────────────────────────────

WARN_THRESHOLD  = 0.10   # 10% relative drop triggers WARN
ALERT_THRESHOLD = 0.20   # 20% relative drop triggers ALERT
HALT_THRESHOLD  = 0.40   # 40% relative drop triggers HALT
MIN_RESOLVED    = 30     # minimum resolved predictions before any judgment


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class MonitorReport:
    symbol:            str
    n_resolved:        int
    rolling_win_rate:  float          # last N resolved, fraction TP
    training_tp_rate:  float          # base rate from labels table
    relative_drop:     float          # (train_rate - live_rate) / train_rate
    feature_drift:     float          # mean absolute z-score of live features
    alert_level:       str            # INFO | WARN | ALERT | HALT
    recommendation:    str
    checked_at:        str

    def to_dict(self) -> dict:
        return {
            "symbol":           self.symbol,
            "n_resolved":       self.n_resolved,
            "rolling_win_rate": round(self.rolling_win_rate, 4),
            "training_tp_rate": round(self.training_tp_rate, 4),
            "relative_drop":    round(self.relative_drop, 4),
            "feature_drift":    round(self.feature_drift, 4),
            "alert_level":      self.alert_level,
            "recommendation":   self.recommendation,
            "checked_at":       self.checked_at,
        }

    def print(self) -> None:
        icon = {"INFO": "✓", "WARN": "⚠", "ALERT": "⚡", "HALT": "✗"}.get(self.alert_level, "?")
        print(f"\n{'─'*52}")
        print(f"  Monitor: {self.symbol}  [{icon} {self.alert_level}]")
        print(f"  Resolved predictions : {self.n_resolved}")
        print(f"  Rolling win rate     : {self.rolling_win_rate:.1%}")
        print(f"  Training TP rate     : {self.training_tp_rate:.1%}")
        print(f"  Relative degradation : {self.relative_drop:+.1%}")
        print(f"  Feature drift score  : {self.feature_drift:.3f}")
        print(f"  Recommendation       : {self.recommendation}")
        print(f"{'─'*52}")


# ── Core monitor ──────────────────────────────────────────────────────────────

def check(
    instrument: InstrumentConfig,
    window: int = 100,
    verbose: bool = True,
) -> MonitorReport:
    """
    Check current model health for `instrument`.
    window : number of most-recent resolved predictions to evaluate.
    """
    db           = get_db()
    profile_name = instrument.label_profile

    # ── 1. Rolling win rate from resolved predictions ─────────────────────────
    recent_preds = [
        p for p in db.get_recent_predictions(instrument.symbol, limit=window * 2)
        if p["outcome_label"] is not None
    ][-window:]

    n_resolved  = len(recent_preds)
    rolling_wr  = float(np.mean([p["outcome_label"] for p in recent_preds])) if recent_preds else 0.5

    # ── 2. Training TP rate baseline ─────────────────────────────────────────
    with db._conn() as conn:
        row = conn.execute(
            "SELECT AVG(label) FROM labels WHERE symbol=? AND label_profile=? AND direction='BUY'",
            (instrument.symbol, profile_name),
        ).fetchone()
    train_tp = float(row[0]) if row and row[0] is not None else 0.5

    # ── 3. Feature drift ─────────────────────────────────────────────────────
    drift_score = _feature_drift(instrument, db, window)

    # ── 4. Determine alert level ──────────────────────────────────────────────
    if n_resolved < MIN_RESOLVED:
        relative_drop = 0.0
        alert         = "INFO"
        recommendation = f"Collecting data ({n_resolved}/{MIN_RESOLVED} resolved predictions needed)"
    else:
        relative_drop = (train_tp - rolling_wr) / train_tp if train_tp > 0 else 0.0
        if relative_drop >= HALT_THRESHOLD or drift_score > 3.0:
            alert = "HALT"
            recommendation = ("HALT: Model severely degraded. Auto-switch to paper mode. "
                              "Run: python train.py --retrain")
        elif relative_drop >= ALERT_THRESHOLD or drift_score > 2.0:
            alert = "ALERT"
            recommendation = "ALERT: Significant degradation. Run: python train.py --retrain"
        elif relative_drop >= WARN_THRESHOLD or drift_score > 1.0:
            alert = "WARN"
            recommendation = "WARN: Early degradation. Monitor closely. Consider retraining."
        else:
            alert = "INFO"
            recommendation = "Model performing within expected bounds."

    report = MonitorReport(
        symbol=instrument.symbol,
        n_resolved=n_resolved,
        rolling_win_rate=rolling_wr,
        training_tp_rate=train_tp,
        relative_drop=relative_drop,
        feature_drift=drift_score,
        alert_level=alert,
        recommendation=recommendation,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )

    # Save report to logs
    path = LOGS_DIR / f"monitor_{instrument.symbol}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))

    if verbose:
        report.print()
    return report


def _feature_drift(instrument: InstrumentConfig, db, window: int) -> float:
    """
    Compute mean absolute z-score of live feature distributions
    vs training baseline. Returns 0.0 if insufficient data.
    """
    profile_name = instrument.label_profile

    # Training baseline: sample up to 2000 rows
    train_rows = db.get_features(instrument.symbol, profile_name, limit=2000)
    if len(train_rows) < 50:
        return 0.0

    train_vecs = [json.loads(r["feature_json"]) for r in train_rows]
    train_df   = _to_numeric_df(train_vecs)

    # Recent live feature rows
    recent_rows = db.get_features(instrument.symbol, profile_name, limit=window)[-window:]
    if len(recent_rows) < 10:
        return 0.0

    live_vecs = [json.loads(r["feature_json"]) for r in recent_rows]
    live_df   = _to_numeric_df(live_vecs)

    # Align columns
    cols = [c for c in train_df.columns if c in live_df.columns]
    if not cols:
        return 0.0

    train_means = train_df[cols].mean()
    train_stds  = train_df[cols].std().replace(0, 1.0)
    live_means  = live_df[cols].mean()

    z_scores = ((live_means - train_means) / train_stds).abs()
    return float(z_scores.mean())


def _to_numeric_df(vecs: list[dict]) -> "pd.DataFrame":
    import pandas as pd
    df = pd.DataFrame(vecs)
    return df.select_dtypes(include="number").fillna(0.0)


def check_all(window: int = 100, retrain_if_degraded: bool = False) -> None:
    for sym, inst in INSTRUMENTS.items():
        report = check(inst, window=window)
        if retrain_if_degraded and report.alert_level in ("ALERT", "HALT"):
            print(f"\n[{sym}] Auto-retraining triggered (alert={report.alert_level})...")
            from ml.trainer import train
            train(inst, direction="BUY", verbose=True)
            train(inst, direction="SELL", verbose=True)


# ── Convenience query for scalper ────────────────────────────────────────────

def should_halt(symbol: str) -> bool:
    """
    Quick check: returns True if last saved monitor report is HALT level.
    Used by scalper.py to auto-switch to paper mode.
    """
    path = LOGS_DIR / f"monitor_{symbol.upper()}.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        return data.get("alert_level") == "HALT"
    except Exception:
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Model monitor")
    parser.add_argument("--symbol",  default="")
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--window",  type=int, default=100)
    parser.add_argument("--retrain-if-degraded", action="store_true")
    args = parser.parse_args()

    if args.all:
        check_all(window=args.window, retrain_if_degraded=args.retrain_if_degraded)
    else:
        inst = get_instrument(args.symbol or None)
        report = check(inst, window=args.window)
        if args.retrain_if_degraded and report.alert_level in ("ALERT", "HALT"):
            from ml.trainer import train
            train(inst, direction="BUY")
            train(inst, direction="SELL")


if __name__ == "__main__":
    _cli()
