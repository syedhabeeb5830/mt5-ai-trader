"""
ML Platform — Training Pipeline
─────────────────────────────────────────────────────────────────────────────
Walk-forward training with XGBoost, LightGBM, or Random Forest.

Design:
  • Reads (features JOIN labels) from SQLite
  • Builds a flat numpy matrix — no lookahead (strict temporal ordering)
  • Walk-forward cross-validation (N splits, always train on past, test on future)
  • Trains final model on last K folds of training data
  • Calibrates probabilities with isotonic regression
  • Persists model to models/<SYMBOL>_<version>.pkl
  • Prints: accuracy, precision, recall, F1, ROC-AUC per fold + overall

Supported model types (from InstrumentConfig.model_type):
  "xgboost"       — XGBClassifier
  "lightgbm"      — LGBMClassifier
  "random_forest" — RandomForestClassifier

Usage:
  python -m ml.trainer                             # active symbol
  python -m ml.trainer --symbol EURUSD
  python -m ml.trainer --direction SELL
  python -m ml.trainer --all                       # all instruments
  python -m ml.trainer --symbol XAUUSD --universal # universal model (all symbols)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml.database import get_db
from ml.instrument_config import (
    INSTRUMENTS, InstrumentConfig, MODELS_DIR, WALK_FORWARD_SPLITS,
    MIN_SAMPLES_TO_TRAIN, UNIVERSAL_MODEL_ENABLED,
    get_instrument, get_label_profile,
)


MAX_DATASET_ROWS = 1_000_000


# ── Model factory ─────────────────────────────────────────────────────────────

def _build_model(model_type: str, seed: int = 42) -> Any:
    model_type = model_type.lower()
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                use_label_encoder=False, eval_metric="logloss",
                random_state=seed, verbosity=0, n_jobs=-1,
            )
        except ImportError:
            print("[WARN] xgboost not installed — falling back to random_forest")
            model_type = "random_forest"

    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_per_leaf=10,
                random_state=seed, verbosity=-1, n_jobs=-1,
            )
        except ImportError:
            print("[WARN] lightgbm not installed — falling back to random_forest")
            model_type = "random_forest"

    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        random_state=seed, n_jobs=-1,
    )


# ── Dataset builder ───────────────────────────────────────────────────────────

@dataclass
class DatasetDiagnostics:
    symbol:                str
    direction:             str
    label_profile:         str
    features_available:    int = 0
    labels_available:      int = 0
    merged_rows:           int = 0
    feature_without_label: int = 0
    label_without_feature: int = 0
    duplicate_feature_ts:  int = 0
    duplicate_label_ts:    int = 0
    missing_values:        int = 0
    class_0:               int = 0
    class_1:               int = 0

    @property
    def rows_dropped(self) -> int:
        return self.feature_without_label + self.label_without_feature

    def cause(self, min_samples: int = MIN_SAMPLES_TO_TRAIN) -> str:
        if self.features_available == 0:
            return "no feature rows for symbol/profile"
        if self.labels_available == 0:
            return "no label rows for symbol/profile/direction"
        if self.merged_rows == 0:
            return "features and labels have no overlapping candle_id values"
        if self.merged_rows < min_samples:
            return f"merged rows below minimum ({self.merged_rows} < {min_samples})"
        if self.class_0 == 0 or self.class_1 == 0:
            return "only one label class present"
        return "ok"


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return not bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return True


def _build_dataset_with_diagnostics(
    symbol:        str,
    label_profile: str,
    direction:     str = "BUY",
    extra_symbols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int], DatasetDiagnostics]:
    """
    Join features + labels and return the dataset plus exact drop diagnostics.
    Rows are sorted by ts (chronological).  Joining is by candle_id so labels
    and features must refer to the same entry-TF candle.
    """
    db = get_db()
    symbols = [symbol] + (extra_symbols or [])
    diag = DatasetDiagnostics(symbol=symbol, direction=direction, label_profile=label_profile)

    all_rows: list[dict] = []
    for sym in symbols:
        feat_rows  = db.get_features(sym, label_profile, limit=MAX_DATASET_ROWS)
        label_rows = db.get_labels(sym, label_profile, direction=direction, limit=MAX_DATASET_ROWS)

        diag.features_available += len(feat_rows)
        diag.labels_available   += len(label_rows)

        feature_ids = {r["candle_id"] for r in feat_rows}
        label_by_id = {r["candle_id"]: r["label"] for r in label_rows}
        label_ids = set(label_by_id)

        feature_ts = [r["ts"] for r in feat_rows]
        label_ts   = [r["ts"] for r in label_rows]
        diag.duplicate_feature_ts += len(feature_ts) - len(set(feature_ts))
        diag.duplicate_label_ts   += len(label_ts) - len(set(label_ts))
        diag.feature_without_label += len(feature_ids - label_ids)
        diag.label_without_feature += len(label_ids - feature_ids)

        for fr in feat_rows:
            cid = fr["candle_id"]
            if cid not in label_by_id:
                continue
            fvec = json.loads(fr["feature_json"])
            diag.missing_values += sum(1 for v in fvec.values() if _is_missing_value(v))
            fvec["_label"] = label_by_id[cid]
            fvec["_ts"]    = fr["ts"]
            if len(symbols) > 1:
                for s in symbols:
                    fvec[f"sym_{s.lower()}"] = float(sym == s)
            all_rows.append(fvec)

    if not all_rows:
        return np.array([]), np.array([]), [], [], diag

    df = pd.DataFrame(all_rows).sort_values("_ts").reset_index(drop=True)
    ts_list = df["_ts"].tolist()
    y = df["_label"].values.astype(int)
    diag.merged_rows = int(len(y))
    diag.class_0 = int((y == 0).sum())
    diag.class_1 = int((y == 1).sum())

    df = df.drop(columns=["_label", "_ts"])
    feature_names = list(df.columns)
    X = df.fillna(0.0).values.astype(np.float32)
    return X, y, feature_names, ts_list, diag

def _build_dataset(
    symbol:        str,
    label_profile: str,
    direction:     str = "BUY",
    extra_symbols: list[str] | None = None,   # universal model
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """
    Join features + labels tables and return (X, y, feature_names, timestamps).
    Rows are sorted by ts (chronological).
    """
    X, y, feature_names, ts_list, _ = _build_dataset_with_diagnostics(
        symbol, label_profile, direction, extra_symbols,
    )
    return X, y, feature_names, ts_list


# ── Walk-forward evaluator ────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold:      int
    n_train:   int
    n_test:    int
    accuracy:  float
    precision: float
    recall:    float
    f1:        float
    roc_auc:   float


def _evaluate_folds(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_splits: int,
) -> list[FoldResult]:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )
    from sklearn.calibration import CalibratedClassifierCV

    n = len(X)
    fold_size = n // (n_splits + 1)
    results   = []

    for fold in range(n_splits):
        train_end  = fold_size * (fold + 1)
        test_start = train_end
        test_end   = min(train_end + fold_size, n)

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        train_class_counts = np.bincount(y_tr.astype(int), minlength=2)
        if len(np.unique(y_tr)) < 2 or train_class_counts.min() < 3 or len(X_te) < 10:
            continue

        base  = _build_model(model_type)
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)

        results.append(FoldResult(
            fold=fold + 1,
            n_train=len(X_tr),
            n_test=len(X_te),
            accuracy =round(float(accuracy_score(y_te, preds)),  4),
            precision=round(float(precision_score(y_te, preds, zero_division=0)), 4),
            recall   =round(float(recall_score(y_te, preds,    zero_division=0)), 4),
            f1       =round(float(f1_score(y_te, preds,        zero_division=0)), 4),
            roc_auc  =round(float(roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5), 4),
        ))

    return results


@dataclass
class ThresholdResult:
    threshold: float
    trades:    int
    precision: float
    recall:    float
    f1:        float


@dataclass
class ModelDiagnostics:
    holdout_rows:       int
    accuracy:           float
    precision:          float
    recall:             float
    f1:                 float
    roc_auc:            float
    confusion_matrix:   tuple[int, int, int, int]  # tn, fp, fn, tp
    prob_quantiles:     dict[str, float]
    threshold_analysis: list[ThresholdResult]
    top_features:       list[tuple[str, float]]


def _feature_importance(model: Any, feature_names: list[str]) -> list[tuple[str, float]]:
    importances = []
    for calibrated in getattr(model, "calibrated_classifiers_", []):
        estimator = getattr(calibrated, "estimator", None)
        values = getattr(estimator, "feature_importances_", None)
        if values is not None:
            importances.append(np.asarray(values, dtype=float))
    if not importances:
        return []
    avg = np.mean(importances, axis=0)
    pairs = sorted(zip(feature_names, avg), key=lambda x: abs(x[1]), reverse=True)
    return [(name, round(float(val), 6)) for name, val in pairs[:10]]


def _model_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_type: str,
) -> Optional[ModelDiagnostics]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

    if len(X) < 100:
        return None
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]
    train_counts = np.bincount(y_tr.astype(int), minlength=2)
    if len(X_te) < 10 or len(np.unique(y_tr)) < 2 or train_counts.min() < 3:
        return None

    base = _build_model(model_type)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, preds, labels=[0, 1]).ravel()
    quantiles = np.quantile(probs, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    thresholds: list[ThresholdResult] = []
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        p = (probs >= threshold).astype(int)
        thresholds.append(ThresholdResult(
            threshold=threshold,
            trades=int(p.sum()),
            precision=round(float(precision_score(y_te, p, zero_division=0)), 4),
            recall=round(float(recall_score(y_te, p, zero_division=0)), 4),
            f1=round(float(f1_score(y_te, p, zero_division=0)), 4),
        ))

    return ModelDiagnostics(
        holdout_rows=len(X_te),
        accuracy=round(float(accuracy_score(y_te, preds)), 4),
        precision=round(float(precision_score(y_te, preds, zero_division=0)), 4),
        recall=round(float(recall_score(y_te, preds, zero_division=0)), 4),
        f1=round(float(f1_score(y_te, preds, zero_division=0)), 4),
        roc_auc=round(float(roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5), 4),
        confusion_matrix=(int(tn), int(fp), int(fn), int(tp)),
        prob_quantiles={
            "min": round(float(quantiles[0]), 4),
            "p10": round(float(quantiles[1]), 4),
            "p25": round(float(quantiles[2]), 4),
            "p50": round(float(quantiles[3]), 4),
            "p75": round(float(quantiles[4]), 4),
            "p90": round(float(quantiles[5]), 4),
            "max": round(float(quantiles[6]), 4),
        },
        threshold_analysis=thresholds,
        top_features=_feature_importance(model, feature_names),
    )


def _print_dataset_diagnostics(diag: DatasetDiagnostics) -> None:
    print(f"  [{diag.symbol}/{diag.direction}] Dataset assembly:")
    print(f"    Features available : {diag.features_available:,}")
    print(f"    Labels available   : {diag.labels_available:,}")
    print(f"    Merged rows        : {diag.merged_rows:,}")
    print(f"    Rows dropped       : {diag.rows_dropped:,} "
          f"(feature_without_label={diag.feature_without_label:,}, "
          f"label_without_feature={diag.label_without_feature:,})")
    print(f"    Missing values     : {diag.missing_values:,} (filled with 0.0)")
    print(f"    Duplicate ts       : features={diag.duplicate_feature_ts:,}, labels={diag.duplicate_label_ts:,}")
    print(f"    Class distribution : 0={diag.class_0:,}, 1={diag.class_1:,}")
    if diag.merged_rows < MIN_SAMPLES_TO_TRAIN or diag.class_0 == 0 or diag.class_1 == 0:
        print(f"    Cause              : {diag.cause()}")


def _print_model_diagnostics(diag: ModelDiagnostics) -> None:
    tn, fp, fn, tp = diag.confusion_matrix
    q = diag.prob_quantiles
    print("\n  Model diagnostics (last 20% holdout):")
    print(f"    Rows={diag.holdout_rows:,}  Acc={diag.accuracy:.3f}  Prec={diag.precision:.3f}  "
          f"Rec={diag.recall:.3f}  F1={diag.f1:.3f}  AUC={diag.roc_auc:.3f}")
    print(f"    Confusion matrix: TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(f"    Probability distribution: min={q['min']:.3f} p10={q['p10']:.3f} "
          f"p25={q['p25']:.3f} p50={q['p50']:.3f} p75={q['p75']:.3f} "
          f"p90={q['p90']:.3f} max={q['max']:.3f}")
    print("    Threshold analysis:")
    for row in diag.threshold_analysis:
        print(f"      p>={row.threshold:.2f}: trades={row.trades:>5,}  "
              f"Prec={row.precision:.3f}  Rec={row.recall:.3f}  F1={row.f1:.3f}")
    if diag.top_features:
        print("    Top feature importance:")
        for name, score in diag.top_features[:8]:
            print(f"      {name:28s} {score:.6f}")


# ── Final model training + persistence ───────────────────────────────────────

@dataclass
class TrainingResult:
    symbol:        str
    direction:     str
    model_version: str
    model_type:    str
    label_profile: str
    n_samples:     int
    n_features:    int
    feature_names: list[str]
    fold_results:  list[FoldResult]
    avg_roc_auc:   float
    model_path:    str


def train(
    instrument:  InstrumentConfig,
    direction:   str = "BUY",
    verbose:     bool = True,
    extra_symbols: list[str] | None = None,
) -> Optional[TrainingResult]:
    """
    Train (and persist) a model for `instrument` + `direction`.
    Returns TrainingResult or None if insufficient data.
    """
    from sklearn.calibration import CalibratedClassifierCV

    profile_name = instrument.label_profile
    X, y, feature_names, ts_list, dataset_diag = _build_dataset_with_diagnostics(
        instrument.symbol, profile_name, direction, extra_symbols,
    )

    if verbose:
        _print_dataset_diagnostics(dataset_diag)

    if len(X) < MIN_SAMPLES_TO_TRAIN:
        if verbose:
            print(f"  [{instrument.symbol}/{direction}] Only {len(X)} samples — "
                  f"need {MIN_SAMPLES_TO_TRAIN}. Collect more data first.")
            print(f"  [{instrument.symbol}/{direction}] Exact cause: {dataset_diag.cause()}")
        return None

    n_classes = len(np.unique(y))
    if n_classes < 2:
        if verbose:
            print(f"  [{instrument.symbol}/{direction}] All labels are the same class "
                  f"(TP-rate={y.mean():.1%}). Adjust label_point_value or collect more data.")
        return None

    class_counts = np.bincount(y.astype(int), minlength=2)
    if class_counts.min() < 3:
        if verbose:
            print(f"  [{instrument.symbol}/{direction}] Not enough samples per class for calibrated training "
                  f"(class 0={class_counts[0]}, class 1={class_counts[1]}, need at least 3 each).")
        return None

    if verbose:
        tp_rate = y.mean()
        print(f"  [{instrument.symbol}/{direction}] {len(X)} samples, "
              f"{len(feature_names)} features, TP-rate={tp_rate:.1%}")

    # Walk-forward evaluation
    folds = _evaluate_folds(X, y, instrument.model_type, WALK_FORWARD_SPLITS)
    if verbose and folds:
        print(f"\n  Walk-forward results ({WALK_FORWARD_SPLITS} folds):")
        print(f"  {'Fold':>4}  {'Train':>6}  {'Test':>5}  {'Acc':>5}  {'Prec':>5}  "
              f"{'Rec':>5}  {'F1':>5}  {'AUC':>5}")
        for fr in folds:
            print(f"  {fr.fold:>4}  {fr.n_train:>6}  {fr.n_test:>5}  "
                  f"{fr.accuracy:.3f}  {fr.precision:.3f}  {fr.recall:.3f}  "
                  f"{fr.f1:.3f}  {fr.roc_auc:.3f}")

    avg_auc = float(np.mean([f.roc_auc for f in folds])) if folds else 0.5

    diagnostics = _model_diagnostics(X, y, feature_names, instrument.model_type)
    if verbose and diagnostics:
        _print_model_diagnostics(diagnostics)

    # Train final model on all data
    base      = _build_model(instrument.model_type)
    final_mdl = CalibratedClassifierCV(base, method="isotonic", cv=3)
    final_mdl.fit(X, y)

    # Version string: SYMBOL_direction_YYYYMMDD_vN
    today   = datetime.now(timezone.utc).strftime("%Y%m%d")
    version = f"{instrument.symbol}_{direction}_{today}"
    path    = MODELS_DIR / f"{version}.pkl"

    with open(path, "wb") as f:
        pickle.dump({
            "model":         final_mdl,
            "feature_names": feature_names,
            "version":       version,
            "symbol":        instrument.symbol,
            "direction":     direction,
            "model_type":    instrument.model_type,
            "label_profile": profile_name,
            "n_samples":     len(X),
            "avg_roc_auc":   avg_auc,
            "trained_at":    datetime.now(timezone.utc).isoformat(),
        }, f)

    if verbose:
        print(f"\n  Saved: {path}  (AUC={avg_auc:.3f})")

    return TrainingResult(
        symbol=instrument.symbol, direction=direction,
        model_version=version, model_type=instrument.model_type,
        label_profile=profile_name, n_samples=len(X),
        n_features=len(feature_names), feature_names=feature_names,
        fold_results=folds, avg_roc_auc=avg_auc, model_path=str(path),
    )


def train_all(
    direction: str = "BUY",
    verbose: bool = True,
    universal: bool = False,
) -> None:
    if universal:
        # Train one model on all instruments combined
        symbols = list(INSTRUMENTS.keys())
        primary_sym = symbols[0]
        print(f"\n[UNIVERSAL] Training across: {symbols}")
        inst = INSTRUMENTS[primary_sym]
        train(inst, direction=direction, verbose=verbose,
              extra_symbols=symbols[1:])
        return

    for sym, inst in INSTRUMENTS.items():
        print(f"\n[{sym}] Training ({direction})...")
        train(inst, direction=direction, verbose=verbose)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="ML trainer")
    parser.add_argument("--symbol",    default="")
    parser.add_argument("--direction", default="BUY", choices=["BUY", "SELL", "BOTH"])
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--universal", action="store_true", help="Train universal model")
    args = parser.parse_args()

    dirs = ["BUY", "SELL"] if args.direction == "BOTH" else [args.direction]

    if args.all or args.universal:
        for d in dirs:
            train_all(direction=d, universal=args.universal)
    else:
        inst = get_instrument(args.symbol or None)
        for d in dirs:
            print(f"\n[{inst.symbol}] Training {d}...")
            train(inst, direction=d)

    print("\nNext: python scalper.py --paper  (live paper trading with ML signals)")


if __name__ == "__main__":
    _cli()
