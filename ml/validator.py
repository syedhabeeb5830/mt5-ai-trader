"""
ML Platform Validator

Audits the existing ML pipeline end-to-end:
  candles -> features -> labels -> merged samples -> walk-forward label test.

Zero features or zero merged samples is treated as a pipeline failure.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ml.database import get_db
from ml.instrument_config import (
    INSTRUMENTS, InstrumentConfig, MIN_SAMPLES_TO_TRAIN, WALK_FORWARD_SPLITS,
    get_instrument, get_label_profile,
)
from ml.trainer import _build_dataset_with_diagnostics, _build_model


@dataclass
class CandleStats:
    timeframe: str
    count:     int
    distinct:  int
    first_ts:  int | None
    last_ts:   int | None

    @property
    def duplicates(self) -> int:
        return self.count - self.distinct


@dataclass
class FeatureStats:
    count:             int = 0
    duplicate_ts:      int = 0
    missing_values:    int = 0
    rows_with_missing: int = 0


@dataclass
class LabelStats:
    total:        int = 0
    by_direction: dict[str, int] = field(default_factory=dict)
    tp_rate:      dict[str, float] = field(default_factory=dict)
    duplicate_ts: dict[str, int] = field(default_factory=dict)


@dataclass
class BacktestMetrics:
    direction:     str
    folds:         int = 0
    trades:        int = 0
    wins:          int = 0
    losses:        int = 0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    expectancy:    float = 0.0
    max_drawdown:  float = 0.0
    avg_auc:       float = 0.5
    reason:        str = ""


@dataclass
class SymbolValidation:
    symbol:           str
    candles:          dict[str, CandleStats]
    features:         FeatureStats
    labels:           LabelStats
    merged_samples:   dict[str, int]
    merge_causes:     dict[str, str]
    alignment_issues: list[str]
    backtests:        dict[str, BacktestMetrics]
    data_score:       int
    feature_score:    int
    label_score:      int
    model_score:      int
    classification:   str

    @property
    def ok(self) -> bool:
        if self.features.count == 0:
            return False
        if any(v == 0 for v in self.merged_samples.values()):
            return False
        return self.classification != "Broken"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def _candle_stats(symbol: str, timeframes: list[str]) -> dict[str, CandleStats]:
    db = get_db()
    out: dict[str, CandleStats] = {}
    with db._conn() as conn:
        for tf in timeframes:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n, COUNT(DISTINCT ts) AS d,
                       MIN(ts) AS first_ts, MAX(ts) AS last_ts
                FROM candles WHERE symbol=? AND timeframe=?
                """,
                (symbol, tf),
            ).fetchone()
            out[tf] = CandleStats(
                timeframe=tf,
                count=int(row["n"] or 0),
                distinct=int(row["d"] or 0),
                first_ts=row["first_ts"],
                last_ts=row["last_ts"],
            )
    return out


def _feature_stats(symbol: str, profile: str) -> FeatureStats:
    db = get_db()
    stats = FeatureStats()
    with db._conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n, COUNT(DISTINCT ts) AS d
            FROM features WHERE symbol=? AND label_profile=?
            """,
            (symbol, profile),
        ).fetchone()
        stats.count = int(row["n"] or 0)
        stats.duplicate_ts = stats.count - int(row["d"] or 0)

        rows = conn.execute(
            "SELECT feature_json FROM features WHERE symbol=? AND label_profile=? ORDER BY ts ASC",
            (symbol, profile),
        )
        for row in rows:
            row_missing = 0
            try:
                fvec = json.loads(row["feature_json"])
            except json.JSONDecodeError:
                row_missing = 1
                fvec = {}
            row_missing += sum(1 for value in fvec.values() if _is_missing(value))
            if row_missing:
                stats.rows_with_missing += 1
                stats.missing_values += row_missing
    return stats


def _label_stats(symbol: str, profile: str) -> LabelStats:
    db = get_db()
    stats = LabelStats()
    with db._conn() as conn:
        rows = conn.execute(
            """
            SELECT direction, COUNT(*) AS n, COUNT(DISTINCT ts) AS d, AVG(label) AS tp_rate
            FROM labels WHERE symbol=? AND label_profile=?
            GROUP BY direction
            """,
            (symbol, profile),
        ).fetchall()
    for row in rows:
        direction = row["direction"]
        count = int(row["n"] or 0)
        stats.by_direction[direction] = count
        stats.tp_rate[direction] = float(row["tp_rate"] or 0.0)
        stats.duplicate_ts[direction] = count - int(row["d"] or 0)
        stats.total += count
    return stats


def _alignment_issues(inst: InstrumentConfig, candles: dict[str, CandleStats]) -> list[str]:
    issues: list[str] = []
    entry = candles.get(inst.entry_tf)
    if entry is None or entry.count == 0:
        return [f"missing entry timeframe {inst.entry_tf}"]
    for tf in inst.timeframes:
        stat = candles.get(tf)
        if stat is None or stat.count == 0:
            issues.append(f"{tf}: no candles")
            continue
        if stat.duplicates:
            issues.append(f"{tf}: {stat.duplicates} duplicate timestamps")
        if stat.last_ts is not None and entry.first_ts is not None and stat.last_ts < entry.first_ts:
            issues.append(f"{tf}: ends before entry timeframe starts")
        if stat.first_ts is not None and entry.last_ts is not None and stat.first_ts > entry.last_ts:
            issues.append(f"{tf}: starts after entry timeframe ends")
    return issues


def walk_forward_label_backtest(inst: InstrumentConfig, direction: str) -> BacktestMetrics:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

    profile = get_label_profile(inst.label_profile)
    X, y, _feature_names, _ts, diag = _build_dataset_with_diagnostics(
        inst.symbol, inst.label_profile, direction,
    )
    if len(X) < MIN_SAMPLES_TO_TRAIN:
        return BacktestMetrics(direction=direction, reason=diag.cause())

    fold_size = len(X) // (WALK_FORWARD_SPLITS + 1)
    if fold_size < 10:
        return BacktestMetrics(direction=direction, reason="fold size below 10 rows")

    threshold = inst.buy_threshold if direction == "BUY" else (1 - inst.sell_threshold)
    threshold = max(threshold, inst.min_confidence)
    win_r = profile.tp_points / profile.sl_points if profile.sl_points > 0 else 1.0

    pnls: list[float] = []
    aucs: list[float] = []
    folds = 0
    for fold in range(WALK_FORWARD_SPLITS):
        train_end = fold_size * (fold + 1)
        test_end = min(train_end + fold_size, len(X))
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[train_end:test_end], y[train_end:test_end]
        counts = np.bincount(y_tr.astype(int), minlength=2)
        if len(X_te) < 10 or len(np.unique(y_tr)) < 2 or counts.min() < 3:
            continue
        model = CalibratedClassifierCV(_build_model(inst.model_type), method="isotonic", cv=3)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        if len(np.unique(y_te)) > 1:
            aucs.append(float(roc_auc_score(y_te, probs)))
        selected = probs >= threshold
        for label in y_te[selected]:
            pnls.append(win_r if int(label) == 1 else -1.0)
        folds += 1

    if not pnls:
        return BacktestMetrics(
            direction=direction,
            folds=folds,
            avg_auc=round(float(np.mean(aucs)), 4) if aucs else 0.5,
            reason=f"no probabilities cleared threshold {threshold:.2f}",
        )

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    return BacktestMetrics(
        direction=direction,
        folds=folds,
        trades=len(pnls),
        wins=len(wins),
        losses=len(losses),
        win_rate=round(len(wins) / len(pnls) * 100, 1),
        profit_factor=round(gross_profit / gross_loss, 3) if gross_loss > 0 else round(gross_profit, 3),
        expectancy=round(float(np.mean(pnls)), 4),
        max_drawdown=round(float(drawdown.max()), 4) if len(drawdown) else 0.0,
        avg_auc=round(float(np.mean(aucs)), 4) if aucs else 0.5,
    )


def _score_data(inst: InstrumentConfig, candles: dict[str, CandleStats], issues: list[str]) -> int:
    entry_count = candles.get(inst.entry_tf, CandleStats(inst.entry_tf, 0, 0, None, None)).count
    if entry_count == 0:
        return 0
    score = 10
    if entry_count < MIN_SAMPLES_TO_TRAIN:
        score -= 4
    if issues:
        score -= min(6, len(issues) * 2)
    return max(0, score)


def _score_feature(features: FeatureStats, merged_samples: dict[str, int]) -> int:
    if features.count == 0:
        return 0
    score = 10
    if any(v == 0 for v in merged_samples.values()):
        score -= 8
    elif any(v < MIN_SAMPLES_TO_TRAIN for v in merged_samples.values()):
        score -= 4
    if features.rows_with_missing:
        score -= min(3, features.rows_with_missing)
    if features.duplicate_ts:
        score -= 3
    return max(0, score)


def _score_label(labels: LabelStats, merged_samples: dict[str, int]) -> int:
    if labels.total == 0:
        return 0
    score = 10
    for direction in ("BUY", "SELL"):
        if labels.by_direction.get(direction, 0) == 0:
            score -= 5
        tp_rate = labels.tp_rate.get(direction, 0.0)
        if tp_rate <= 0.02 or tp_rate >= 0.98:
            score -= 3
        elif tp_rate <= 0.10 or tp_rate >= 0.90:
            score -= 1
        if merged_samples.get(direction, 0) < MIN_SAMPLES_TO_TRAIN:
            score -= 2
    return max(0, score)


def _score_model(backtests: dict[str, BacktestMetrics]) -> int:
    usable = [b for b in backtests.values() if b.folds > 0]
    if not usable:
        return 0
    avg_auc = float(np.mean([b.avg_auc for b in usable]))
    total_trades = sum(b.trades for b in usable)
    best_pf = max((b.profit_factor for b in usable), default=0.0)
    score = 3
    if avg_auc >= 0.55:
        score += 2
    if avg_auc >= 0.60:
        score += 2
    if avg_auc >= 0.65:
        score += 1
    if total_trades >= 20:
        score += 1
    if best_pf >= 1.0:
        score += 1
    return min(10, score)


def _classification(
    data_score: int,
    feature_score: int,
    label_score: int,
    model_score: int,
    features: FeatureStats,
    merged_samples: dict[str, int],
    backtests: dict[str, BacktestMetrics],
) -> str:
    if features.count == 0 or any(v == 0 for v in merged_samples.values()):
        return "Broken"
    if any(v < MIN_SAMPLES_TO_TRAIN for v in merged_samples.values()):
        return "Not Tradable"
    if model_score < 5:
        return "Not Tradable"
    trades = sum(b.trades for b in backtests.values())
    pf = max((b.profit_factor for b in backtests.values()), default=0.0)
    auc = max((b.avg_auc for b in backtests.values()), default=0.5)
    if trades >= 20 and pf >= 1.0 and auc >= 0.58:
        return "Paper Trade Ready"
    return "Paper Trade Only"


def validate_instrument(inst: InstrumentConfig, run_backtest: bool = True) -> SymbolValidation:
    profile = inst.label_profile
    candles = _candle_stats(inst.symbol, inst.timeframes)
    features = _feature_stats(inst.symbol, profile)
    labels = _label_stats(inst.symbol, profile)
    alignment = _alignment_issues(inst, candles)

    merged_samples: dict[str, int] = {}
    merge_causes: dict[str, str] = {}
    for direction in ("BUY", "SELL"):
        _X, _y, _names, _ts, diag = _build_dataset_with_diagnostics(inst.symbol, profile, direction)
        merged_samples[direction] = diag.merged_rows
        merge_causes[direction] = diag.cause()

    backtests = {
        direction: walk_forward_label_backtest(inst, direction) if run_backtest else BacktestMetrics(direction)
        for direction in ("BUY", "SELL")
    }

    data_score = _score_data(inst, candles, alignment)
    feature_score = _score_feature(features, merged_samples)
    label_score = _score_label(labels, merged_samples)
    model_score = _score_model(backtests)
    classification = _classification(
        data_score, feature_score, label_score, model_score,
        features, merged_samples, backtests,
    )

    return SymbolValidation(
        symbol=inst.symbol,
        candles=candles,
        features=features,
        labels=labels,
        merged_samples=merged_samples,
        merge_causes=merge_causes,
        alignment_issues=alignment,
        backtests=backtests,
        data_score=data_score,
        feature_score=feature_score,
        label_score=label_score,
        model_score=model_score,
        classification=classification,
    )


def _print_symbol_report(report: SymbolValidation) -> None:
    inst = INSTRUMENTS[report.symbol]
    entry_candles = report.candles.get(inst.entry_tf, CandleStats(inst.entry_tf, 0, 0, None, None)).count
    print(f"\n[{report.symbol}] Validation")
    print("-" * 58)
    print(f"  Candles ({inst.entry_tf}) : {entry_candles:,}")
    for tf, stat in report.candles.items():
        print(f"    {tf:>4}: {stat.count:>8,} rows  duplicates={stat.duplicates:,}")
    print(f"  Features          : {report.features.count:,} rows  "
          f"duplicates={report.features.duplicate_ts:,}  "
          f"missing_values={report.features.missing_values:,}")
    print(f"  Labels            : {report.labels.total:,} rows")
    for direction in ("BUY", "SELL"):
        print(f"    {direction:4s}: labels={report.labels.by_direction.get(direction, 0):>8,}  "
              f"TP-rate={report.labels.tp_rate.get(direction, 0.0):>6.1%}  "
              f"merged_samples={report.merged_samples.get(direction, 0):>8,}  "
              f"cause={report.merge_causes.get(direction, 'ok')}")
    if report.alignment_issues:
        print("  Alignment issues:")
        for issue in report.alignment_issues:
            print(f"    - {issue}")

    print("  Walk-forward label backtest:")
    for direction, bt in report.backtests.items():
        print(f"    {direction:4s}: trades={bt.trades:>5,}  win_rate={bt.win_rate:>5.1f}%  "
              f"PF={bt.profit_factor:>5.3f}  expectancy={bt.expectancy:>7.4f}R  "
              f"maxDD={bt.max_drawdown:>7.4f}R  AUC={bt.avg_auc:.3f}"
              + (f"  ({bt.reason})" if bt.reason else ""))

    print(f"  Readiness scores  : Data {report.data_score}/10 | Feature {report.feature_score}/10 | "
          f"Label {report.label_score}/10 | Model {report.model_score}/10")
    print(f"  Classification    : {report.classification}")


def validate(
    symbols: list[str] | None = None,
    all_instruments: bool = False,
    run_backtest: bool = True,
    strict: bool = True,
) -> bool:
    if all_instruments:
        instruments = list(INSTRUMENTS.values())
    elif symbols:
        instruments = [get_instrument(s) for s in symbols]
    else:
        instruments = [get_instrument(None)]

    reports: list[SymbolValidation] = []
    for i, inst in enumerate(instruments, start=1):
        suffix = "with walk-forward backtest" if run_backtest else "structural checks only"
        print(f"  [{i}/{len(instruments)}] Validating {inst.symbol} ({suffix})...", flush=True)
        reports.append(validate_instrument(inst, run_backtest=run_backtest))

    print("\nValidation summary")
    print("=" * 96)
    print(f"{'Symbol':8s} {'Candles':>9s} {'Features':>9s} {'Labels':>9s} "
          f"{'BUY samples':>11s} {'SELL samples':>12s} {'Class':>18s}")
    for report in reports:
        inst = INSTRUMENTS[report.symbol]
        entry_count = report.candles.get(inst.entry_tf, CandleStats(inst.entry_tf, 0, 0, None, None)).count
        print(f"{report.symbol:8s} {entry_count:9,} {report.features.count:9,} {report.labels.total:9,} "
              f"{report.merged_samples.get('BUY', 0):11,} {report.merged_samples.get('SELL', 0):12,} "
              f"{report.classification:>18s}")

    for report in reports:
        _print_symbol_report(report)

    ok = all(report.ok for report in reports)
    if strict and not ok:
        print("\nVALIDATION FAILED: at least one instrument has zero features, zero samples, or a broken pipeline.")
    elif ok:
        print("\nVALIDATION PASSED: all selected instruments have non-zero features and merged samples.")
    else:
        print("\nVALIDATION WARNING: issues were found, but strict mode is disabled.")
    return ok or not strict
