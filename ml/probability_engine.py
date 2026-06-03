"""
ML Platform — Probability Engine
─────────────────────────────────────────────────────────────────────────────
Loads a persisted model and produces P(TP before SL) for a feature vector.

Thread-safe model cache: models are loaded once per process and reused.

Key design:
  • Model file path: models/<SYMBOL>_<DIRECTION>_<DATE>.pkl
  • Latest model auto-selected by mtime (most recently trained)
  • Returns a ProbabilityResult with the probability AND top SHAP contributors
  • SHAP is computed only if shap package is installed (graceful fallback)

Usage in live code:
  engine = ProbabilityEngine(instrument)
  result = engine.predict(feature_vector)    # feature_vector: dict[str, float]
  print(result.probability)                  # 0.81
  print(result.top_features)                 # [{"name":"rsi14_m5","value":54.2,"shap":0.12}]
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import threading

import numpy as np

from ml.instrument_config import MODELS_DIR, InstrumentConfig, get_instrument


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class FeatureContribution:
    name:  str
    value: float
    shap:  float          # positive = pushes toward TP, negative = toward SL

@dataclass
class ProbabilityResult:
    probability:   float                           # P(TP before SL), 0–1
    direction:     str                             # "BUY" | "SELL"
    model_version: str
    top_features:  list[FeatureContribution] = field(default_factory=list)
    n_features:    int = 0
    model_type:    str = ""

    @property
    def confident(self) -> bool:
        return self.probability >= 0.55

    def explain(self, max_features: int = 4) -> str:
        """Human-readable explanation string."""
        lines = [f"{self.direction}  P={self.probability:.0%}  ({self.model_version})"]
        for fc in self.top_features[:max_features]:
            arrow = "↑" if fc.shap > 0 else "↓"
            lines.append(f"  {arrow} {fc.name}={fc.value:.3g}  (Δ{fc.shap:+.3f})")
        return "\n".join(lines)


# ── Model bundle ──────────────────────────────────────────────────────────────

@dataclass
class _ModelBundle:
    model:          object
    feature_names:  list[str]
    version:        str
    symbol:         str
    direction:      str
    model_type:     str
    label_profile:  str
    # SHAP explainer (lazy-initialised)
    _explainer:     object | None = field(default=None, repr=False)
    _lock:          threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_explainer(self) -> object | None:
        with self._lock:
            if self._explainer is not None:
                return self._explainer
            try:
                import shap
                base_model = getattr(self.model, "base_estimator", self.model)
                if hasattr(base_model, "get_booster"):       # XGBoost
                    self._explainer = shap.TreeExplainer(base_model)
                elif hasattr(base_model, "booster_"):        # LightGBM
                    self._explainer = shap.TreeExplainer(base_model)
                else:
                    self._explainer = shap.Explainer(self.model.predict_proba)
            except Exception:
                self._explainer = None
        return self._explainer


# ── Engine ────────────────────────────────────────────────────────────────────

class ProbabilityEngine:
    """
    Load-once, predict-many probability engine for a single instrument.
    Keeps one bundle per (symbol, direction) pair in an in-process cache.
    """

    _cache: dict[str, "_ModelBundle"] = {}
    _lock  = threading.Lock()

    def __init__(self, instrument: InstrumentConfig) -> None:
        self._instrument = instrument

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        features:  dict[str, float],
        direction: str = "BUY",
    ) -> Optional[ProbabilityResult]:
        """
        features : flat dict from feature_engine.build_feature_vector()
        direction: "BUY" | "SELL"
        Returns ProbabilityResult or None if no model available.
        """
        bundle = self._load(self._instrument.symbol, direction)
        if bundle is None:
            return None

        X = self._align(features, bundle.feature_names)
        prob = float(bundle.model.predict_proba(X)[0, 1])
        top  = self._shap_contributions(X, bundle, features, n=6)

        return ProbabilityResult(
            probability=round(prob, 4),
            direction=direction,
            model_version=bundle.version,
            top_features=top,
            n_features=len(bundle.feature_names),
            model_type=bundle.model_type,
        )

    def version(self, direction: str = "BUY") -> str | None:
        bundle = self._load(self._instrument.symbol, direction)
        return bundle.version if bundle else None

    def reload(self, direction: str = "BUY") -> None:
        """Force re-load model from disk (useful after retraining)."""
        key = f"{self._instrument.symbol}_{direction}"
        with self.__class__._lock:
            self.__class__._cache.pop(key, None)

    # ── Internal ──────────────────────────────────────────────────────────────

    @classmethod
    def _load(cls, symbol: str, direction: str) -> Optional["_ModelBundle"]:
        key = f"{symbol}_{direction}"
        with cls._lock:
            if key in cls._cache:
                return cls._cache[key]
        # Find latest model file for this symbol + direction
        pattern = f"{symbol}_{direction}_*.pkl"
        candidates = sorted(MODELS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return None
        path = candidates[-1]   # most recently trained
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[ProbabilityEngine] Failed to load {path}: {e}")
            return None
        bundle = _ModelBundle(
            model=data["model"], feature_names=data["feature_names"],
            version=data["version"], symbol=data["symbol"],
            direction=data.get("direction", direction),
            model_type=data.get("model_type", "unknown"),
            label_profile=data.get("label_profile", ""),
        )
        with cls._lock:
            cls._cache[key] = bundle
        return bundle

    @staticmethod
    def _align(features: dict[str, float], names: list[str]) -> np.ndarray:
        """Build a (1, n_features) array aligned to `names`. Missing → 0."""
        row = np.zeros((1, len(names)), dtype=np.float32)
        for i, n in enumerate(names):
            v = features.get(n, 0.0)
            row[0, i] = float(v) if v == v else 0.0   # NaN guard
        return row

    @staticmethod
    def _shap_contributions(
        X: np.ndarray,
        bundle: "_ModelBundle",
        features: dict[str, float],
        n: int = 6,
    ) -> list[FeatureContribution]:
        try:
            explainer = bundle.get_explainer()
            if explainer is None:
                return _fallback_contributions(features, bundle.feature_names, n)
            import shap
            shap_vals = explainer(X)
            # shap_vals shape depends on model wrapper; normalise to 1D
            vals = shap_vals.values if hasattr(shap_vals, "values") else shap_vals
            if vals.ndim == 3:
                vals = vals[0, :, 1]   # multi-class, class 1
            elif vals.ndim == 2:
                vals = vals[0]
            else:
                vals = vals.flatten()

            idx = np.argsort(np.abs(vals))[::-1][:n]
            return [
                FeatureContribution(
                    name=bundle.feature_names[i],
                    value=round(float(features.get(bundle.feature_names[i], 0)), 4),
                    shap=round(float(vals[i]), 4),
                )
                for i in idx
            ]
        except Exception:
            return _fallback_contributions(features, bundle.feature_names, n)


def _fallback_contributions(
    features: dict[str, float],
    names: list[str],
    n: int,
) -> list[FeatureContribution]:
    """Rough importance proxy when SHAP is unavailable: largest abs feature values."""
    scored = sorted(
        [(nm, features.get(nm, 0.0)) for nm in names],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return [FeatureContribution(name=nm, value=round(v, 4), shap=0.0) for nm, v in scored[:n]]


# ── Convenience factory ───────────────────────────────────────────────────────

def get_engine(symbol: str | None = None) -> ProbabilityEngine:
    inst = get_instrument(symbol)
    return ProbabilityEngine(inst)
