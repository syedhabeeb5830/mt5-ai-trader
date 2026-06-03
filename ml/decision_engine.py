"""
ML Platform — Decision Engine
─────────────────────────────────────────────────────────────────────────────
Converts probability scores into BUY / SELL / WAIT signals.
Thresholds are fully configurable per instrument (InstrumentConfig).

Decision logic:
  P_buy  >= buy_threshold   AND  P_buy  >= min_confidence  →  BUY
  P_sell >= (1-sell_threshold) AND P_sell >= min_confidence  →  SELL
  else                                                       →  WAIT

Both BUY and SELL models are queried. The one with higher conviction wins.
If both are above threshold (extremely rare), the higher probability wins.

Output: DecisionResult with signal, probability, explanation string, and
        full list of SHAP top features for the dashboard.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ml.instrument_config import InstrumentConfig, get_instrument
from ml.probability_engine import ProbabilityEngine, ProbabilityResult
from momentum import Signal   # reuse existing BUY/SELL/WAIT enum


@dataclass
class DecisionResult:
    signal:      Signal
    probability: float             # leading model probability
    direction:   str               # "BUY" | "SELL" | "WAIT"
    explanation: str               # human-readable reason
    buy_result:  Optional[ProbabilityResult] = None
    sell_result: Optional[ProbabilityResult] = None

    @property
    def acted(self) -> bool:
        return self.signal != Signal.WAIT

    def to_dict(self) -> dict:
        top = []
        leading = self.buy_result if self.signal == Signal.BUY else self.sell_result
        if leading:
            top = [
                {"name": f.name, "value": f.value, "shap": f.shap}
                for f in leading.top_features[:4]
            ]
        return {
            "signal":      self.signal.value,
            "probability": self.probability,
            "explanation": self.explanation,
            "top_features": top,
        }


class DecisionEngine:
    """
    Wraps ProbabilityEngine for a single instrument and applies threshold logic.
    Instantiate once per session and call decide() for every signal opportunity.
    """

    def __init__(self, instrument: InstrumentConfig) -> None:
        self._inst   = instrument
        self._engine = ProbabilityEngine(instrument)

    def decide(self, features: dict[str, float]) -> DecisionResult:
        """
        features : flat feature dict from feature_engine.get_live_features()
        Returns DecisionResult with BUY / SELL / WAIT signal.
        """
        buy_r  = self._engine.predict(features, "BUY")
        sell_r = self._engine.predict(features, "SELL")

        cfg = self._inst
        buy_thresh  = cfg.buy_threshold
        sell_thresh = cfg.sell_threshold    # P(BUY TP) <= this means short side is confident
        min_conf    = cfg.min_confidence

        has_buy  = buy_r  is not None and buy_r.probability  >= buy_thresh  and buy_r.probability  >= min_conf
        has_sell = sell_r is not None and sell_r.probability >= (1 - sell_thresh) and sell_r.probability >= min_conf

        # Resolve conflict: take higher conviction
        if has_buy and has_sell:
            has_buy  = buy_r.probability  >= sell_r.probability
            has_sell = not has_buy

        if has_buy:
            return DecisionResult(
                signal=Signal.BUY,
                probability=buy_r.probability,
                direction="BUY",
                explanation=self._explain(buy_r, "BUY"),
                buy_result=buy_r,
                sell_result=sell_r,
            )

        if has_sell:
            return DecisionResult(
                signal=Signal.SELL,
                probability=sell_r.probability,
                direction="SELL",
                explanation=self._explain(sell_r, "SELL"),
                buy_result=buy_r,
                sell_result=sell_r,
            )

        # WAIT — build a brief explanation for the dashboard
        p_buy  = buy_r.probability  if buy_r  else 0.0
        p_sell = sell_r.probability if sell_r else 0.0
        explanation = (
            f"WAIT  "
            f"P(buy)={p_buy:.0%} (need {buy_thresh:.0%})  "
            f"P(sell)={p_sell:.0%}"
        )
        return DecisionResult(
            signal=Signal.WAIT,
            probability=max(p_buy, p_sell),
            direction="WAIT",
            explanation=explanation,
            buy_result=buy_r,
            sell_result=sell_r,
        )

    @staticmethod
    def _explain(result: ProbabilityResult, direction: str) -> str:
        lines = [f"{direction}  {result.probability:.0%}  [{result.model_version}]"]
        for fc in result.top_features[:4]:
            tag   = "+" if fc.shap > 0 else "−"
            lines.append(f"  {tag} {fc.name}={fc.value:.3g}")
        return "\n".join(lines)

    @property
    def model_version(self) -> str:
        v = self._engine.version("BUY")
        return v or "no_model"

    def reload_models(self) -> None:
        """Force reload after retraining."""
        self._engine.reload("BUY")
        self._engine.reload("SELL")


# ── Convenience factory ───────────────────────────────────────────────────────

def get_decision_engine(symbol: str | None = None) -> DecisionEngine:
    return DecisionEngine(get_instrument(symbol))
