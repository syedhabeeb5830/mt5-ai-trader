from strategies.base import BaseStrategy, StrategySignal
from strategies.ema7_tbm_v2 import EMA7TBMv2
from strategies.ema7_tbm_v3 import EMA7TBMv3
from strategies.sqrt_levels_v4 import SQRTLevelsV4
from strategies.momentum_scalper import MomentumScalper
from strategies.ml_scalper import MLScalper

REGISTRY = {
    "ema7_tbm_v2":      EMA7TBMv2,
    "ema7_tbm_v3":      EMA7TBMv3,
    "sqrt_levels_v4":   SQRTLevelsV4,
    "momentum_scalper": MomentumScalper,
    "ml_scalper":       MLScalper,
}

def get_strategy(name: str) -> BaseStrategy:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: '{name}'. Available: {list(REGISTRY.keys())}")
    return cls()
