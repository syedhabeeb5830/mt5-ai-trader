"""
ML Platform — Instrument & Profile Configuration
─────────────────────────────────────────────────────────────────────────────
All ML-related configuration lives here. Nothing in the ML codebase
is hardcoded for a specific symbol, timeframe, SL/TP, or model type.

Add a new instrument by appending to INSTRUMENTS.
Add a new label profile by appending to LABEL_PROFILES.
Everything else adapts automatically.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

ML_DIR      = Path("ml")
MODELS_DIR  = Path(os.getenv("ML_MODELS_DIR",  "models"))
DATA_DB     = Path(os.getenv("ML_DATA_DB",     "ml/data/market.db"))
LOGS_DIR    = Path(os.getenv("ML_LOGS_DIR",    "logs/ml"))

for _d in (MODELS_DIR, DATA_DB.parent, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Timeframe helpers (shared across ml modules) ─────────────────────────────

TF_TO_MINUTES: dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}


# ── Label Profiles ────────────────────────────────────────────────────────────

@dataclass
class LabelProfile:
    """
    Defines the TP/SL/horizon used to create supervised-learning labels.

    tp_points  : price points (in instrument quote units) the trade must move
                 in the BUY/SELL direction to be labelled 1.
    sl_points  : price points move against the trade before labelled 0.
    horizon_bars: maximum bars (in entry timeframe) to scan forward.
                 If neither TP nor SL is hit within this window → label = 0.
    """
    name:          str
    tp_points:     float
    sl_points:     float
    horizon_bars:  int
    description:   str = ""

LABEL_PROFILES: dict[str, LabelProfile] = {
    "scalp": LabelProfile(
        name="scalp", tp_points=4.0, sl_points=2.0, horizon_bars=12,
        description="Scalp: 4pt TP / 2pt SL / 12 bars forward",
    ),
    "intraday": LabelProfile(
        name="intraday", tp_points=10.0, sl_points=5.0, horizon_bars=24,
        description="Intraday: 10pt TP / 5pt SL / 24 bars forward",
    ),
    "swing": LabelProfile(
        name="swing", tp_points=50.0, sl_points=25.0, horizon_bars=48,
        description="Swing: 50pt TP / 25pt SL / 48 bars forward",
    ),
    "momentum": LabelProfile(
        name="momentum", tp_points=5.0, sl_points=1.0, horizon_bars=6,
        description="Momentum: matches live scalper defaults (TP=5, SL=1)",
    ),
}

# Which profile to use for training (override with ML_LABEL_PROFILE env var)
ACTIVE_LABEL_PROFILE: str = os.getenv("ML_LABEL_PROFILE", "momentum")


# ── Instrument Registry ───────────────────────────────────────────────────────

@dataclass
class InstrumentConfig:
    """
    Per-instrument configuration. Controls data collection, feature
    computation, model selection, and decision thresholds.
    """
    symbol:            str
    display_name:      str
    point_value:       float          # $ value of 1 point at 0.01 lot
    pip_size:          float          # smallest meaningful move (e.g. 0.01 for FX, 1.0 for Gold)
    spread_typical:    float          # typical spread in points (used in labeling)

    # Timeframes to collect and use for features
    timeframes:        list[str]      # all TFs to store (e.g. ["M1","M5","M15","H1"])
    entry_tf:          str            # TF whose candle close triggers a signal

    # Model settings
    label_profile:     str            # key into LABEL_PROFILES
    model_type:        str            # "xgboost" | "lightgbm" | "random_forest"
    model_mode:        str            # "per_instrument" | "universal"

    # Decision thresholds
    buy_threshold:     float          # P(TP) >= this → BUY
    sell_threshold:    float          # P(TP) <= this → SELL (inverse: P of opposite direction)
    min_confidence:    float = 0.55   # below this → always WAIT regardless of direction

    # Session filter (UTC hours, "" = 24h)
    trade_hours_utc:   str = ""

    # TP/SL scaling: profile tp_points × label_point_value = raw price distance.
    # Set to pip_size for FX (EURUSD=0.0001), 1.0 for Gold/BTC (profile in raw $).
    label_point_value: float = 1.0

    # Extra arbitrary settings passed through to feature engine / trainer
    extra:             dict[str, Any] = field(default_factory=dict)


INSTRUMENTS: dict[str, InstrumentConfig] = {
    "XAUUSD": InstrumentConfig(
        symbol="XAUUSD",   display_name="Gold",
        point_value=0.10,  pip_size=0.1,      spread_typical=0.30,
        timeframes=["M1", "M5", "M15", "H1", "H4"],
        entry_tf="M5",
        label_profile="momentum",
        model_type=os.getenv("ML_MODEL_TYPE_XAUUSD", "xgboost"),
        model_mode="per_instrument",
        buy_threshold=float(os.getenv("ML_BUY_THRESH_XAUUSD", "0.72")),
        sell_threshold=float(os.getenv("ML_SELL_THRESH_XAUUSD", "0.28")),
        trade_hours_utc=os.getenv("ML_HOURS_XAUUSD", "14-18"),
    ),
    "EURUSD": InstrumentConfig(
        symbol="EURUSD",   display_name="Euro/Dollar",
        point_value=0.10,  pip_size=0.0001,   spread_typical=0.00015,
        timeframes=["M5", "M15", "H1", "H4"],
        entry_tf="M15",
        label_profile="intraday",
        model_type=os.getenv("ML_MODEL_TYPE_EURUSD", "lightgbm"),
        model_mode="per_instrument",
        buy_threshold=float(os.getenv("ML_BUY_THRESH_EURUSD", "0.65")),
        sell_threshold=float(os.getenv("ML_SELL_THRESH_EURUSD", "0.35")),
        trade_hours_utc=os.getenv("ML_HOURS_EURUSD", "8-17"),
        label_point_value=0.0001,  # intraday tp=10 → 10 pips = 0.0010
    ),
    "GBPUSD": InstrumentConfig(
        symbol="GBPUSD",   display_name="Pound/Dollar",
        point_value=0.10,  pip_size=0.0001,   spread_typical=0.0002,
        timeframes=["M5", "M15", "H1", "H4"],
        entry_tf="M15",
        label_profile="intraday",
        model_type=os.getenv("ML_MODEL_TYPE_GBPUSD", "lightgbm"),
        model_mode="per_instrument",
        buy_threshold=float(os.getenv("ML_BUY_THRESH_GBPUSD", "0.68")),
        sell_threshold=float(os.getenv("ML_SELL_THRESH_GBPUSD", "0.32")),
        trade_hours_utc=os.getenv("ML_HOURS_GBPUSD", "8-17"),
        label_point_value=0.0001,  # intraday tp=10 → 10 pips = 0.0010
    ),
    "USDJPY": InstrumentConfig(
        symbol="USDJPY",   display_name="Dollar/Yen",
        point_value=0.10,  pip_size=0.01,     spread_typical=0.02,
        timeframes=["M5", "M15", "H1", "H4"],
        entry_tf="M15",
        label_profile="intraday",
        model_type=os.getenv("ML_MODEL_TYPE_USDJPY", "xgboost"),
        model_mode="per_instrument",
        buy_threshold=float(os.getenv("ML_BUY_THRESH_USDJPY", "0.68")),
        sell_threshold=float(os.getenv("ML_SELL_THRESH_USDJPY", "0.32")),
        trade_hours_utc=os.getenv("ML_HOURS_USDJPY", "0-9"),
        label_point_value=0.01,    # intraday tp=10 → 10 pips = 0.10 yen
    ),
    "BTCUSD": InstrumentConfig(
        symbol="BTCUSD",   display_name="Bitcoin/Dollar",
        point_value=0.01,  pip_size=1.0,      spread_typical=5.0,
        timeframes=["M1", "M5", "M15", "H1"],
        entry_tf="M1",
        label_profile="scalp",
        model_type=os.getenv("ML_MODEL_TYPE_BTCUSD", "xgboost"),
        model_mode="per_instrument",
        buy_threshold=float(os.getenv("ML_BUY_THRESH_BTCUSD", "0.80")),
        sell_threshold=float(os.getenv("ML_SELL_THRESH_BTCUSD", "0.20")),
        trade_hours_utc="",        # BTC is 24h
        label_point_value=50.0,   # scalp tp=4 → $200 move (BTC prices in $k)
    ),
}

# Which symbol to trade in this session (override with SYMBOL env var or scalper --symbol flag)
ACTIVE_SYMBOL: str = os.getenv("SYMBOL", "XAUUSD")

SYMBOL_ALIASES: dict[str, str] = {
    "GOLD": "XAUUSD",
}


def get_instrument(symbol: str | None = None) -> InstrumentConfig:
    """Return config for `symbol`, falling back to ACTIVE_SYMBOL."""
    key = SYMBOL_ALIASES.get((symbol or ACTIVE_SYMBOL).upper(), (symbol or ACTIVE_SYMBOL).upper())
    if key not in INSTRUMENTS:
        raise ValueError(
            f"Unknown instrument '{key}'. "
            f"Add it to ml/instrument_config.py INSTRUMENTS dict. "
            f"Known: {list(INSTRUMENTS)}"
        )
    return INSTRUMENTS[key]


def get_label_profile(name: str | None = None) -> LabelProfile:
    """Return label profile by name, falling back to ACTIVE_LABEL_PROFILE."""
    key = name or ACTIVE_LABEL_PROFILE
    if key not in LABEL_PROFILES:
        raise ValueError(
            f"Unknown label profile '{key}'. "
            f"Known: {list(LABEL_PROFILES)}"
        )
    return LABEL_PROFILES[key]


# ── Retraining schedule ───────────────────────────────────────────────────────

RETRAIN_SCHEDULE: str   = os.getenv("ML_RETRAIN_SCHEDULE", "weekly")   # daily|weekly|monthly
MIN_SAMPLES_TO_TRAIN:int = int(os.getenv("ML_MIN_SAMPLES",  "500"))
WALK_FORWARD_SPLITS: int = int(os.getenv("ML_WF_SPLITS",    "5"))


# ── Universal model mode ──────────────────────────────────────────────────────

# If True, a single model is trained across ALL instruments (adds symbol as a feature).
# If False, each instrument gets its own model file.
UNIVERSAL_MODEL_ENABLED: bool = os.getenv("ML_UNIVERSAL_MODEL", "false").lower() in ("true", "1")
