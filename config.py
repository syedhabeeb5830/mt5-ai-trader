"""
mt5-ai-trader — Configuration
All values loaded from .env file. Never hardcode credentials here.
Copy .env.example → .env and fill in your values.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── MT5 API ───────────────────────────────────────────────────────────────────
MT5_API  = os.getenv("MT5_API_URL", "http://localhost:8000")
MT5_KEY  = os.getenv("MT5_API_KEY", "")
HEADERS  = {"X-API-Key": MT5_KEY} if MT5_KEY else {}

# ── AI Provider ───────────────────────────────────────────────────────────────
AI_PROVIDER  = os.getenv("AI_PROVIDER", "claude")     # claude | openai | gemini
AI_API_KEY   = os.getenv("AI_API_KEY", "")
AI_MODEL     = os.getenv("AI_MODEL", "claude-sonnet-4-6")
AI_BASE_URL  = os.getenv("AI_BASE_URL", "")           # override for OpenRouter etc.

# ── Symbol ────────────────────────────────────────────────────────────────────
SYMBOL = os.getenv("SYMBOL", "XAUUSD")

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")  # empty = use CSV fallback

# ── Scalp Parameters ─────────────────────────────────────────────────────────
SL_POINTS  = float(os.getenv("SL_POINTS",  "1.0"))
TP_POINTS  = float(os.getenv("TP_POINTS",  "5.0"))
VOLUME     = float(os.getenv("VOLUME",     "0.01"))

# ── Entry Signal Tuning ───────────────────────────────────────────────────────
POLL_MS           = int(os.getenv("POLL_MS",           "200"))
MOMENTUM_WINDOW   = int(os.getenv("MOMENTUM_WINDOW",   "8"))
MIN_DIRECTION_PCT = float(os.getenv("MIN_DIRECTION_PCT","0.75"))
MIN_MOVE_POINTS   = float(os.getenv("MIN_MOVE_POINTS", "0.5"))
MAX_SPREAD_POINTS = float(os.getenv("MAX_SPREAD_POINTS","0.8"))
DEVIATION_POINTS  = int(os.getenv("DEVIATION_POINTS",  "3"))

# ── Position Management ───────────────────────────────────────────────────────
MAX_POSITIONS        = int(os.getenv("MAX_POSITIONS",        "1"))
POSITION_TIMEOUT_SEC = int(os.getenv("POSITION_TIMEOUT_SEC", "120"))

# ── Risk Guard ────────────────────────────────────────────────────────────────
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "-50.0"))
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES",   "30"))

# ── Misc ──────────────────────────────────────────────────────────────────────
MAGIC   = 420001
COMMENT = "mt5-ai-trader"
PAPER   = os.getenv("PAPER", "false").lower() in ("true", "1", "yes")

# ── Dynamic Strategy Selection ────────────────────────────────────────────────
# Set ACTIVE_STRATEGY=auto to auto-select via logs/recommendation.json,
# or set a specific name e.g. ACTIVE_STRATEGY=ema7_tbm_v2
ACTIVE_STRATEGY   = os.getenv("ACTIVE_STRATEGY",   "momentum_scalper")

# Restrict live trading to specific UTC hours. Format: "START-END" e.g. "14-18".
# Leave empty for 24h trading.
TRADE_HOURS_UTC   = os.getenv("TRADE_HOURS_UTC",   "")

# How often (in seconds) the live strategy runner refreshes OHLCV bars from MT5.
OHLCV_REFRESH_SEC = int(os.getenv("OHLCV_REFRESH_SEC", "60"))
