"""
Live Strategy Runner — Phase 3
─────────────────────────────────────────────────────────────────────────────
Bridges any BaseStrategy (OHLCV-based) to the live scalper loop.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │ scalper.py                                               │
  │   every POLL_MS → runner.get_signal(bid, ask) → Signal  │
  └─────────────────────────────────────────────────────────┘
           ↓ refreshes every OHLCV_REFRESH_SEC
  ┌─────────────────────────────────────────────────────────┐
  │ LiveBarFeed                                              │
  │   GET /bars/{symbol}/{tf}?count=N  →  pd.DataFrame      │
  └─────────────────────────────────────────────────────────┘
           ↓
  ┌─────────────────────────────────────────────────────────┐
  │ BaseStrategy.evaluate(bars)  →  TradeSetup | None       │
  └─────────────────────────────────────────────────────────┘

If MT5 API is unreachable, the runner falls back to Signal.WAIT and logs
a warning once per refresh cycle (no flooding).

Trade hour filter:
  config.TRADE_HOURS_UTC = "14-18"   → only signal between 14:00–18:00 UTC
  config.TRADE_HOURS_UTC = ""        → no filter (24h)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, TYPE_CHECKING

import pandas as pd

import config
from momentum import Signal
from strategies.base import BaseStrategy, StrategySignal, TradeSetup

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


# ── MT5 timeframe → yfinance / API interval ───────────────────────────────────

_TF_TO_BARS = {
    "1M":  1440, "5M": 288, "15M": 96,
    "1H":  24,   "4H": 6,   "D1":  1,
}


# ── Trade hour filter ─────────────────────────────────────────────────────────

def _parse_hours(spec: str) -> Optional[tuple]:
    """
    Parse TRADE_HOURS_UTC string like "14-18" → (14, 18).
    Returns None if spec is empty or invalid.
    """
    spec = spec.strip()
    if not spec:
        return None
    try:
        parts = spec.split("-")
        return int(parts[0]), int(parts[1])
    except Exception:
        logger.warning("Invalid TRADE_HOURS_UTC=%r — hour filter disabled", spec)
        return None


def within_trade_hours(spec: str = "") -> bool:
    """Return True if the current UTC hour is within the configured trade window."""
    hours = _parse_hours(spec or config.TRADE_HOURS_UTC)
    if hours is None:
        return True   # no filter
    start, end = hours
    hour = datetime.now(timezone.utc).hour
    if start <= end:
        return start <= hour < end
    # overnight wrap (e.g. "22-02")
    return hour >= start or hour < end


# ── Live bar fetcher ──────────────────────────────────────────────────────────

class LiveBarFeed:
    """
    Fetches OHLCV bars from the MT5 REST API.

    GET /bars/{symbol}/{tf}?count={n}
    Response JSON:  [{"time": ..., "open": ..., "high": ..., "low": ...,
                      "close": ..., "volume": ...}, ...]
    """

    def __init__(self, client: "httpx.AsyncClient"):
        self._client = client
        self._cache:      Dict[str, pd.DataFrame] = {}
        self._last_fetch: Dict[str, float]        = {}

    async def get(self, symbol: str, tf: str, count: int) -> Optional[pd.DataFrame]:
        """
        Fetch `count` OHLCV bars for `symbol` / `tf`.
        Returns a DataFrame with columns [open, high, low, close, volume]
        and a DatetimeIndex, or None on failure.
        """
        url = f"/bars/{symbol}/{tf}"
        try:
            resp = await self._client.get(url, params={"count": count}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("LiveBarFeed.get(%s/%s) failed: %s", symbol, tf, e)
            return None

        if not data:
            return None

        try:
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            df["volume"] = df.get("volume", 0).astype(float)
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.warning("LiveBarFeed parse error (%s/%s): %s", symbol, tf, e)
            return None

    async def refresh_all(
        self, symbol: str, timeframes: List[str], days: int
    ) -> Dict[str, pd.DataFrame]:
        """Refresh all timeframes and return the bars dict."""
        bars: Dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            count = _TF_TO_BARS.get(tf, 24) * (days + 5)
            df    = await self.get(symbol, tf, count)
            if df is not None:
                bars[tf] = df
            elif tf in self._cache:
                bars[tf] = self._cache[tf]   # fall back to stale cache
        self._cache = bars
        return bars


# ── Live strategy runner ──────────────────────────────────────────────────────

class LiveStrategyRunner:
    """
    Wraps a BaseStrategy for use inside the live scalper loop.

    Usage:
        runner = LiveStrategyRunner(strategy_instance, http_client)
        await runner.start()
        ...
        signal = runner.get_signal(bid, ask)   # fast, cached
    """

    def __init__(
        self,
        strategy:    BaseStrategy,
        client:      "httpx.AsyncClient",
        symbol:      str   = "",
        days:        int   = 2,
        refresh_sec: int   = 0,
    ):
        self._strategy    = strategy
        self._feed        = LiveBarFeed(client)
        self._symbol      = symbol or config.SYMBOL
        self._days        = days
        self._refresh_sec = refresh_sec or config.OHLCV_REFRESH_SEC
        self._timeframes  = self._resolve_timeframes()
        self._entry_tf    = self._resolve_entry_tf()

        self._last_signal:  Signal              = Signal.WAIT
        self._last_setup:   Optional[TradeSetup] = None
        self._last_refresh: float               = 0.0
        self._refresh_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._warn_once:    bool                = False

    def _resolve_timeframes(self) -> List[str]:
        try:
            from backtest.run import STRATEGY_TIMEFRAMES
            return STRATEGY_TIMEFRAMES.get(self._strategy.name, ["1H"])
        except Exception:
            return ["1H"]

    def _resolve_entry_tf(self) -> str:
        try:
            from backtest.run import STRATEGY_ENTRY_TF
            return STRATEGY_ENTRY_TF.get(self._strategy.name, "1H")
        except Exception:
            return "1H"

    # ── Background refresh loop ──────────────────────────────────────────────

    async def _refresh_loop(self) -> None:
        """Continuously refresh bars and re-evaluate the strategy."""
        while True:
            await self._do_refresh()
            await asyncio.sleep(self._refresh_sec)

    async def _do_refresh(self) -> None:
        bars = await self._feed.refresh_all(self._symbol, self._timeframes, self._days)

        if not bars or self._entry_tf not in bars:
            if not self._warn_once:
                logger.warning(
                    "LiveStrategyRunner: no bars for %s/%s — holding WAIT",
                    self._symbol, self._entry_tf,
                )
                self._warn_once = True
            return

        self._warn_once = False
        self._last_refresh = time.monotonic()

        try:
            setup = self._strategy.evaluate(bars)
        except Exception as e:
            logger.warning("strategy.evaluate() error: %s", e)
            setup = None

        if setup is None:
            self._last_signal = Signal.WAIT
        elif setup.signal == StrategySignal.BUY:
            self._last_signal = Signal.BUY
        elif setup.signal == StrategySignal.SELL:
            self._last_signal = Signal.SELL
        else:
            self._last_signal = Signal.WAIT

        self._last_setup = setup

    async def start(self) -> None:
        """Do an immediate refresh, then launch the background loop."""
        await self._do_refresh()
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    def stop(self) -> None:
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()

    # ── Public signal API ────────────────────────────────────────────────────

    def get_signal(self, bid: float, ask: float) -> Signal:
        """
        Returns the current cached signal.
        Applies the TRADE_HOURS_UTC filter before returning.
        Thread-safe: just reads a field; refresh happens in async task.
        """
        if not within_trade_hours():
            return Signal.WAIT
        return self._last_signal

    @property
    def last_setup(self) -> Optional[TradeSetup]:
        return self._last_setup

    @property
    def strategy_name(self) -> str:
        return self._strategy.name

    @property
    def seconds_since_refresh(self) -> float:
        if self._last_refresh == 0:
            return float("inf")
        return time.monotonic() - self._last_refresh

    def debug_snapshot(self) -> dict:
        """Compatible with MomentumEngine.debug_snapshot() for the dashboard."""
        return {
            "strategy":      self._strategy.name,
            "signal":        self._last_signal.value,
            "last_refresh":  f"{self.seconds_since_refresh:.0f}s ago",
            "entry_tf":      self._entry_tf,
            "hour_filter":   config.TRADE_HOURS_UTC or "none",
        }


# ── Factory — build runner from strategy name ─────────────────────────────────

def build_runner(
    strategy_name: str,
    client:        "httpx.AsyncClient",
    symbol:        str   = "",
    days:          int   = 2,
    refresh_sec:   int   = 0,
) -> LiveStrategyRunner:
    """
    Instantiate a strategy by name and wrap it in a LiveStrategyRunner.
    Raises KeyError if strategy_name is not in REGISTRY.
    """
    from strategies import REGISTRY
    if strategy_name not in REGISTRY:
        raise KeyError(
            f"Unknown strategy {strategy_name!r}. "
            f"Available: {list(REGISTRY.keys())}"
        )
    strategy = REGISTRY[strategy_name]()
    return LiveStrategyRunner(
        strategy    = strategy,
        client      = client,
        symbol      = symbol,
        days        = days,
        refresh_sec = refresh_sec,
    )


# ── Auto-select from last saved recommendation ────────────────────────────────

def auto_select_strategy(fallback: str = "momentum_scalper") -> str:
    """
    Read logs/recommendation.json and return the recommended strategy name.
    If no recommendation file exists or has no winner, returns `fallback`.
    """
    from analytics.strategy_selector import load_recommendation
    rec = load_recommendation()
    if rec and rec.get("recommended"):
        winner = rec["recommended"]
        ts     = rec.get("generated_at", "unknown time")
        print(f"[AUTO] Using recommended strategy: {winner}  (from {ts})")
        return winner
    print(f"[AUTO] No recommendation found — falling back to: {fallback}")
    return fallback
