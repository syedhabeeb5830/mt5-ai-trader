"""
Order Manager — places and closes orders via MT5 REST API.

For each trade:
  - Market order (BUY/SELL)
  - Stop loss at entry ± SL_POINTS
  - Take profit at entry ± TP_POINTS

All methods are async.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

import config
from momentum import Signal


@dataclass
class OpenTrade:
    ticket:     int
    direction:  Signal
    entry:      float
    sl:         float
    tp:         float
    volume:     float
    opened_at:  float = field(default_factory=time.time)
    closed:     bool  = False
    closed_pnl: float = 0.0


class OrderManager:

    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=config.MT5_API,
            headers=config.HEADERS,
            timeout=5.0,
        )
        self.trades: List[OpenTrade] = []

    # ── Query ────────────────────────────────────────────────────────────────

    async def open_positions(self) -> List[dict]:
        """Live positions from MT5 (filtered by magic number)."""
        try:
            r = await self._client.get("/positions")
            if r.status_code == 200:
                return [p for p in r.json() if p.get("magic") == config.MAGIC]
        except Exception:
            pass
        return []

    async def get_position(self, ticket: int) -> Optional[dict]:
        try:
            r = await self._client.get(f"/position/{ticket}")
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    # ── Place ────────────────────────────────────────────────────────────────

    async def enter(self, signal: Signal, ask: float, bid: float) -> Optional[OpenTrade]:
        if signal == Signal.BUY:
            entry_price = ask
            sl = round(entry_price - config.SL_POINTS, 2)
            tp = round(entry_price + config.TP_POINTS, 2)
        elif signal == Signal.SELL:
            entry_price = bid
            sl = round(entry_price + config.SL_POINTS, 2)
            tp = round(entry_price - config.TP_POINTS, 2)
        else:
            return None

        payload = {
            "symbol":     config.SYMBOL,
            "order_type": signal.value,
            "volume":     config.VOLUME,
            "sl":         sl,
            "tp":         tp,
            "deviation":  config.DEVIATION_POINTS,
            "magic":      config.MAGIC,
            "comment":    config.COMMENT,
        }

        if config.PAPER:
            import random
            ticket = random.randint(100000, 999999)
            trade = OpenTrade(
                ticket=ticket, direction=signal,
                entry=entry_price, sl=sl, tp=tp,
                volume=config.VOLUME,
            )
            self.trades.append(trade)
            return trade

        try:
            r = await self._client.post("/order", json=payload)
            result = r.json()
        except Exception as e:
            print(f"[order] request error: {e}")
            return None

        if result.get("success"):
            trade = OpenTrade(
                ticket=result["order"],
                direction=signal,
                entry=result.get("price", entry_price),
                sl=sl, tp=tp,
                volume=config.VOLUME,
            )
            self.trades.append(trade)
            return trade
        else:
            print(f"[order] failed: {result.get('retcode_description')}")
            return None

    # ── Close ────────────────────────────────────────────────────────────────

    async def close(self, trade: OpenTrade, current_price: float) -> bool:
        if trade.closed:
            return True

        if config.PAPER:
            trade.closed = True
            if trade.direction == Signal.BUY:
                trade.closed_pnl = (current_price - trade.entry) * trade.volume * 100
            else:
                trade.closed_pnl = (trade.entry - current_price) * trade.volume * 100
            return True

        try:
            r = await self._client.delete(f"/position/{trade.ticket}")
            result = r.json()
            if result.get("success"):
                trade.closed = True
                return True
        except Exception as e:
            print(f"[close] error: {e}")
        return False

    # ── Monitor & Timeout ────────────────────────────────────────────────────

    async def check_timeouts(self, current_price: float) -> None:
        if config.POSITION_TIMEOUT_SEC <= 0:
            return
        now = time.time()
        for trade in self.trades:
            if not trade.closed and (now - trade.opened_at) > config.POSITION_TIMEOUT_SEC:
                await self.close(trade, current_price)

    async def sync_closed_from_api(self) -> None:
        """Mark local trades as closed if no longer in MT5 positions."""
        if config.PAPER:
            return
        live_tickets = {p["ticket"] for p in await self.open_positions()}
        for trade in self.trades:
            if not trade.closed and trade.ticket not in live_tickets:
                trade.closed = True

    # ── Stats ────────────────────────────────────────────────────────────────

    @property
    def open_count(self) -> int:
        return sum(1 for t in self.trades if not t.closed)

    @property
    def total_today(self) -> int:
        return len(self.trades)

    @property
    def realized_pnl(self) -> float:
        return sum(t.closed_pnl for t in self.trades if t.closed)

    async def close_all(self, current_price: float) -> None:
        for trade in self.trades:
            if not trade.closed:
                await self.close(trade, current_price)
