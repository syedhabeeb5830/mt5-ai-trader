"""
Tick Feed — polls MT5 API for real-time price ticks.
Returns TickData dataclass. Thread-safe deque for momentum engine.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote

import httpx

import config


@dataclass
class Tick:
    time_ms: int
    bid: float
    ask: float
    mid: float = field(init=False)
    spread: float = field(init=False)

    def __post_init__(self):
        self.mid    = (self.bid + self.ask) / 2
        self.spread = round(self.ask - self.bid, 5)


class TickFeed:
    """
    Async tick poller. Fills a rolling deque of the last N ticks.
    Consumer reads `feed.ticks` — always fresh.
    """

    def __init__(self, maxlen: int = 50):
        self.ticks: deque[Tick] = deque(maxlen=maxlen)
        self.last_error: Optional[str] = None
        self._running = False
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def latest(self) -> Optional[Tick]:
        return self.ticks[-1] if self.ticks else None

    async def start(self) -> None:
        self._running = True
        self._client = httpx.AsyncClient(
            base_url=config.MT5_API,
            headers=config.HEADERS,
            timeout=2.0,
        )
        await self._poll_loop()

    async def stop(self) -> None:
        self._running = False
        if self._client:
            await self._client.aclose()

    async def _poll_loop(self) -> None:
        url = f"/symbol/{quote(config.SYMBOL, safe='')}/tick"
        interval = config.POLL_MS / 1000.0

        while self._running:
            t0 = time.monotonic()
            try:
                resp = await self._client.get(url)
                if resp.status_code == 200:
                    d = resp.json()
                    tick = Tick(
                        time_ms=d["time_msc"],
                        bid=float(d["bid"]),
                        ask=float(d["ask"]),
                    )
                    if not self.ticks or tick.time_ms > self.ticks[-1].time_ms:
                        self.ticks.append(tick)
                    self.last_error = None
                else:
                    self.last_error = f"HTTP {resp.status_code}"
            except Exception as e:
                self.last_error = str(e)

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0, interval - elapsed))
