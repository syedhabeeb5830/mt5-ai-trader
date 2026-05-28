"""
Trade Logger — persists trades and sessions to PostgreSQL (or CSV fallback).
Falls back to CSV files if DATABASE_URL is not set — no setup required.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import config


class TradeLogger:
    """
    Writes trades and session summaries to either:
    - PostgreSQL (if DATABASE_URL is set in .env)
    - CSV files in logs/ directory (fallback, always works)
    """

    def __init__(self):
        self._pool = None
        self._use_db = bool(config.DATABASE_URL)
        self._logs_dir = Path("logs")
        self._logs_dir.mkdir(exist_ok=True)
        self._session_start = datetime.now(timezone.utc)

    async def init(self) -> None:
        if not self._use_db:
            return
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(config.DATABASE_URL, min_size=1, max_size=3)
            await self._ensure_schema()
        except Exception as e:
            print(f"[DB] Could not connect to database: {e}")
            print("[DB] Falling back to CSV logging.")
            self._use_db = False

    async def _ensure_schema(self) -> None:
        schema = Path(__file__).parent / "schema.sql"
        if schema.exists() and self._pool:
            sql = schema.read_text()
            async with self._pool.acquire() as conn:
                await conn.execute(sql)

    # ── Trade logging ─────────────────────────────────────────────────────────

    async def log_trade_open(self, trade: Any, tick: Any) -> None:
        if self._use_db and self._pool:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO trades
                       (ticket, symbol, direction, entry, sl, tp, volume, opened_at,
                        spread_at_entry, paper)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)""",
                    trade.ticket, config.SYMBOL, trade.direction.value,
                    trade.entry, trade.sl, trade.tp, trade.volume,
                    datetime.now(timezone.utc),
                    tick.spread if tick else None,
                    config.PAPER,
                )
        else:
            self._csv_append("trades.csv", {
                "ticket":  trade.ticket,
                "symbol":  config.SYMBOL,
                "direction": trade.direction.value,
                "entry":   trade.entry,
                "sl":      trade.sl,
                "tp":      trade.tp,
                "volume":  trade.volume,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "spread":  tick.spread if tick else "",
                "paper":   config.PAPER,
            })

    async def log_trade_close(self, trade: Any) -> None:
        if self._use_db and self._pool:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """UPDATE trades SET closed_at=$1, pnl=$2
                       WHERE ticket=$3""",
                    datetime.now(timezone.utc), trade.closed_pnl, trade.ticket,
                )
        else:
            self._csv_append("closed_trades.csv", {
                "ticket":    trade.ticket,
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "pnl":       trade.closed_pnl,
            })

    async def log_session(self, summary: dict) -> None:
        snap = {
            "SL_POINTS":  config.SL_POINTS,
            "TP_POINTS":  config.TP_POINTS,
            "VOLUME":     config.VOLUME,
            "MOMENTUM_WINDOW": config.MOMENTUM_WINDOW,
        }
        if self._use_db and self._pool:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO sessions
                       (started_at, ended_at, total_trades, total_pnl,
                        halted, halt_reason, config_snapshot)
                       VALUES ($1,$2,$3,$4,$5,$6,$7)""",
                    self._session_start,
                    datetime.now(timezone.utc),
                    summary.get("total_trades", 0),
                    summary.get("realized_pnl", 0),
                    summary.get("halted", False),
                    summary.get("halt_reason", ""),
                    json.dumps(snap),
                )
        else:
            self._csv_append("sessions.csv", {
                "started_at":   self._session_start.isoformat(),
                "ended_at":     datetime.now(timezone.utc).isoformat(),
                "total_trades": summary.get("total_trades", 0),
                "total_pnl":    summary.get("realized_pnl", 0),
                "halted":       summary.get("halted", False),
                "halt_reason":  summary.get("halt_reason", ""),
                "config":       json.dumps(snap),
            })

    # ── Query ─────────────────────────────────────────────────────────────────

    async def get_recent_trades(self, days: int = 7) -> List[dict]:
        if self._use_db and self._pool:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT ticket, symbol, direction, entry, pnl, opened_at, closed_at
                       FROM trades
                       WHERE opened_at > NOW() - INTERVAL '%s days'
                       ORDER BY opened_at DESC""" % days
                )
                return [dict(r) for r in rows]
        else:
            return self._csv_read("closed_trades.csv")

    async def get_recent_sessions(self, days: int = 7) -> List[dict]:
        if self._use_db and self._pool:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT * FROM sessions
                       WHERE started_at > NOW() - INTERVAL '%s days'
                       ORDER BY started_at DESC""" % days
                )
                return [dict(r) for r in rows]
        else:
            return self._csv_read("sessions.csv")

    # ── CSV helpers ───────────────────────────────────────────────────────────

    def _csv_append(self, filename: str, row: dict) -> None:
        path = self._logs_dir / filename
        is_new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def _csv_read(self, filename: str) -> List[dict]:
        path = self._logs_dir / filename
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
