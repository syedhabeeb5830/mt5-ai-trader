"""
mt5-ai-trader — Main Orchestrator
─────────────────────────────────────────────────────────────────────────────
How it works every POLL_MS milliseconds:
  1. TickFeed   → fetches latest price tick from MT5
  2. Momentum   → evaluates BUY / SELL / WAIT signal
  3. RiskGuard  → checks daily limits before entry
  4. OrderManager → places / monitors trades
  5. Dashboard  → refreshes terminal display

Usage:
  python scalper.py            # live mode
  python scalper.py --paper    # paper/simulation mode (no real orders)
  python scalper.py --status   # check MT5 connection and print account info
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

import config
from dashboard import Dashboard
from db.logger import TradeLogger
from momentum import MomentumEngine, Signal
from order_manager import OrderManager
from risk_guard import RiskGuard
from tick_feed import TickFeed


# ── One-shot status check ─────────────────────────────────────────────────────

async def print_status() -> None:
    async with httpx.AsyncClient(base_url=config.MT5_API, headers=config.HEADERS, timeout=5) as c:
        print("── HEALTH ──────────────────────────────────")
        try:
            h = await c.get("/health")
            print(json.dumps(h.json(), indent=2))
        except Exception as e:
            print(f"error: {e}")

        print("\n── ACCOUNT ─────────────────────────────────")
        try:
            a = await c.get("/account")
            print(json.dumps(a.json(), indent=2))
        except Exception as e:
            print(f"error: {e}")

        print("\n── LIVE TICK ───────────────────────────────")
        try:
            from urllib.parse import quote
            t = await c.get(f"/symbol/{quote(config.SYMBOL, safe='')}/tick")
            print(json.dumps(t.json(), indent=2))
        except Exception as e:
            print(f"error: {e}")

        print("\n── OPEN POSITIONS ──────────────────────────")
        try:
            p = await c.get("/positions")
            print(json.dumps(p.json(), indent=2))
        except Exception as e:
            print(f"error: {e}")


# ── Core engine loop ──────────────────────────────────────────────────────────

async def run_scalper() -> None:
    feed    = TickFeed(maxlen=100)
    engine  = MomentumEngine()
    manager = OrderManager()
    guard   = RiskGuard(manager)
    dash    = Dashboard(feed, manager, guard)
    logger  = TradeLogger()

    tick_count   = 0
    cooldown_end = 0.0
    COOLDOWN_SEC = 2.0

    # ── Pre-flight: verify MT5 is reachable ───────────────────────────────
    async with httpx.AsyncClient(base_url=config.MT5_API, headers=config.HEADERS, timeout=5) as c:
        try:
            h = await c.get("/health")
            health = h.json()
        except Exception as e:
            print(f"[ERROR] Cannot reach MT5 API at {config.MT5_API}: {e}")
            print("Make sure mt5_server.py is running. See README for setup.")
            return

        if not health.get("mt5_connected"):
            print(f"[WARN] MT5 terminal is NOT connected (status: {health.get('status')})")
            if "--force" not in sys.argv:
                print("Start MT5 terminal and try again, or use --force to run in paper mode.")
                return
            config.PAPER = True
            print("[INFO] Forced paper mode — no live trades will be placed.")

    print(f"[OK] MT5 connected. Starting engine...\n")

    await logger.init()
    feed_task = asyncio.create_task(feed.start())

    with dash.start():
        try:
            while True:
                await asyncio.sleep(config.POLL_MS / 1000)

                tick = feed.latest
                if tick is None:
                    continue

                tick_count += 1

                if tick_count % 10 == 0:
                    await manager.sync_closed_from_api()

                await manager.check_timeouts(tick.mid)

                signal  = engine.evaluate(feed.ticks)
                mom_dbg = engine.debug_snapshot(feed.ticks)

                now = time.time()
                if (
                    signal != Signal.WAIT
                    and guard.can_open_position()
                    and now > cooldown_end
                    and tick.spread <= config.MAX_SPREAD_POINTS
                ):
                    trade = await manager.enter(signal, tick.ask, tick.bid)
                    if trade:
                        cooldown_end = now + COOLDOWN_SEC
                        await logger.log_trade_open(trade, tick)

                # Log any newly closed trades
                for t in manager.trades:
                    if t.closed and not getattr(t, "_logged", False):
                        await logger.log_trade_close(t)
                        t._logged = True  # type: ignore[attr-defined]

                dash.update(signal, mom_dbg, tick_count)

        except KeyboardInterrupt:
            pass
        finally:
            feed_task.cancel()
            if manager.open_count > 0:
                mid = feed.latest.mid if feed.latest else 0
                await manager.close_all(mid)

    # Session summary
    summary = {
        "total_trades": manager.total_today,
        "realized_pnl": manager.realized_pnl,
        "halted": guard.halted,
        "halt_reason": guard.halt_reason,
    }
    await logger.log_session(summary)

    print(f"\n── Session Summary ─────────────────────────────")
    print(f"Total trades : {summary['total_trades']}")
    print(f"Realized PnL : ${summary['realized_pnl']:+.2f}")
    print(f"Halted       : {summary['halted']} {summary['halt_reason']}")
    print(f"\nTrades saved to logs. Run the AI analyzer:")
    print(f"  python -m ai_loop.analyst")


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--status" in sys.argv:
        asyncio.run(print_status())
        sys.exit(0)

    if "--paper" in sys.argv:
        config.PAPER = True

    mode = "[PAPER]" if config.PAPER else "[LIVE]"
    print(f"MT5 AI Trader {mode}")
    print(f"Symbol: {config.SYMBOL}  SL: {config.SL_POINTS}pt  TP: {config.TP_POINTS}pt  Vol: {config.VOLUME}lot")
    print(f"MT5 API: {config.MT5_API}")
    print(f"Press Ctrl+C to stop.\n")
    asyncio.run(run_scalper())
