"""
mt5-ai-trader — Main Orchestrator
─────────────────────────────────────────────────────────────────────────────
How it works every POLL_MS milliseconds:
  1. TickFeed   → fetches latest price tick from MT5
  2. Signal     → momentum engine OR live strategy runner
  3. RiskGuard  → checks daily limits before entry
  4. OrderManager → places / monitors trades
  5. Dashboard  → refreshes terminal display

Usage:
  python scalper.py                        # use ACTIVE_STRATEGY from .env
  python scalper.py --paper                # paper/simulation mode
  python scalper.py --strategy ema7_tbm_v2 # pin a specific strategy
  python scalper.py --auto-strategy        # auto-select from recommendation
  python scalper.py --status               # check MT5 connection
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

async def run_scalper(strategy_name: str = "") -> None:
    """
    Main trading loop.

    strategy_name : the strategy to use for signals.
      "momentum_scalper" (or "")  → use MomentumEngine (tick-based, no MT5 bars needed)
      any other REGISTRY name      → use LiveStrategyRunner (OHLCV-based, needs MT5 bars)
    """
    feed    = TickFeed(maxlen=100)
    manager = OrderManager()
    guard   = RiskGuard(manager)
    dash    = Dashboard(feed, manager, guard)
    logger  = TradeLogger()

    tick_count   = 0
    cooldown_end = 0.0
    COOLDOWN_SEC = 2.0

    # Resolve strategy name (default from config)
    active_strategy = strategy_name or config.ACTIVE_STRATEGY or "momentum_scalper"

    async with httpx.AsyncClient(base_url=config.MT5_API, headers=config.HEADERS, timeout=5) as c:
        # ── Pre-flight: verify MT5 is reachable ──────────────────────────────
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

        # ── Build signal source ───────────────────────────────────────────────
        live_runner = None
        if active_strategy == "momentum_scalper":
            signal_engine: MomentumEngine = MomentumEngine()
            print(f"[OK] MT5 connected. Strategy: momentum_scalper (tick-momentum)\n")
        else:
            from analytics.live_strategy import build_runner
            live_runner   = build_runner(active_strategy, c)
            signal_engine = None  # type: ignore[assignment]
            await live_runner.start()
            print(f"[OK] MT5 connected. Strategy: {active_strategy} "
                  f"(OHLCV/{', '.join(live_runner._timeframes)})\n")

        hour_filter = config.TRADE_HOURS_UTC
        if hour_filter:
            print(f"[INFO] Trade hour filter: {hour_filter} UTC\n")

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

                    await manager.check_paper_exits(tick.bid, tick.ask)
                    await manager.check_timeouts(tick.mid)

                    # ── Get signal from active strategy ───────────────────────
                    if live_runner is not None:
                        signal  = live_runner.get_signal(tick.bid, tick.ask)
                        dbg     = live_runner.debug_snapshot()
                    else:
                        signal  = signal_engine.evaluate(feed.ticks)
                        dbg     = signal_engine.debug_snapshot(feed.ticks)

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

                    dash.update(signal, dbg, tick_count)

            except KeyboardInterrupt:
                pass
            finally:
                if live_runner is not None:
                    live_runner.stop()
                feed_task.cancel()
                if manager.open_count > 0:
                    mid = feed.latest.mid if feed.latest else 0
                    await manager.close_all(mid)

    # Session summary
    summary = {
        "total_trades":  manager.total_today,
        "realized_pnl":  manager.realized_pnl,
        "halted":        guard.halted,
        "halt_reason":   guard.halt_reason,
        "strategy_used": active_strategy,
    }
    await logger.log_session(summary)

    print(f"\n── Session Summary ─────────────────────────────")
    print(f"Strategy     : {active_strategy}")
    print(f"Total trades : {summary['total_trades']}")
    print(f"Realized PnL : ${summary['realized_pnl']:+.2f}")
    print(f"Halted       : {summary['halted']} {summary['halt_reason']}")
    print(f"\nTrades saved to logs. Run the AI analyzer:")
    print(f"  python -m ai_loop.analyst")


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    if "--status" in sys.argv:
        asyncio.run(print_status())
        sys.exit(0)

    parser = argparse.ArgumentParser(description="MT5 AI Trader — Live Scalper")
    parser.add_argument("--paper",         action="store_true",
                        help="Paper/simulation mode (no real orders)")
    parser.add_argument("--force",         action="store_true",
                        help="Force paper mode even if MT5 is not connected")
    parser.add_argument("--status",        action="store_true",
                        help="Check MT5 connection and print account info")
    parser.add_argument("--strategy",      default=None,
                        help="Pin a specific strategy by name, e.g. ema7_tbm_v2")
    parser.add_argument("--auto-strategy", action="store_true",
                        help="Auto-select best strategy from logs/recommendation.json")
    args, _ = parser.parse_known_args()

    if args.paper:
        config.PAPER = True

    # Resolve which strategy to run
    chosen_strategy = ""
    if args.strategy:
        from strategies import REGISTRY
        if args.strategy not in REGISTRY:
            print(f"[ERROR] Unknown strategy: {args.strategy!r}")
            print(f"Available: {list(REGISTRY.keys())}")
            sys.exit(1)
        chosen_strategy = args.strategy
        print(f"[INFO] Strategy pinned: {chosen_strategy}")
    elif args.auto_strategy:
        from analytics.live_strategy import auto_select_strategy
        chosen_strategy = auto_select_strategy(fallback=config.ACTIVE_STRATEGY)
    else:
        chosen_strategy = config.ACTIVE_STRATEGY or "momentum_scalper"
        print(f"[INFO] Strategy from config: {chosen_strategy}")

    mode = "[PAPER]" if config.PAPER else "[LIVE]"
    print(f"MT5 AI Trader {mode}")
    print(f"Symbol  : {config.SYMBOL}  SL: {config.SL_POINTS}pt  TP: {config.TP_POINTS}pt  Vol: {config.VOLUME}lot")
    print(f"MT5 API : {config.MT5_API}")
    if config.TRADE_HOURS_UTC:
        print(f"Hours   : {config.TRADE_HOURS_UTC} UTC")
    print(f"Press Ctrl+C to stop.\n")

    asyncio.run(run_scalper(chosen_strategy))
