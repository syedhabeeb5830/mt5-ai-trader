"""
Terminal Dashboard — renders live bot state using Rich.
Refreshes 4x per second via Live display.
"""

from __future__ import annotations

import time
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config
from momentum import Signal
from order_manager import OpenTrade, OrderManager
from risk_guard import RiskGuard
from tick_feed import Tick, TickFeed


console = Console()


def build_layout(
    feed: TickFeed,
    manager: OrderManager,
    guard: RiskGuard,
    signal: Signal,
    momentum_dbg: dict,
    tick_count: int,
) -> Layout:

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )

    # ── Header ──────────────────────────────────────────────────────────────
    tick = feed.latest
    mode_tag = "[yellow][PAPER][/yellow]" if config.PAPER else "[green][LIVE][/green]"
    if tick:
        price_str = (
            f"[bold white]BID [cyan]{tick.bid:.2f}[/cyan]  "
            f"ASK [cyan]{tick.ask:.2f}[/cyan]  "
            f"SPREAD [yellow]{tick.spread:.3f}[/yellow][/bold white]"
        )
    else:
        price_str = "[dim]waiting for tick...[/dim]"

    layout["header"].update(Panel(
        f"[bold]MT5 AI Trader  {config.SYMBOL}[/bold]  {mode_tag}   "
        f"{price_str}   "
        f"[dim]{datetime.now().strftime('%H:%M:%S.%f')[:-3]}[/dim]",
        style="bold blue",
    ))

    # ── Signal Panel ────────────────────────────────────────────────────────
    sig_color = {"BUY": "green", "SELL": "red", "WAIT": "dim"}.get(signal.value, "white")
    sig_text = Text(f"  {signal.value}  ", style=f"bold {sig_color} on black")
    is_live_strategy = "strategy" in momentum_dbg or "last_refresh" in momentum_dbg

    mom_table = Table.grid(padding=(0, 1))
    mom_table.add_column(style="dim")
    mom_table.add_column(style="white")
    mom_table.add_row("Signal", sig_text)

    if is_live_strategy:
        mom_table.add_row("Strategy", f"[cyan]{momentum_dbg.get('strategy', 'unknown')}[/cyan]")
        mom_table.add_row("Entry TF", str(momentum_dbg.get('entry_tf', '')))
        mom_table.add_row("Refresh", str(momentum_dbg.get('last_refresh', '')))
        mom_table.add_row("Hour filter", str(momentum_dbg.get('hour_filter', 'none')))
        if momentum_dbg.get("rr") is not None:
            mom_table.add_row("RR", f"{float(momentum_dbg.get('rr')):.2f}")
        if momentum_dbg.get("entry") is not None:
            mom_table.add_row("Entry", f"{float(momentum_dbg.get('entry')):.2f}")
        if momentum_dbg.get("sl") is not None:
            mom_table.add_row("SL", f"{float(momentum_dbg.get('sl')):.2f}")
        if momentum_dbg.get("tp") is not None:
            mom_table.add_row("TP", f"{float(momentum_dbg.get('tp')):.2f}")
        if momentum_dbg.get("reason"):
            mom_table.add_row("Reason", str(momentum_dbg.get('reason', '')))
        mom_table.add_row("Ticks", str(tick_count))
    else:
        mom_table.add_row("UP ticks", f"[green]{momentum_dbg.get('up', 0)}[/green]")
        mom_table.add_row("DN ticks", f"[red]{momentum_dbg.get('down', 0)}[/red]")
        mom_table.add_row("Move pts", f"[yellow]{momentum_dbg.get('move', 0.0):.3f}[/yellow]")
        mom_table.add_row("Spread",   f"[yellow]{momentum_dbg.get('spread', 0.0):.3f}[/yellow]")
        mom_table.add_row("Ticks",    str(tick_count))
    mom_table.add_row("SL / TP",  f"{config.SL_POINTS} / {config.TP_POINTS} pts")
    mom_table.add_row("Volume",   f"{config.VOLUME} lot")

    panel_title = "[bold]ML Strategy[/bold]" if is_live_strategy else "[bold]Momentum Signal[/bold]"
    layout["left"].update(Panel(mom_table, title=panel_title, border_style="cyan"))

    # ── Positions Table ──────────────────────────────────────────────────────
    pos_table = Table("Ticket", "Dir", "Entry", "SL", "TP", "Age(s)", "Est PnL",
                      header_style="bold", border_style="dim")

    open_trades = [t for t in manager.trades if not t.closed]
    for t in open_trades:
        age = int(time.time() - t.opened_at)
        dir_color = "green" if t.direction == Signal.BUY else "red"
        cur = feed.latest
        if cur:
            pnl_est = (cur.mid - t.entry) * (1 if t.direction == Signal.BUY else -1) * t.volume * 100
        else:
            pnl_est = 0.0
        pnl_color = "green" if pnl_est >= 0 else "red"
        pos_table.add_row(
            str(t.ticket),
            f"[{dir_color}]{t.direction.value}[/{dir_color}]",
            f"{t.entry:.2f}",
            f"[red]{t.sl:.2f}[/red]",
            f"[green]{t.tp:.2f}[/green]",
            str(age),
            f"[{pnl_color}]{pnl_est:+.2f}[/{pnl_color}]",
        )

    if not open_trades:
        pos_table.add_row("[dim]no open positions[/dim]", "", "", "", "", "", "")

    layout["right"].update(Panel(pos_table, title="[bold]Open Positions[/bold]", border_style="cyan"))

    # ── Footer ───────────────────────────────────────────────────────────────
    pnl_color = "green" if manager.realized_pnl >= 0 else "red"
    halt_str = f"[red]HALTED: {guard.halt_reason}[/red]" if guard.halted else "[green]trading[/green]"
    feed_err = f"  [red]feed: {feed.last_error}[/red]" if feed.last_error else ""

    layout["footer"].update(Panel(
        f"Trades today: [bold]{manager.total_today}[/bold] / {config.MAX_DAILY_TRADES}   "
        f"Realized PnL: [{pnl_color}]${manager.realized_pnl:+.2f}[/{pnl_color}]   "
        f"Limit: ${config.DAILY_LOSS_LIMIT:.2f}   "
        f"Status: {halt_str}{feed_err}",
        style="dim",
    ))

    return layout


class Dashboard:

    def __init__(self, feed: TickFeed, manager: OrderManager, guard: RiskGuard):
        self._feed    = feed
        self._manager = manager
        self._guard   = guard
        self._signal  = Signal.WAIT
        self._mom_dbg: dict = {}
        self._tick_count = 0
        self._live: Live | None = None

    def update(self, signal: Signal, mom_dbg: dict, tick_count: int) -> None:
        self._signal     = signal
        self._mom_dbg    = mom_dbg
        self._tick_count = tick_count
        if self._live:
            self._live.update(build_layout(
                self._feed, self._manager, self._guard,
                self._signal, self._mom_dbg, self._tick_count,
            ))

    def start(self) -> "Live":
        self._live = Live(
            build_layout(self._feed, self._manager, self._guard, Signal.WAIT, {}, 0),
            refresh_per_second=4,
            screen=True,
        )
        return self._live
