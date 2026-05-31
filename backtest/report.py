"""
Backtest Report — pretty terminal output + CSV export.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from backtest.engine import BacktestResult
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


def print_report(result: BacktestResult) -> None:
    pf_color = "green" if result.profit_factor >= 1.0 else "red"
    pnl_color = "green" if result.total_pnl >= 0 else "red"

    console.print()
    console.print(Panel(
        f"[bold]Backtest: {result.strategy_name.upper()}[/bold]  |  {result.symbol}  |  "
        f"{result.start_date} → {result.end_date}",
        style="bold blue",
    ))

    # ── Key metrics grid ──────────────────────────────────────────────────────
    t = Table.grid(padding=(0, 4))
    t.add_column(style="dim")
    t.add_column(style="bold white")
    t.add_column(style="dim")
    t.add_column(style="bold white")

    t.add_row("Total Trades",   str(result.total_trades),
              "Win Rate",       f"{result.win_rate}%")
    t.add_row("Winning",        f"[green]{result.winning_trades}[/green]",
              "Losing",         f"[red]{result.losing_trades}[/red]")
    t.add_row("Profit Factor",  f"[{pf_color}]{result.profit_factor}[/{pf_color}]",
              "Total PnL",      f"[{pnl_color}]${result.total_pnl:+.2f}[/{pnl_color}]")
    t.add_row("Avg Win",        f"[green]${result.avg_win:+.2f}[/green]",
              "Avg Loss",       f"[red]${result.avg_loss:+.2f}[/red]")
    t.add_row("Largest Win",    f"[green]${result.largest_win:+.2f}[/green]",
              "Largest Loss",   f"[red]${result.largest_loss:+.2f}[/red]")
    t.add_row("Max Drawdown",   f"[red]${result.max_drawdown:.2f}[/red]",
              "",               "")

    console.print(t)
    console.print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    if result.profit_factor >= 1.5:
        verdict = "[bold green]STRONG — Consider paper trading[/bold green]"
    elif result.profit_factor >= 1.0:
        verdict = "[bold yellow]PROFITABLE — Verify with more data[/bold yellow]"
    elif result.profit_factor >= 0.8:
        verdict = "[bold red]MARGINAL — Needs improvement[/bold red]"
    else:
        verdict = "[bold red]LOSING — Do not use live[/bold red]"

    console.print(f"Verdict: {verdict}")
    console.print()

    if result.total_trades < 30:
        console.print("[yellow]Warning: Less than 30 trades — results may not be statistically reliable.[/yellow]")
        console.print("[yellow]Run longer backtest period before drawing conclusions.[/yellow]")
        console.print()


def save_csv(result: BacktestResult, output_dir: str = "logs/backtest") -> str:
    """Save trade list to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{result.strategy_name}_{timestamp}.csv"

    import csv
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "entry_time", "exit_time", "direction", "entry", "sl", "tp",
            "exit_price", "pnl", "pnl_r", "result", "reason"
        ])
        for t in result.trades:
            writer.writerow([
                t.entry_time, t.exit_time, t.direction,
                t.entry, t.sl, t.tp, t.exit_price,
                round(t.pnl, 2), round(t.pnl_r, 2),
                t.result, t.reason,
            ])
    return filename


def compare_results(results: list[BacktestResult]) -> None:
    """Side-by-side comparison table of multiple strategies."""
    t = Table("Strategy", "Trades", "Win Rate", "Profit Factor", "Total PnL", "Max DD",
              header_style="bold", border_style="dim")

    for r in sorted(results, key=lambda x: x.profit_factor, reverse=True):
        pf_color  = "green" if r.profit_factor >= 1.0 else "red"
        pnl_color = "green" if r.total_pnl >= 0 else "red"
        t.add_row(
            r.strategy_name,
            str(r.total_trades),
            f"{r.win_rate}%",
            f"[{pf_color}]{r.profit_factor}[/{pf_color}]",
            f"[{pnl_color}]${r.total_pnl:+.2f}[/{pnl_color}]",
            f"${r.max_drawdown:.2f}",
        )

    console.print()
    console.print("[bold]Strategy Comparison[/bold]")
    console.print(t)
