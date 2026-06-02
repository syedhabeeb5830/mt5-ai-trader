"""
Analytics Engine — Phase 2
─────────────────────────────────────────────────────────────────────────────
Pure statistical analysis of trade logs.
No AI, no MT5, no network required.

Computes:
  Win Rate, Avg Win/Loss, Expectancy, Profit Factor,
  Largest/Median Win/Loss, Max Drawdown, Sharpe, Sortino,
  Consecutive Streaks, Trade Distribution by Hour,
  Avg Holding Time, Daily PnL, Monthly PnL

Input:  logs/trades.csv + logs/closed_trades.csv  (merged on ticket)
        OR a BacktestResult object
Output: logs/analytics/report_YYYYMMDD_HHMMSS.{json,txt}

Usage:
  python -m analytics.engine                 # analyse all logs
  python -m analytics.engine --days 30       # last 30 days
  python -m analytics.engine --csv path.csv  # specific closed_trades CSV
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    pnl:        float
    direction:  str   = ""
    entry:      float = 0.0
    sl:         float = 0.0
    tp:         float = 0.0
    opened_at:  Optional[datetime] = None
    closed_at:  Optional[datetime] = None
    result:     str   = ""   # WIN | LOSS | TIMEOUT | ""

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def is_loss(self) -> bool:
        return self.pnl < 0

    @property
    def holding_seconds(self) -> Optional[float]:
        if self.opened_at and self.closed_at:
            return (self.closed_at - self.opened_at).total_seconds()
        return None

    @property
    def open_hour(self) -> Optional[int]:
        return self.opened_at.hour if self.opened_at else None


@dataclass
class AnalyticsReport:
    """Complete analytics snapshot. Serialisable to JSON and plain text."""
    generated_at:        str
    label:               str
    date_from:           str
    date_to:             str
    total_trades:        int
    winning_trades:      int
    losing_trades:       int
    win_rate:            float           # %
    avg_win:             float
    avg_loss:            float
    largest_win:         float
    largest_loss:        float
    median_win:          float
    median_loss:         float
    gross_profit:        float
    gross_loss:          float
    net_pnl:             float
    profit_factor:       float
    expectancy:          float           # $ per trade
    max_drawdown:        float           # $ peak-to-trough
    avg_holding_sec:     Optional[float]
    max_consecutive_wins:  int
    max_consecutive_losses: int
    sharpe_ratio:        Optional[float]
    sortino_ratio:       Optional[float]
    pnl_by_hour:         Dict[int, float]    = field(default_factory=dict)
    trades_by_hour:      Dict[int, int]      = field(default_factory=dict)
    daily_pnl:           Dict[str, float]    = field(default_factory=dict)
    monthly_pnl:         Dict[str, float]    = field(default_factory=dict)
    verdict:             str = ""
    warning:             str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert int keys to str for JSON compatibility
        d["pnl_by_hour"]    = {str(k): v for k, v in self.pnl_by_hour.items()}
        d["trades_by_hour"] = {str(k): v for k, v in self.trades_by_hour.items()}
        return d

    def to_text(self) -> str:
        bar  = "=" * 62
        line = "-" * 62

        def fmt_float(v: Optional[float], prefix: str = "$") -> str:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "N/A (insufficient data)"
            return f"{prefix}{v:+.2f}" if prefix == "$" else f"{v:.3f}"

        def pct(v: float) -> str:
            return f"{v:.1f}%"

        lines = [
            bar,
            f"  {self.label.upper()} — PERFORMANCE REPORT",
            f"  Period : {self.date_from} → {self.date_to}",
            f"  Generated : {self.generated_at} UTC",
            bar,
            "",
            "OVERVIEW",
            line,
            f"  Total Trades         : {self.total_trades}",
            f"  Win Rate             : {pct(self.win_rate)}",
            f"  Winning Trades       : {self.winning_trades}",
            f"  Losing Trades        : {self.losing_trades}",
            "",
            "PNL SUMMARY",
            line,
            f"  Net PnL              : {fmt_float(self.net_pnl)}",
            f"  Gross Profit         : {fmt_float(self.gross_profit)}",
            f"  Gross Loss           : {fmt_float(-self.gross_loss)}",
            f"  Avg Win              : {fmt_float(self.avg_win)}",
            f"  Avg Loss             : {fmt_float(self.avg_loss)}",
            f"  Largest Win          : {fmt_float(self.largest_win)}",
            f"  Largest Loss         : {fmt_float(self.largest_loss)}",
            f"  Median Win           : {fmt_float(self.median_win)}",
            f"  Median Loss          : {fmt_float(self.median_loss)}",
            "",
            "RISK METRICS",
            line,
            f"  Profit Factor        : {self.profit_factor:.3f}",
            f"  Expectancy           : {fmt_float(self.expectancy)}/trade",
            f"  Max Drawdown         : ${self.max_drawdown:.2f}",
            f"  Sharpe Ratio         : {fmt_float(self.sharpe_ratio, prefix='')}",
            f"  Sortino Ratio        : {fmt_float(self.sortino_ratio, prefix='')}",
        ]

        if self.avg_holding_sec is not None:
            mins = self.avg_holding_sec / 60
            lines.append(f"  Avg Holding Time     : {mins:.1f} min")

        lines += [
            "",
            "STREAKS",
            line,
            f"  Max Consecutive Wins   : {self.max_consecutive_wins}",
            f"  Max Consecutive Losses : {self.max_consecutive_losses}",
        ]

        if self.daily_pnl:
            lines += [
                "",
                "MONTHLY PnL",
                line,
            ]
            for month, pnl in sorted(self.monthly_pnl.items()):
                sign = "+" if pnl >= 0 else ""
                lines.append(f"  {month}  :  {sign}${pnl:.2f}")

        if self.pnl_by_hour:
            lines += [
                "",
                "PnL BY HOUR (UTC)",
                line,
            ]
            for h in sorted(self.pnl_by_hour):
                n   = self.trades_by_hour.get(h, 0)
                pnl = self.pnl_by_hour[h]
                bar_len = min(int(abs(pnl) / 0.5), 30)
                bar_str = ("+" if pnl >= 0 else "-") * bar_len
                lines.append(f"  {h:02d}:00  [{n:3d} trades]  ${pnl:+7.2f}  {bar_str}")

        lines += [
            "",
            "VERDICT",
            line,
            f"  {self.verdict}",
        ]
        if self.warning:
            lines.append(f"  WARNING: {self.warning}")

        lines.append(bar)
        return "\n".join(lines)


# ── Core computations (pure functions, no I/O) ────────────────────────────────

def _median(values: list) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return float(s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2)


def _max_drawdown(pnl_series: list) -> float:
    equity = peak = max_dd = 0.0
    for pnl in pnl_series:
        equity += pnl
        peak    = max(peak, equity)
        max_dd  = max(max_dd, peak - equity)
    return round(max_dd, 2)


def _consecutive_streaks(is_win_list: list) -> tuple:
    max_w = max_l = cur_w = cur_l = 0
    for w in is_win_list:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _sharpe(daily_pnls: list) -> Optional[float]:
    if len(daily_pnls) < 10:
        return None
    n    = len(daily_pnls)
    mean = sum(daily_pnls) / n
    var  = sum((p - mean) ** 2 for p in daily_pnls) / n
    std  = var ** 0.5
    if std == 0:
        return None
    return round((mean / std) * (252 ** 0.5), 3)


def _sortino(daily_pnls: list) -> Optional[float]:
    if len(daily_pnls) < 10:
        return None
    mean     = sum(daily_pnls) / len(daily_pnls)
    downside = [p for p in daily_pnls if p < 0]
    if len(downside) < 3:
        return None
    down_var = sum(p ** 2 for p in downside) / len(downside)
    down_std = down_var ** 0.5
    if down_std == 0:
        return None
    return round((mean / down_std) * (252 ** 0.5), 3)


def _verdict(profit_factor: float, total_trades: int, expectancy: float) -> tuple:
    """Returns (verdict_str, warning_str)."""
    warning = ""
    if total_trades < 30:
        warning = f"Only {total_trades} trades — results not statistically reliable. Need ≥30."
    if profit_factor >= 1.5 and total_trades >= 30 and expectancy > 0:
        return "STRONG — Positive expectancy with adequate sample size.", warning
    elif profit_factor >= 1.0 and expectancy > 0:
        return "PROFITABLE — Edge shown but verify with more trades/out-of-sample.", warning
    elif profit_factor >= 0.8:
        return "MARGINAL — Below break-even. Do not trade real money.", warning
    else:
        return "LOSING — Edge not proven. Do not trade real money.", warning


# ── Main analysis function ────────────────────────────────────────────────────

def analyze(trades: List[TradeRecord], label: str = "Strategy") -> AnalyticsReport:
    """
    Core analytics computation. Pure function — no I/O.
    Takes a list of TradeRecord objects, returns AnalyticsReport.
    """
    if not trades:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return AnalyticsReport(
            generated_at=now, label=label,
            date_from="N/A", date_to="N/A",
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            largest_win=0.0, largest_loss=0.0,
            median_win=0.0, median_loss=0.0,
            gross_profit=0.0, gross_loss=0.0, net_pnl=0.0,
            profit_factor=0.0, expectancy=0.0, max_drawdown=0.0,
            avg_holding_sec=None,
            max_consecutive_wins=0, max_consecutive_losses=0,
            sharpe_ratio=None, sortino_ratio=None,
            verdict="NO DATA — No closed trades found.",
        )

    wins   = [t for t in trades if t.is_win]
    losses = [t for t in trades if t.is_loss]
    n      = len(trades)

    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in losses))
    net_pnl      = round(gross_profit - gross_loss, 2)
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0

    win_rate  = round(len(wins) / n * 100, 2) if n > 0 else 0.0
    avg_win   = round(gross_profit / len(wins), 2)   if wins   else 0.0
    avg_loss  = round(-gross_loss  / len(losses), 2) if losses else 0.0   # negative
    expectancy = round((win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss), 4)

    win_pnls  = [t.pnl for t in wins]
    loss_pnls = [t.pnl for t in losses]
    pnl_seq   = [t.pnl for t in trades]

    max_dd = _max_drawdown(pnl_seq)
    max_w, max_l = _consecutive_streaks([t.is_win for t in trades])

    # Dates
    dated = [t for t in trades if t.opened_at]
    date_from = str(min(t.opened_at for t in dated).date()) if dated else "N/A"
    date_to   = str(max(t.opened_at for t in dated).date()) if dated else "N/A"

    # Daily PnL grouping
    daily: Dict[str, float] = defaultdict(float)
    for t in dated:
        key = str(t.opened_at.date())
        daily[key] += t.pnl
    daily_pnls = list(daily.values())

    # Monthly PnL
    monthly: Dict[str, float] = defaultdict(float)
    for day_str, pnl in daily.items():
        month_key = day_str[:7]   # "YYYY-MM"
        monthly[month_key] += pnl
    monthly = {k: round(v, 2) for k, v in monthly.items()}

    # PnL by hour
    pnl_by_hour:    Dict[int, float] = defaultdict(float)
    trades_by_hour: Dict[int, int]   = defaultdict(int)
    for t in dated:
        h = t.opened_at.hour
        pnl_by_hour[h]    += t.pnl
        trades_by_hour[h] += 1
    pnl_by_hour = {k: round(v, 2) for k, v in pnl_by_hour.items()}

    # Holding time
    holding_times = [t.holding_seconds for t in trades if t.holding_seconds is not None]
    avg_holding   = round(sum(holding_times) / len(holding_times), 1) if holding_times else None

    sharpe  = _sharpe(daily_pnls)
    sortino = _sortino(daily_pnls)
    verdict, warning = _verdict(profit_factor, n, expectancy)

    return AnalyticsReport(
        generated_at  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        label         = label,
        date_from     = date_from,
        date_to       = date_to,
        total_trades  = n,
        winning_trades  = len(wins),
        losing_trades   = len(losses),
        win_rate        = win_rate,
        avg_win         = avg_win,
        avg_loss        = avg_loss,
        largest_win     = round(max(win_pnls,  default=0.0), 2),
        largest_loss    = round(min(loss_pnls, default=0.0), 2),
        median_win      = round(_median(win_pnls), 2),
        median_loss     = round(_median(loss_pnls), 2),
        gross_profit    = round(gross_profit, 2),
        gross_loss      = round(gross_loss, 2),
        net_pnl         = net_pnl,
        profit_factor   = profit_factor,
        expectancy      = expectancy,
        max_drawdown    = max_dd,
        avg_holding_sec = avg_holding,
        max_consecutive_wins   = max_w,
        max_consecutive_losses = max_l,
        sharpe_ratio    = sharpe,
        sortino_ratio   = sortino,
        pnl_by_hour     = dict(pnl_by_hour),
        trades_by_hour  = dict(trades_by_hour),
        daily_pnl       = dict(sorted(daily.items())),
        monthly_pnl     = monthly,
        verdict         = verdict,
        warning         = warning,
    )


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def load_trades_from_logs(logs_dir: str = "logs", days: int = 0) -> List[TradeRecord]:
    """
    Load and merge logs/trades.csv + logs/closed_trades.csv.
    Falls back to closed_trades.csv only (minimal metrics).
    days=0 means all records.
    """
    logs = Path(logs_dir)
    trades_file = logs / "trades.csv"
    closed_file = logs / "closed_trades.csv"

    open_map: dict = {}
    if trades_file.exists():
        with open(trades_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ticket = row.get("ticket", "")
                open_map[ticket] = row

    records: List[TradeRecord] = []

    if closed_file.exists():
        with open(closed_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ticket    = row.get("ticket", "")
                pnl_str   = row.get("pnl", "0")
                closed_at = _parse_dt(row.get("closed_at", ""))
                try:
                    pnl = float(pnl_str)
                except (ValueError, TypeError):
                    continue

                meta = open_map.get(ticket, {})
                opened_at = _parse_dt(meta.get("opened_at", ""))

                cutoff = None
                if days > 0:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                if cutoff and opened_at and opened_at < cutoff:
                    continue

                records.append(TradeRecord(
                    pnl       = pnl,
                    direction = meta.get("direction", ""),
                    entry     = float(meta.get("entry", 0) or 0),
                    sl        = float(meta.get("sl", 0) or 0),
                    tp        = float(meta.get("tp", 0) or 0),
                    opened_at = opened_at,
                    closed_at = closed_at,
                ))

    return records


def from_backtest_result(result: Any) -> List[TradeRecord]:
    """Convert a BacktestResult object's trade list to TradeRecord list."""
    records = []
    for t in result.trades:
        opened = t.entry_time if hasattr(t, "entry_time") else None
        closed = t.exit_time  if hasattr(t, "exit_time")  else None
        # Ensure timezone-aware
        if isinstance(opened, datetime) and opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        if isinstance(closed, datetime) and closed.tzinfo is None:
            closed = closed.replace(tzinfo=timezone.utc)
        records.append(TradeRecord(
            pnl       = t.pnl,
            direction = t.direction,
            entry     = t.entry,
            sl        = t.sl,
            tp        = t.tp,
            opened_at = opened,
            closed_at = closed,
            result    = t.result,
        ))
    return records


def save_report(report: AnalyticsReport, output_dir: str = "logs/analytics") -> tuple:
    """Save JSON and text reports. Returns (json_path, txt_path)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{report.label.lower().replace(' ', '_')}_{ts}"

    json_path = out / f"{base_name}.json"
    txt_path  = out / f"{base_name}.txt"

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    txt_path.write_text(report.to_text(), encoding="utf-8")

    return str(json_path), str(txt_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    import argparse
    parser = argparse.ArgumentParser(
        description="Analytics Engine — compute trade performance statistics"
    )
    parser.add_argument("--logs",  default="logs",  help="Logs directory (default: logs)")
    parser.add_argument("--days",  type=int, default=0, help="Analyse last N days (0=all)")
    parser.add_argument("--csv",   default=None,    help="Path to a specific closed_trades.csv")
    parser.add_argument("--label", default="Strategy", help="Strategy label for report")
    parser.add_argument("--save",  action="store_true", help="Save report to logs/analytics/")
    args = parser.parse_args()

    if args.csv:
        # Single CSV: pnl column only, minimal metrics
        records = []
        with open(args.csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    records.append(TradeRecord(
                        pnl=float(row.get("pnl", 0)),
                        closed_at=_parse_dt(row.get("closed_at", "")),
                        opened_at=_parse_dt(row.get("opened_at", "") or row.get("closed_at", "")),
                    ))
                except (ValueError, TypeError):
                    continue
    else:
        records = load_trades_from_logs(args.logs, days=args.days)

    if not records:
        print(f"No closed trade records found in '{args.logs}'.")
        print("Run the scalper in paper mode first: python scalper.py --paper")
        return

    report = analyze(records, label=args.label)
    print(report.to_text())

    if args.save:
        jp, tp = save_report(report)
        print(f"\nSaved: {jp}")
        print(f"Saved: {tp}")


if __name__ == "__main__":
    _cli()
