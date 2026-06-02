"""
Backtest Runner — CLI
─────────────────────────────────────────────────────────────────────────────
Usage:
  python backtest/run.py --strategy ema7_tbm_v2 --days 365
  python backtest/run.py --strategy ema7_tbm_v3 --days 180
  python backtest/run.py --strategy sqrt_levels_v4 --days 365
  python backtest/run.py --all --days 365        # compare all strategies
  python backtest/run.py --strategy ema7_tbm_15m --days 365  # study only

Requirements:
  - MetaTrader5 terminal running and logged in (Windows)
  - OR CSV data files in backtest/data/<SYMBOL>_<TF>.csv

CSV Format (if using files instead of MT5):
  Columns: time, open, high, low, close, volume
  time format: ISO datetime or Unix timestamp
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from backtest.engine import BacktestEngine, fetch_bars, load_csv
from backtest.report import print_report, save_csv, compare_results
from strategies import REGISTRY, get_strategy
from strategies.ema7_tbm_15m import EMA7TBM15M   # study-only, not in registry
from analytics.engine import analyze, from_backtest_result, save_report


# Timeframes each strategy needs
STRATEGY_TIMEFRAMES = {
    "ema7_tbm_v2":      ["4H", "1H"],
    "ema7_tbm_v3":      ["D1", "4H", "1H"],
    "sqrt_levels_v4":   ["D1", "1H"],
    "ema7_tbm_15m":     ["1H", "15M"],   # study only
    "momentum_scalper": ["5M"],
}
STRATEGY_ENTRY_TF = {
    "ema7_tbm_v2":      "1H",
    "ema7_tbm_v3":      "1H",
    "sqrt_levels_v4":   "1H",
    "ema7_tbm_15m":     "15M",
    "momentum_scalper": "5M",
}
ALL_STRATEGIES = {**REGISTRY, "ema7_tbm_15m": EMA7TBM15M}

# Approximate bars needed per timeframe for N days
BARS_PER_DAY = {"1M": 1440, "5M": 288, "15M": 96, "1H": 24, "4H": 6, "D1": 1}


def load_data(symbol: str, timeframes: list[str], days: int) -> dict:
    """Load historical data from MT5 or CSV files."""
    bars = {}
    data_dir = Path("backtest/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    for tf in timeframes:
        n_bars = BARS_PER_DAY.get(tf, 24) * (days + 10)  # extra warmup

        # Try MT5 first
        df = fetch_bars(symbol, tf, n_bars)
        if df is not None:
            bars[tf] = df
            print(f"  [{tf}] Loaded {len(df)} bars from MT5")
            continue

        # Try CSV fallback
        csv_path = data_dir / f"{symbol}_{tf}.csv"
        df = load_csv(str(csv_path))
        if df is not None:
            bars[tf] = df.tail(n_bars)
            print(f"  [{tf}] Loaded {len(bars[tf])} bars from CSV")
            continue

        print(f"  [{tf}] No data found — MT5 not available and no CSV at {csv_path}")
        print(f"        Export from MT5 → File → Open Data Folder → MQL5/Files")

    return bars


def run_strategy(name: str, days: int, symbol: str, volume: float, save: bool) -> None:
    print(f"\nBacktesting: {name} | {symbol} | {days} days\n")

    if name == "ema7_tbm_15m":
        print("Note: This is a STUDY strategy — it loses money. Running for educational purposes.\n")

    timeframes = STRATEGY_TIMEFRAMES.get(name, ["1H"])
    entry_tf   = STRATEGY_ENTRY_TF.get(name, "1H")

    bars = load_data(symbol, timeframes, days)
    if not bars or entry_tf not in bars:
        print(f"Could not load data for {entry_tf}. Start MT5 terminal or add CSV files.")
        return

    # Filter to requested date range
    for tf in bars:
        bars[tf] = bars[tf].tail(BARS_PER_DAY.get(tf, 24) * days)

    strategy = ALL_STRATEGIES[name]()
    engine   = BacktestEngine(symbol=symbol, volume=volume)
    result   = engine.run(strategy, bars, entry_tf=entry_tf)

    print_report(result)

    # Phase 2: run analytics on backtest result
    analytics_records = from_backtest_result(result)
    analytics_report  = analyze(analytics_records, label=name)
    print(analytics_report.to_text())

    if save:
        filepath = save_csv(result)
        print(f"Trade list saved: {filepath}")
        jp, tp = save_report(analytics_report)
        print(f"Analytics JSON : {jp}")
        print(f"Analytics text : {tp}")


def run_strategy_oos(
    name: str, days: int, symbol: str, volume: float, save: bool,
    split: float = 0.7,
) -> None:
    """Run strategy on in-sample AND out-of-sample splits and compare."""
    from analytics.strategy_selector import oos_split

    print(f"\nOut-of-Sample Validation: {name} | {symbol} | {days} days "
          f"(IS={int(split*100)}% / OOS={int((1-split)*100)}%)\n")

    timeframes = STRATEGY_TIMEFRAMES.get(name, ["1H"])
    entry_tf   = STRATEGY_ENTRY_TF.get(name, "1H")
    bars       = load_data(symbol, timeframes, days)

    if not bars or entry_tf not in bars:
        print(f"No data for {entry_tf}.")
        return

    for tf in bars:
        bars[tf] = bars[tf].tail(BARS_PER_DAY.get(tf, 24) * days)

    in_bars, oos_bars = oos_split(bars, split)

    def _run(label: str, b: dict) -> None:
        if entry_tf not in b or len(b[entry_tf]) < 10:
            print(f"  [{label}] Insufficient bars — skipped")
            return
        strategy = ALL_STRATEGIES[name]()
        engine   = BacktestEngine(symbol=symbol, volume=volume)
        result   = engine.run(strategy, b, entry_tf=entry_tf)
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
        print_report(result)
        records = from_backtest_result(result)
        report  = analyze(records, label=f"{name}_{label.lower().replace(' ','_')}")
        print(f"  Profit Factor : {report.profit_factor:.3f}")
        print(f"  Win Rate      : {report.win_rate:.1f}%")
        print(f"  Expectancy    : ${report.expectancy:+.3f}/trade")
        print(f"  Net PnL       : ${report.net_pnl:+.2f}")
        if save:
            save_csv(result)

    _run("IN-SAMPLE   (training)",  in_bars)
    _run("OUT-OF-SAMPLE (forward)", oos_bars)

    print("\n  OOS Gate: PF should be within 80% of IS PF to rule out curve-fitting.\n")


def main():
    parser = argparse.ArgumentParser(description="MT5 AI Trader — Backtest Runner")
    parser.add_argument("--strategy", "-s",
                        choices=list(ALL_STRATEGIES.keys()),
                        help="Strategy to backtest")
    parser.add_argument("--all", action="store_true",
                        help="Backtest all strategies and compare")
    parser.add_argument("--recommend", action="store_true",
                        help="Run all strategies and show ranked leaderboard + recommendation")
    parser.add_argument("--oos", action="store_true",
                        help="Out-of-sample split: run IS (70%%) then OOS (30%%) and compare")
    parser.add_argument("--metric", "-m", default="composite",
                        choices=["profit_factor", "win_rate", "expectancy", "sharpe", "composite"],
                        help="Ranking metric for --recommend (default: composite)")
    parser.add_argument("--days", "-d", type=int, default=365,
                        help="Number of days to backtest (default: 365)")
    parser.add_argument("--symbol", default=None,
                        help="Symbol to test (default: from .env)")
    parser.add_argument("--volume", type=float, default=0.01,
                        help="Lot size (default: 0.01)")
    parser.add_argument("--save", action="store_true",
                        help="Save trade list to CSV")

    args = parser.parse_args()
    symbol = args.symbol or config.SYMBOL

    if not args.strategy and not args.all and not args.recommend:
        parser.print_help()
        print("\nAvailable strategies:")
        for name, cls in ALL_STRATEGIES.items():
            inst = cls()
            print(f"  {name:20s} — {inst.description}")
        sys.exit(0)

    if args.recommend:
        from analytics.strategy_selector import rank_strategies, print_leaderboard, save_recommendation
        print(f"\nComparing all strategies on {symbol} | last {args.days} days …\n")
        ranks = rank_strategies(
            days=args.days, symbol=symbol, volume=args.volume,
            metric=args.metric, verbose=True,
        )
        print_leaderboard(ranks, metric=args.metric)
        if args.save:
            path = save_recommendation(ranks, args.metric, args.days)
            print(f"  Recommendation saved → {path}\n")
        return

    if args.oos:
        if not args.strategy:
            print("--oos requires --strategy NAME")
            sys.exit(1)
        run_strategy_oos(args.strategy, args.days, symbol, args.volume, args.save)
        return

    if args.all:
        results = []
        for name in REGISTRY.keys():  # only proven strategies
            timeframes = STRATEGY_TIMEFRAMES.get(name, ["1H"])
            entry_tf   = STRATEGY_ENTRY_TF.get(name, "1H")
            bars = load_data(symbol, timeframes, args.days)
            if not bars or entry_tf not in bars:
                continue
            for tf in bars:
                bars[tf] = bars[tf].tail(BARS_PER_DAY.get(tf, 24) * args.days)
            strategy = REGISTRY[name]()
            engine   = BacktestEngine(symbol=symbol, volume=args.volume)
            result   = engine.run(strategy, bars, entry_tf=entry_tf)
            print_report(result)
            results.append(result)
            if args.save:
                save_csv(result)
        if len(results) > 1:
            compare_results(results)
    else:
        run_strategy(args.strategy, args.days, symbol, args.volume, args.save)


if __name__ == "__main__":
    main()
