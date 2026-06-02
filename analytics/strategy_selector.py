"""
Strategy Selector — Phase 3
─────────────────────────────────────────────────────────────────────────────
Batch-runs ALL registered strategies on historical data, ranks them by a
configurable metric, and recommends the best one.

Ranking metrics:
  profit_factor  — gross_profit / gross_loss  (default)
  win_rate       — % winning trades
  expectancy     — $ expected value per trade
  sharpe         — Sharpe ratio
  composite      — weighted blend of all four

Composite score (0–1 normalised):
  0.35 × profit_factor  +  0.30 × win_rate
  + 0.20 × expectancy   +  0.15 × sharpe

Usage:
  python -m analytics.strategy_selector                         # default 60d
  python -m analytics.strategy_selector --days 90 --metric win_rate
  python -m analytics.strategy_selector --metric composite --save
  python -m analytics.strategy_selector --auto                  # print winner name only
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ── Lazy heavy imports (kept at function level to keep module fast to import) ─

def _import_backtest():
    from backtest.engine import BacktestEngine, load_csv
    from backtest.run   import load_data, STRATEGY_TIMEFRAMES, STRATEGY_ENTRY_TF, BARS_PER_DAY
    return BacktestEngine, load_csv, load_data, STRATEGY_TIMEFRAMES, STRATEGY_ENTRY_TF, BARS_PER_DAY

def _import_analytics():
    from analytics.engine import analyze, from_backtest_result
    return analyze, from_backtest_result


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class StrategyRank:
    name:          str
    description:   str
    n_trades:      int
    win_rate:      float          # 0–100
    profit_factor: float
    expectancy:    float          # $ per trade
    net_pnl:       float
    max_drawdown:  float
    sharpe:        Optional[float]
    composite:     float          # 0–1 weighted score
    rank:          int = 0        # set after sorting
    recommended:   bool = False   # set on winner

    # ── Display helpers ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sharpe"] = round(self.sharpe, 3) if self.sharpe is not None else None
        return d

    @property
    def verdict_tag(self) -> str:
        if self.n_trades < 20:
            return "THIN DATA"
        if self.profit_factor >= 1.3 and self.win_rate >= 45:
            return "STRONG"
        if self.profit_factor >= 1.1 and self.n_trades >= 50:
            return "GOOD"
        if self.profit_factor >= 1.0 and self.n_trades >= 20:
            return "MARGINAL"
        return "LOSING"


# ── Composite scorer ──────────────────────────────────────────────────────────

def _composite(pf: float, wr: float, exp: float, sharpe: Optional[float]) -> float:
    """
    Normalise each metric to a 0-1 range then weight them.

    Normalisation caps:
      profit_factor : 0 → 0,  2.0 → 1.0  (capped at 2.0)
      win_rate      : 0 → 0,  100 → 1.0
      expectancy    : ≤0 → 0, 1.0+ → 1.0  (capped at $1.0/trade)
      sharpe        : 0 → 0,  3.0+ → 1.0  (capped at 3.0)
    """
    n_pf  = min(max(pf - 1.0, 0.0) / 1.0, 1.0)   # excess above breakeven, cap 1.0
    n_wr  = min(wr / 100.0, 1.0)
    n_exp = min(max(exp, 0.0) / 1.0, 1.0)
    n_sh  = min(max(sharpe or 0.0, 0.0) / 3.0, 1.0)

    return round(0.35 * n_pf + 0.30 * n_wr + 0.20 * n_exp + 0.15 * n_sh, 4)


# ── Scoring metric extractor ──────────────────────────────────────────────────

_METRIC_KEY = {
    "profit_factor": lambda r: r.profit_factor,
    "win_rate":      lambda r: r.win_rate,
    "expectancy":    lambda r: r.expectancy,
    "sharpe":        lambda r: (r.sharpe or 0.0),
    "composite":     lambda r: r.composite,
}

VALID_METRICS = list(_METRIC_KEY.keys())


# ── Core comparison function ──────────────────────────────────────────────────

def rank_strategies(
    days:    int   = 60,
    symbol:  str   = "XAUUSD",
    volume:  float = 0.01,
    metric:  str   = "composite",
    verbose: bool  = True,
) -> List[StrategyRank]:
    """
    Run all registered strategies on `days` of historical data.
    Returns list of StrategyRank objects sorted by `metric` descending.
    Only strategies with data available are included.
    """
    from strategies import REGISTRY

    BacktestEngine, load_csv, load_data, STRAT_TFS, STRAT_ETF, BARS_PER_DAY = _import_backtest()
    analyze, from_backtest_result = _import_analytics()

    if metric not in _METRIC_KEY:
        raise ValueError(f"metric must be one of {VALID_METRICS}, got {metric!r}")

    ranks: List[StrategyRank] = []

    for name, cls in REGISTRY.items():
        inst = cls()
        tfs  = STRAT_TFS.get(name, ["1H"])
        etf  = STRAT_ETF.get(name, "1H")

        if verbose:
            print(f"  Running {name} …", end="", flush=True)

        bars = load_data(symbol, tfs, days)
        if not bars or etf not in bars:
            if verbose:
                print(" NO DATA — skipped")
            continue

        for tf in bars:
            bars[tf] = bars[tf].tail(BARS_PER_DAY.get(tf, 24) * days)

        try:
            engine = BacktestEngine(symbol=symbol, volume=volume)
            result = engine.run(inst, bars, entry_tf=etf)
        except Exception as e:
            if verbose:
                print(f" ERROR: {e}")
            continue

        records = from_backtest_result(result)
        report  = analyze(records, label=name)

        comp = _composite(
            report.profit_factor,
            report.win_rate,
            report.expectancy,
            report.sharpe_ratio,
        )

        rank = StrategyRank(
            name          = name,
            description   = inst.description[:60],
            n_trades      = report.total_trades,
            win_rate      = report.win_rate,
            profit_factor = report.profit_factor,
            expectancy    = report.expectancy,
            net_pnl       = report.net_pnl,
            max_drawdown  = report.max_drawdown,
            sharpe        = report.sharpe_ratio,
            composite     = comp,
        )
        ranks.append(rank)

        if verbose:
            print(f" done  ({report.total_trades} trades, PF {report.profit_factor:.3f}, "
                  f"WR {report.win_rate:.1f}%)")

    # Sort by chosen metric descending
    key_fn = _METRIC_KEY[metric]
    ranks.sort(key=key_fn, reverse=True)

    # Assign ranks and flag winner
    for i, r in enumerate(ranks):
        r.rank = i + 1
        r.recommended = (i == 0 and r.profit_factor >= 1.0 and r.n_trades >= 20)

    return ranks


def recommend(
    days:   int   = 60,
    symbol: str   = "XAUUSD",
    volume: float = 0.01,
    metric: str   = "composite",
) -> Optional[StrategyRank]:
    """
    Returns the top-ranked StrategyRank, or None if no data was found.
    Prints nothing — caller decides what to display.
    """
    ranks = rank_strategies(days=days, symbol=symbol, volume=volume,
                             metric=metric, verbose=False)
    if not ranks:
        return None
    winner = ranks[0]
    return winner if winner.recommended else None


# ── Pretty leaderboard ────────────────────────────────────────────────────────

def print_leaderboard(ranks: List[StrategyRank], metric: str = "composite") -> None:
    bar = "═" * 90
    thin = "─" * 90

    print(f"\n{bar}")
    print(f"  STRATEGY LEADERBOARD  ·  Ranked by: {metric.upper()}")
    print(bar)
    print(f"  {'#':>2}  {'Strategy':<22}  {'Trades':>6}  {'WinRate':>7}  "
          f"{'ProfFactor':>10}  {'Expect':>7}  {'Sharpe':>6}  {'Score':>6}  {'Tag':<10}")
    print(thin)

    for r in ranks:
        marker  = " ★" if r.recommended else "  "
        sharpe  = f"{r.sharpe:.2f}" if r.sharpe is not None else "  N/A"
        print(f"  {r.rank:>2}{marker} {r.name:<22}  {r.n_trades:>6}  "
              f"{r.win_rate:>6.1f}%  {r.profit_factor:>10.3f}  "
              f"{r.expectancy:>+7.3f}  {sharpe:>6}  {r.composite:>6.3f}  {r.verdict_tag:<10}")

    print(thin)
    winner = next((r for r in ranks if r.recommended), None)
    if winner:
        print(f"\n  ★  RECOMMENDATION: {winner.name.upper()}")
        print(f"     PF={winner.profit_factor:.3f}  WR={winner.win_rate:.1f}%  "
              f"Expectancy=${winner.expectancy:+.3f}/trade  "
              f"Composite={winner.composite:.3f}")
        print(f"     {winner.description}")
    else:
        print("\n  ⚠  No strategy passed the minimum gate (PF ≥ 1.0, N ≥ 20 trades).")
        print("     Expand data range (--days) or fetch more history.")
    print(f"{bar}\n")


# ── Persistence ───────────────────────────────────────────────────────────────

def save_recommendation(ranks: List[StrategyRank], metric: str, days: int) -> str:
    """Save leaderboard JSON to logs/recommendation.json (overwrite)."""
    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "recommendation.json"

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "ranking_metric": metric,
        "lookback_days": days,
        "recommended": ranks[0].name if ranks and ranks[0].recommended else None,
        "leaderboard": [r.to_dict() for r in ranks],
    }
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


def load_recommendation() -> Optional[dict]:
    """Load last saved recommendation from logs/recommendation.json."""
    path = Path("logs/recommendation.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ── Out-of-sample split helper ────────────────────────────────────────────────

def oos_split(bars: dict, split: float = 0.7) -> tuple:
    """
    Split a bars dict into (in_sample, out_of_sample) at `split` fraction.
    Returns two dicts with the same keys.
    """
    in_s:  dict = {}
    out_s: dict = {}
    for tf, df in bars.items():
        n      = len(df)
        cutoff = int(n * split)
        in_s[tf]  = df.iloc[:cutoff].copy()
        out_s[tf] = df.iloc[cutoff:].copy()
    return in_s, out_s


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import config

    parser = argparse.ArgumentParser(
        description="Strategy Selector — rank all strategies and recommend the best one."
    )
    parser.add_argument("--days",   "-d", type=int,   default=60,
                        help="Lookback days for backtest comparison (default: 60)")
    parser.add_argument("--metric", "-m", default="composite",
                        choices=VALID_METRICS,
                        help="Ranking metric (default: composite)")
    parser.add_argument("--symbol", default=None,
                        help="Symbol (default: from .env)")
    parser.add_argument("--volume", type=float, default=0.01,
                        help="Lot size (default: 0.01)")
    parser.add_argument("--save",   action="store_true",
                        help="Save recommendation to logs/recommendation.json")
    parser.add_argument("--auto",   action="store_true",
                        help="Print only the recommended strategy name (for scripting)")

    args   = parser.parse_args()
    symbol = args.symbol or config.SYMBOL

    if not args.auto:
        print(f"\nComparing all strategies on {symbol} | last {args.days} days …\n")

    ranks = rank_strategies(
        days    = args.days,
        symbol  = symbol,
        volume  = args.volume,
        metric  = args.metric,
        verbose = not args.auto,
    )

    if args.auto:
        winner = next((r for r in ranks if r.recommended), None)
        print(winner.name if winner else "none")
        sys.exit(0)

    print_leaderboard(ranks, metric=args.metric)

    if args.save:
        path = save_recommendation(ranks, args.metric, args.days)
        print(f"  Recommendation saved → {path}\n")


if __name__ == "__main__":
    main()
