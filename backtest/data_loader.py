"""
Data Loader — downloads historical OHLCV data via yfinance.
─────────────────────────────────────────────────────────────────────────────
Used when MT5 is not available (office laptop, CI, etc.).
Data source: GC=F (COMEX Gold Futures) via Yahoo Finance.

Note on GC=F vs XAUUSD:
  GC=F is Gold Futures. Spot price is ~$5–15 lower (cost of carry / basis).
  Directional correlation with XAUUSD is >99.9%.
  Strategy signal testing is valid. PnL absolute values are comparable.

yfinance limitations:
  1M  data : max  7 days
  5M  data : max 60 days
  1H  data : max 730 days
  D1  data : unlimited

Usage:
  python backtest/data_loader.py                    # downloads 5M (60d) + 1H (365d)
  python backtest/data_loader.py --tf 5M            # 5-minute only
  python backtest/data_loader.py --tf 1H --days 180
  python backtest/data_loader.py --list             # show what's saved
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_TICKER  = "GC=F"
DEFAULT_OUT_DIR = "backtest/data"

# yfinance interval string → (max_days, file_suffix)
INTERVAL_CONFIG = {
    "1M":  ("1m",  7,    "XAUUSD_1M.csv"),
    "5M":  ("5m",  60,   "XAUUSD_5M.csv"),
    "1H":  ("1h",  730,  "XAUUSD_1H.csv"),
    "4H":  ("1h",  730,  "XAUUSD_4H.csv"),   # downloaded as 1H then resampled
    "D1":  ("1d",  3650, "XAUUSD_D1.csv"),
}


def _require_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        print("[ERROR] yfinance not installed.")
        print("        Run: pip install yfinance pandas numpy")
        sys.exit(1)


def download(
    timeframe:  str = "5M",
    days:       int = 0,
    ticker:     str = DEFAULT_TICKER,
    output_dir: str = DEFAULT_OUT_DIR,
    silent:     bool = False,
) -> Optional[Path]:
    """
    Download OHLCV data from Yahoo Finance and save as CSV.

    Parameters
    ----------
    timeframe  : "1M" | "5M" | "1H" | "D1"
    days       : how many days to request (0 = use yfinance maximum)
    ticker     : Yahoo Finance ticker symbol (default GC=F = Gold Futures)
    output_dir : directory to save CSV files
    silent     : suppress console output (for testing)

    Returns the saved file path, or None on failure.
    """
    yf = _require_yfinance()

    if timeframe not in INTERVAL_CONFIG:
        print(f"[ERROR] Unknown timeframe '{timeframe}'. Choose from: {list(INTERVAL_CONFIG)}")
        return None

    yf_interval, max_days, filename = INTERVAL_CONFIG[timeframe]
    resample_to_4h = (timeframe == "4H")
    actual_days = min(days, max_days) if days > 0 else max_days

    if not silent:
        print(f"Downloading {ticker} [{timeframe}]  interval={yf_interval}  "
              f"period={actual_days}d ...")

    try:
        import pandas as pd
        t = yf.Ticker(ticker)
        df = t.history(
            period=f"{actual_days}d",
            interval=yf_interval,
            auto_adjust=True,
            actions=False,
        )
    except Exception as e:
        if not silent:
            print(f"  [FAIL] Download error: {e}")
        return None

    if df is None or df.empty:
        if not silent:
            print(f"  [FAIL] No data returned. Check ticker '{ticker}' and your internet connection.")
        return None

    # ── Standardise ───────────────────────────────────────────────────────────
    import pandas as pd
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        if not silent:
            print(f"  [FAIL] Missing columns: {missing}. Got: {list(df.columns)}")
        return None

    df = df[["open", "high", "low", "close", "volume"]].dropna()

    # ── Resample 1H → 4H if needed ────────────────────────────────────────────
    if resample_to_4h:
        df = df.resample("4h", closed="left", label="left").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / filename

    df.to_csv(save_path)

    if not silent:
        first = str(df.index[0].date())
        last  = str(df.index[-1].date())
        print(f"  Saved {len(df):,} bars  ({first} → {last})  →  {save_path}")

    return save_path


def list_saved(output_dir: str = DEFAULT_OUT_DIR) -> None:
    """Print a summary of all cached data files."""
    out = Path(output_dir)
    if not out.exists():
        print(f"No data directory at '{output_dir}'. Run the downloader first.")
        return

    files = sorted(out.glob("*.csv"))
    if not files:
        print(f"No CSV files in '{output_dir}'.")
        return

    try:
        import pandas as pd
        print(f"\nCached data files in '{output_dir}':\n")
        for f in files:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            first = str(df.index[0].date()) if len(df) > 0 else "?"
            last  = str(df.index[-1].date()) if len(df) > 0 else "?"
            print(f"  {f.name:30s}  {len(df):7,} bars  {first} → {last}")
        print()
    except Exception as e:
        for f in files:
            print(f"  {f}")
        print(f"  (Could not read details: {e})")


def download_all(output_dir: str = DEFAULT_OUT_DIR) -> None:
    """Download full dataset for all strategies: 5M (60d) + 1H + 4H (730d) + D1 (10yr)."""
    print("Downloading full dataset for all strategies...\n")
    download("5M",  0,   output_dir=output_dir)   # momentum_scalper
    download("1H",  730, output_dir=output_dir)   # ema7_tbm_v2 / v3
    download("4H",  730, output_dir=output_dir)   # ema7_tbm_v2 / v3 (resampled 4H)
    download("D1",  0,   output_dir=output_dir)   # ema7_tbm_v3 / sqrt_levels_v4
    print("\nDone. Run leaderboard with:")
    print("  python -m backtest.run --recommend --days 60")
    print("  python -m backtest.run --recommend --days 180")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Download Gold OHLCV data for offline backtesting"
    )
    parser.add_argument("--tf",  choices=list(INTERVAL_CONFIG), default=None,
                        help="Timeframe to download (default: 5M + 1H)")
    parser.add_argument("--days", type=int, default=0,
                        help="Days of data to request (0=maximum available)")
    parser.add_argument("--ticker", default=DEFAULT_TICKER,
                        help=f"Yahoo Finance ticker (default: {DEFAULT_TICKER})")
    parser.add_argument("--out",  default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--list", action="store_true",
                        help="List cached data files and exit")
    args = parser.parse_args()

    if args.list:
        list_saved(args.out)
        return

    if args.tf:
        download(args.tf, args.days, args.ticker, args.out)
    else:
        download_all(args.out)


if __name__ == "__main__":
    _cli()
