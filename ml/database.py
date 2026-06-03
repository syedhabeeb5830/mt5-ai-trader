"""
ML Platform — SQLite Database Layer
─────────────────────────────────────────────────────────────────────────────
Single SQLite file (ml/data/market.db) with 5 tables:

  candles      — raw OHLCV bars (all instruments, all timeframes)
  features     — engineered feature vectors keyed to a candle
  labels       — TP/SL outcome labels keyed to a candle + profile
  predictions  — live model outputs per candle (for retraining & monitoring)
  ml_trades    — resolved trade outcomes from paper/live sessions

All access goes through Database.  Import-safe: no I/O at module level.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator

from ml.instrument_config import DATA_DB


# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA = """
-- Raw OHLCV bars
CREATE TABLE IF NOT EXISTS candles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    ts          INTEGER NOT NULL,          -- Unix epoch seconds (UTC bar open time)
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL,
    spread      REAL    DEFAULT 0.0,
    UNIQUE (symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_candles_sym_tf_ts ON candles (symbol, timeframe, ts DESC);

-- Engineered feature vectors (one row per candle + label_profile)
CREATE TABLE IF NOT EXISTS features (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    candle_id       INTEGER NOT NULL REFERENCES candles(id),
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    ts              INTEGER NOT NULL,
    label_profile   TEXT    NOT NULL,      -- e.g. "momentum", "intraday"
    feature_json    TEXT    NOT NULL,      -- JSON dict of all features
    created_at      INTEGER DEFAULT (strftime('%s','now')),
    UNIQUE (candle_id, label_profile)
);
CREATE INDEX IF NOT EXISTS idx_features_sym_ts ON features (symbol, ts DESC);

-- Supervised labels (TP/SL forward scan outcomes)
CREATE TABLE IF NOT EXISTS labels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    candle_id       INTEGER NOT NULL REFERENCES candles(id),
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    ts              INTEGER NOT NULL,
    label_profile   TEXT    NOT NULL,
    direction       TEXT    NOT NULL,      -- "BUY" | "SELL"
    label           INTEGER NOT NULL,      -- 1 = TP first, 0 = SL first / timeout
    bars_to_exit    INTEGER,               -- how many bars until outcome
    exit_price      REAL,
    created_at      INTEGER DEFAULT (strftime('%s','now')),
    UNIQUE (candle_id, label_profile, direction)
);
CREATE INDEX IF NOT EXISTS idx_labels_sym_ts ON labels (symbol, ts DESC);

-- Live / paper prediction log (feeds retraining & monitoring)
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    ts              INTEGER NOT NULL,      -- bar open time of the signal candle
    label_profile   TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,      -- e.g. "XAUUSD_v3"
    direction       TEXT    NOT NULL,      -- "BUY" | "SELL"
    probability     REAL    NOT NULL,      -- P(TP before SL)
    threshold_used  REAL    NOT NULL,
    top_features    TEXT,                  -- JSON: [{"name":..,"value":..,"shap":..}, ...]
    acted           INTEGER DEFAULT 0,     -- 1 if an actual order was placed
    outcome_label   INTEGER,               -- 1/0 filled in after trade resolves
    outcome_pnl     REAL,
    created_at      INTEGER DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_pred_sym_ts ON predictions (symbol, ts DESC);

-- Resolved trade outcomes (paper + live)
CREATE TABLE IF NOT EXISTS ml_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket          INTEGER,
    symbol          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    entry_price     REAL    NOT NULL,
    sl              REAL    NOT NULL,
    tp              REAL    NOT NULL,
    volume          REAL    NOT NULL,
    probability     REAL,
    model_version   TEXT,
    opened_at       INTEGER NOT NULL,
    closed_at       INTEGER,
    pnl             REAL,
    exit_reason     TEXT,                  -- "TP" | "SL" | "TIMEOUT" | "MANUAL"
    paper           INTEGER DEFAULT 1,
    created_at      INTEGER DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_mltrades_sym ON ml_trades (symbol, opened_at DESC);
"""


# ── Connection manager ─────────────────────────────────────────────────────────

class Database:
    """
    Thread-safe SQLite wrapper.  Instantiate once and share the instance.
    All writes are auto-committed.  Read methods return list[dict].
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path or DATA_DB)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Internal ──────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ── Candles ───────────────────────────────────────────────────────────────

    def upsert_candles(self, rows: list[dict]) -> int:
        """
        Insert or ignore OHLCV rows.
        rows: list of dicts with keys: symbol, timeframe, ts, open, high, low, close, volume, spread
        Returns number of rows inserted.
        """
        sql = """
            INSERT OR IGNORE INTO candles (symbol, timeframe, ts, open, high, low, close, volume, spread)
            VALUES (:symbol, :timeframe, :ts, :open, :high, :low, :close, :volume, :spread)
        """
        with self._conn() as conn:
            before = conn.total_changes
            conn.executemany(sql, rows)
            return conn.total_changes - before

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 5000,
        since_ts: int | None = None,
    ) -> list[dict]:
        sql = """
            SELECT * FROM candles
            WHERE symbol=? AND timeframe=?
            {since}
            ORDER BY ts ASC
            LIMIT ?
        """.format(since="AND ts > ?" if since_ts else "")
        params = (symbol, timeframe, since_ts, limit) if since_ts else (symbol, timeframe, limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def latest_candle_ts(self, symbol: str, timeframe: str) -> int | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(ts) as ts FROM candles WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()
        return row["ts"] if row and row["ts"] else None

    # ── Features ──────────────────────────────────────────────────────────────

    def upsert_features(self, rows: list[dict]) -> int:
        """rows: dicts with candle_id, symbol, timeframe, ts, label_profile, feature_json (str)."""
        sql = """
            INSERT OR REPLACE INTO features
              (candle_id, symbol, timeframe, ts, label_profile, feature_json)
            VALUES
              (:candle_id, :symbol, :timeframe, :ts, :label_profile, :feature_json)
        """
        with self._conn() as conn:
            before = conn.total_changes
            conn.executemany(sql, rows)
            return conn.total_changes - before

    def get_features(
        self,
        symbol: str,
        label_profile: str,
        limit: int = 50000,
        since_ts: int | None = None,
    ) -> list[dict]:
        sql = """
            SELECT f.*, c.open, c.high, c.low, c.close, c.volume, c.spread
            FROM features f
            JOIN candles c ON c.id = f.candle_id
            WHERE f.symbol=? AND f.label_profile=?
            {since}
            ORDER BY f.ts ASC
            LIMIT ?
        """.format(since="AND f.ts > ?" if since_ts else "")
        params = (symbol, label_profile, since_ts, limit) if since_ts else (symbol, label_profile, limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Labels ────────────────────────────────────────────────────────────────

    def upsert_labels(self, rows: list[dict]) -> int:
        """rows: dicts with candle_id, symbol, timeframe, ts, label_profile, direction, label,
                 bars_to_exit, exit_price."""
        sql = """
            INSERT OR REPLACE INTO labels
              (candle_id, symbol, timeframe, ts, label_profile, direction,
               label, bars_to_exit, exit_price)
            VALUES
              (:candle_id, :symbol, :timeframe, :ts, :label_profile, :direction,
               :label, :bars_to_exit, :exit_price)
        """
        with self._conn() as conn:
            before = conn.total_changes
            conn.executemany(sql, rows)
            return conn.total_changes - before

    def get_labels(
        self,
        symbol: str,
        label_profile: str,
        direction: str = "BUY",
        limit: int = 50000,
    ) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM labels WHERE symbol=? AND label_profile=? AND direction=? "
                "ORDER BY ts ASC LIMIT ?",
                (symbol, label_profile, direction, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Predictions ───────────────────────────────────────────────────────────

    def insert_prediction(self, pred: dict) -> int:
        sql = """
            INSERT INTO predictions
              (symbol, timeframe, ts, label_profile, model_version, direction,
               probability, threshold_used, top_features, acted)
            VALUES
              (:symbol, :timeframe, :ts, :label_profile, :model_version, :direction,
               :probability, :threshold_used, :top_features, :acted)
        """
        with self._conn() as conn:
            conn.execute(sql, pred)
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def resolve_prediction(self, pred_id: int, outcome_label: int, outcome_pnl: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE predictions SET outcome_label=?, outcome_pnl=? WHERE id=?",
                (outcome_label, outcome_pnl, pred_id),
            )

    def get_recent_predictions(self, symbol: str, limit: int = 200) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE symbol=? ORDER BY created_at DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── ML Trades ─────────────────────────────────────────────────────────────

    def insert_ml_trade(self, trade: dict) -> int:
        sql = """
            INSERT INTO ml_trades
              (ticket, symbol, direction, entry_price, sl, tp, volume,
               probability, model_version, opened_at, paper)
            VALUES
              (:ticket, :symbol, :direction, :entry_price, :sl, :tp, :volume,
               :probability, :model_version, :opened_at, :paper)
        """
        with self._conn() as conn:
            conn.execute(sql, trade)
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def resolve_ml_trade(
        self, trade_id: int, closed_at: int, pnl: float, exit_reason: str
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE ml_trades SET closed_at=?, pnl=?, exit_reason=? WHERE id=?",
                (closed_at, pnl, exit_reason, trade_id),
            )

    # ── Stats helpers ─────────────────────────────────────────────────────────

    def candle_count(self, symbol: str, timeframe: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM candles WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()[0]

    def feature_count(self, symbol: str, label_profile: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM features WHERE symbol=? AND label_profile=?",
                (symbol, label_profile),
            ).fetchone()[0]

    def label_count(self, symbol: str, label_profile: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM labels WHERE symbol=? AND label_profile=?",
                (symbol, label_profile),
            ).fetchone()[0]

    def summary(self) -> dict:
        """Human-readable DB summary dict."""
        with self._conn() as conn:
            candles  = conn.execute("SELECT symbol, timeframe, COUNT(*) as n FROM candles GROUP BY symbol, timeframe").fetchall()
            features = conn.execute("SELECT symbol, label_profile, COUNT(*) as n FROM features GROUP BY symbol, label_profile").fetchall()
            labels   = conn.execute("SELECT symbol, label_profile, direction, COUNT(*) as n, AVG(label) as wr FROM labels GROUP BY symbol, label_profile, direction").fetchall()
            preds    = conn.execute("SELECT symbol, COUNT(*) as n FROM predictions GROUP BY symbol").fetchall()
        return {
            "candles":     [dict(r) for r in candles],
            "features":    [dict(r) for r in features],
            "labels":      [dict(r) for r in labels],
            "predictions": [dict(r) for r in preds],
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_db: Database | None = None

def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
