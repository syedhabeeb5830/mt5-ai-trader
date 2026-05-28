-- mt5-ai-trader — Database Schema
-- Run once to initialize: psql $DATABASE_URL -f db/schema.sql
-- Or it runs automatically on first startup.

CREATE TABLE IF NOT EXISTS trades (
    id          SERIAL PRIMARY KEY,
    ticket      BIGINT,
    symbol      TEXT,
    direction   TEXT,          -- BUY | SELL
    entry       NUMERIC(12,5),
    sl          NUMERIC(12,5),
    tp          NUMERIC(12,5),
    volume      NUMERIC(10,5),
    opened_at   TIMESTAMPTZ,
    closed_at   TIMESTAMPTZ,
    pnl         NUMERIC(12,2),
    spread_at_entry NUMERIC(8,5),
    paper       BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id              SERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ,
    ended_at        TIMESTAMPTZ,
    total_trades    INTEGER,
    winning_trades  INTEGER,
    losing_trades   INTEGER,
    total_pnl       NUMERIC(12,2),
    halted          BOOLEAN DEFAULT FALSE,
    halt_reason     TEXT,
    config_snapshot JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ai_reports (
    id          SERIAL PRIMARY KEY,
    report_date DATE,
    report_text TEXT,
    sessions_analyzed INTEGER,
    trades_analyzed   INTEGER,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS trades_opened_at_idx  ON trades (opened_at DESC);
CREATE INDEX IF NOT EXISTS sessions_started_at_idx ON sessions (started_at DESC);
