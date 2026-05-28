# MT5 AI Trader

An open-source, AI-native algorithmic trading system for MetaTrader 5.

Trade Gold, Forex, and other instruments automatically using a momentum-based strategy. After each session, the AI analyzes your trades and suggests improvements — making the system smarter over time.

**Zero credentials in this repo. You configure everything. MIT licensed.**

---

## What This Does

- Reads live price ticks from your MetaTrader5 terminal every 200ms
- Detects momentum signals (strong directional price moves)
- Places trades automatically with stop loss and take profit
- Protects you with daily loss limits and position caps
- Logs every trade to CSV or PostgreSQL
- After each session, sends your trade data to Claude / GPT-4 / Gemini for analysis
- AI tells you exactly what parameters to adjust and why

---

## What You Need

1. **Windows PC** (required — MetaTrader5 only runs on Windows)
2. **MetaTrader5 terminal** — free download from your broker
3. **A broker account** — demo accounts work fine for testing
4. **Python 3.11+** — free from python.org
5. **One AI subscription** — Claude, ChatGPT, or Gemini (any paid plan works)

That's it. No server. No cloud. Runs entirely on your PC.

---

## Quick Start

### Step 1 — Clone the repo

```bash
git clone https://github.com/yourusername/mt5-ai-trader.git
cd mt5-ai-trader
```

### Step 2 — Run setup

```bash
# Windows
setup.bat

# macOS / Linux
bash setup.sh
```

This installs all dependencies and creates your `.env` file.

### Step 3 — Configure your `.env`

Open the `.env` file in any text editor and fill in:

```env
# Your broker's symbol for Gold (check in MT5 — common names below)
SYMBOL=XAUUSD       # try: XAUUSD, GOLD, GOLD.i, XAUUSD.c

# Your AI provider and API key
AI_PROVIDER=claude
AI_API_KEY=sk-ant-your-key-here

# Start small — 0.01 lot = micro lot = ~$1 risk per trade
VOLUME=0.01
```

**Where to get your AI API key:**
- Claude: [claude.ai/settings](https://claude.ai) → API Keys
- ChatGPT: [platform.openai.com](https://platform.openai.com) → API Keys
- Gemini: [aistudio.google.com](https://aistudio.google.com) → Get API Key

### Step 4 — Start MT5 terminal

1. Open MetaTrader5
2. Log into your broker account (demo or live)
3. Make sure you can see Gold/XAUUSD in the Market Watch panel

### Step 5 — Start the MT5 bridge

Open a terminal window and run:

```bash
python mt5_server.py
```

You should see: `[OK] Connected to MetaTrader5 terminal`

Keep this window open while trading.

### Step 6 — Test the connection

In a second terminal window:

```bash
python scalper.py --status
```

This shows your account balance, current Gold price, and open positions.

### Step 7 — Paper trade first

```bash
python scalper.py --paper
```

Paper mode runs the full strategy but places no real orders. Watch it for a day to see how it behaves.

### Step 8 — Go live

```bash
python scalper.py
```

Press `Ctrl+C` to stop.

---

## How the AI Loop Works

After each trading session, run:

```bash
python -m ai_loop.analyst
```

Or if you use Claude Code:

```bash
/analyze-trades
```

The AI receives your last 7 days of trade data and current settings, then gives you:

- Win rate analysis
- Why trades are winning or losing
- Exact parameter changes to try
- Risk warnings

Example output:

```
## Performance Assessment
Win rate of 23% over 47 trades suggests the MIN_MOVE_POINTS threshold is too low,
causing entries on weak signals during low-volatility periods.

## Parameter Recommendations
- MIN_MOVE_POINTS: Change from 0.5 → 1.0 | Reason: 68% of losses occurred
  when the pre-trade move was under 0.8 points — these are noise, not signals.
- MAX_SPREAD_POINTS: Change from 0.8 → 0.6 | Reason: 12 trades were entered
  with spreads between 0.6-0.8, all resulting in losses.

## Risk Flag
LOW — Daily loss limit is appropriate for current account size.
```

---

## Strategy Explained

The bot uses **momentum scalping** on Gold (XAUUSD):

```
Every 200ms:
  ↓
Look at the last 8 price ticks
  ↓
If 75%+ moved in the same direction AND total move > 0.5 points:
  → Signal detected
  ↓
If spread is tight enough AND daily limits not hit:
  → Place trade
  ↓
Set Stop Loss: entry ± 1.0 point
Set Take Profit: entry ± 5.0 points
  ↓
Wait for SL or TP to hit (auto-managed by broker)
```

**Math:**
| | Value |
|--|--|
| Default symbol | XAUUSD (Gold) |
| Stop loss | 1.0 point |
| Take profit | 5.0 points |
| Risk:Reward | 1:5 |
| Default volume | 0.01 lot (micro) |
| Risk per trade | ~$1 |
| Profit per win | ~$5 |
| Breakeven win rate | 17% (1 win in 6 trades) |

---

## Tuning Guide

The most common adjustments and when to make them:

| Problem | Fix |
|---|---|
| Too many losses, signals feel random | Raise `MIN_MOVE_POINTS` from 0.5 → 1.0 |
| Missing fast moves | Lower `POLL_MS` from 200 → 100 |
| Getting stopped out by spread | Lower `MAX_SPREAD_POINTS` from 0.8 → 0.5 |
| Too many trades per day | Lower `MAX_DAILY_TRADES` from 30 → 15 |
| Want bigger profits per trade | Raise `TP_POINTS` from 5 → 8 (lower win rate) |
| Volatile sessions (NY open, NFP) | Raise `MIN_MOVE_POINTS` to 2.0 |

All changes go in your `.env` file. Restart the bot after changing.

---

## File Structure

```
mt5-ai-trader/
├── .env.example          ← Copy to .env and configure
├── setup.bat             ← Windows one-command setup
├── setup.sh              ← macOS/Linux setup
├── docker-compose.yml    ← Optional: PostgreSQL for trade logging
│
├── mt5_server.py         ← Bridge between Python and MT5 terminal
├── scalper.py            ← Main bot (run this)
├── config.py             ← Loads all settings from .env
├── tick_feed.py          ← Live price feed
├── momentum.py           ← Signal detection
├── order_manager.py      ← Trade placement and management
├── risk_guard.py         ← Daily loss limits
├── dashboard.py          ← Terminal display
│
├── ai_loop/
│   ├── analyst.py        ← AI trade analysis
│   └── prompts.py        ← AI system prompts
│
├── db/
│   ├── logger.py         ← Trade logging (CSV or PostgreSQL)
│   └── schema.sql        ← Database schema
│
└── logs/                 ← Created automatically, trade CSVs go here
```

---

## Optional: PostgreSQL Logging

By default, trades are logged to CSV files in the `logs/` folder. For full analytics and AI self-improvement, you can optionally enable PostgreSQL:

**Option A — Docker (easiest):**
```bash
# Requires Docker Desktop installed
docker-compose up -d
```

Then update `.env`:
```env
DATABASE_URL=postgresql://trader:password@localhost:5432/mt5trader
```

**Option B — Cloud (any PostgreSQL host works):**
```env
DATABASE_URL=postgresql://user:password@your-host:5432/mt5trader
```

---

## Running with Claude Code (AI Loop Mode)

If you use [Claude Code](https://claude.ai/code), this repo includes commands for autonomous AI-assisted trading:

```bash
# Analyze last 7 days of trades
/analyze-trades

# Run bot continuously with AI loop (reviews and adjusts every session)
/loop-trade
```

Claude Code reads your trade logs, analyzes performance, suggests parameter changes, and can apply them automatically with your approval.

---

## Risk Warning

**Trading involves real financial risk. You can lose money.**

- Always start with a demo account
- Always test with `--paper` mode first
- Start with the minimum lot size (0.01)
- Never trade money you cannot afford to lose
- The AI suggestions are for guidance only — verify before applying
- Past performance does not guarantee future results

---

## Common Issues

**"Cannot reach MT5 API"**
→ Make sure `python mt5_server.py` is running in a separate terminal.

**"MT5 terminal is NOT connected"**
→ Open MetaTrader5 and log into your broker account.

**"Symbol 'XAUUSD' not found"**
→ Check the exact symbol name in your MT5 Market Watch. Update `SYMBOL=` in `.env`.

**"AI analysis failed"**
→ Check your `AI_API_KEY` in `.env`. Make sure you have API access (not just a chat subscription).

**Spread too wide — no trades executing**
→ Try during active market hours (London/NY overlap: 13:00–17:00 UTC).

---

## Contributing

Pull requests welcome. Please:
- Keep it simple — this repo is for beginners
- No credentials or personal config in commits
- Test paper mode before submitting trading logic changes

---

## License

MIT — use freely, modify freely, share freely.
