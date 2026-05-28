#!/usr/bin/env bash
# mt5-ai-trader — Setup Script (macOS / Linux)
# Note: MT5 server (mt5_server.py) only works on Windows.
# On macOS/Linux you can still use this repo with a remote MT5 API or in paper mode.

set -e

echo ""
echo "========================================"
echo " MT5 AI Trader — Setup"
echo "========================================"
echo ""

# Step 1: Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Install from https://python.org/downloads"
    exit 1
fi
echo "[OK] Python found: $(python3 --version)"

# Step 2: Install dependencies
echo ""
echo "Installing Python packages..."
pip3 install httpx rich python-dotenv fastapi uvicorn anthropic openai google-generativeai asyncpg

# Step 3: Create .env
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env from template..."
    cp .env.example .env
    echo "[OK] .env created. Edit it with your settings."
else
    echo "[OK] .env already exists."
fi

# Step 4: Logs directory
mkdir -p logs
echo "[OK] Logs directory ready."

echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Edit .env — set AI_PROVIDER, AI_API_KEY, SYMBOL"
echo "2. On Windows: open MT5, then run: python mt5_server.py"
echo "3. Test: python scalper.py --status"
echo "4. Paper test: python scalper.py --paper"
echo "5. Live: python scalper.py"
echo ""
