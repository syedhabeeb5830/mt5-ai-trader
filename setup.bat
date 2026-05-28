@echo off
REM mt5-ai-trader — One-Command Setup (Windows)
REM Run this once after cloning the repo.

echo.
echo ========================================
echo  MT5 AI Trader — Setup
echo ========================================
echo.

REM Step 1: Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download Python 3.11+ from https://python.org/downloads
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found.

REM Step 2: Install dependencies
echo.
echo Installing Python packages...
pip install httpx rich python-dotenv fastapi uvicorn MetaTrader5 anthropic openai google-generativeai asyncpg
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Package installation failed. Check your internet connection.
    pause
    exit /b 1
)
echo [OK] Packages installed.

REM Step 3: Create .env if not exists
if not exist .env (
    echo.
    echo Creating .env from template...
    copy .env.example .env
    echo [OK] .env created. Edit it with your settings before running.
) else (
    echo [OK] .env already exists.
)

REM Step 4: Create logs directory
if not exist logs mkdir logs
echo [OK] Logs directory ready.

echo.
echo ========================================
echo  Setup complete!
echo ========================================
echo.
echo NEXT STEPS:
echo.
echo 1. Edit .env with your settings:
echo    - Set AI_PROVIDER and AI_API_KEY
echo    - Set SYMBOL to your broker's Gold symbol (e.g. XAUUSD or GOLD.i)
echo    - Adjust SL_POINTS, TP_POINTS, VOLUME for your account size
echo.
echo 2. Open MetaTrader5 and log into your broker account.
echo.
echo 3. Start the MT5 REST server (new terminal window):
echo    python mt5_server.py
echo.
echo 4. Test the connection:
echo    python scalper.py --status
echo.
echo 5. Run in paper mode first (no real money):
echo    python scalper.py --paper
echo.
echo 6. Go live when ready:
echo    python scalper.py
echo.
pause
