"""
mt5-ai-trader — Local MT5 REST Server
─────────────────────────────────────────────────────────────────────────────
Runs a lightweight FastAPI server that wraps the MetaTrader5 Python library.
This is the bridge between the trading bot and your MT5 terminal.

REQUIREMENTS:
  - Windows OS (MetaTrader5 Python lib is Windows-only)
  - MetaTrader5 terminal installed and logged in
  - pip install fastapi uvicorn MetaTrader5

RUN:
  python mt5_server.py

The server starts on http://localhost:8000
Set MT5_API_URL=http://localhost:8000 in your .env file.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

import os
from dotenv import load_dotenv
load_dotenv()

# ── MT5 import (Windows only) ─────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARN] MetaTrader5 package not found. Running in mock mode.")

app = FastAPI(title="MT5 Local REST Server", version="1.0.0")

MT5_KEY = os.getenv("MT5_API_KEY", "")


def _auth(x_api_key: Optional[str] = None):
    """Optional API key check. Skip if MT5_API_KEY is not set."""
    if MT5_KEY and x_api_key != MT5_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _require_mt5():
    if not MT5_AVAILABLE:
        raise HTTPException(status_code=503, detail="MetaTrader5 library not available (Windows only)")
    if not mt5.initialize():
        raise HTTPException(status_code=503, detail="MT5 terminal not running or not logged in")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    if MT5_AVAILABLE:
        if mt5.initialize():
            print("[OK] Connected to MetaTrader5 terminal")
            info = mt5.terminal_info()
            if info:
                print(f"[OK] Terminal: {info.name} build {info.build}")
        else:
            print("[WARN] MT5 terminal not connected. Start MT5 and log in.")


@app.on_event("shutdown")
async def shutdown():
    if MT5_AVAILABLE:
        mt5.shutdown()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health(x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)

    if not MT5_AVAILABLE:
        return {"status": "mock", "mt5_connected": False, "symbols_active": 0}

    connected = mt5.initialize()
    symbols_active = 0
    if connected:
        syms = mt5.symbols_get()
        symbols_active = len(syms) if syms else 0

    return {
        "status": "ok" if connected else "disconnected",
        "mt5_connected": connected,
        "symbols_active": symbols_active,
    }


# ── Account ───────────────────────────────────────────────────────────────────

@app.get("/account")
async def get_account(x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)
    _require_mt5()

    info = mt5.account_info()
    if info is None:
        raise HTTPException(status_code=404, detail="No account info")
    return {
        "login":     info.login,
        "balance":   info.balance,
        "equity":    info.equity,
        "margin":    info.margin,
        "free_margin": info.margin_free,
        "currency":  info.currency,
        "leverage":  info.leverage,
        "server":    info.server,
    }


# ── Tick ─────────────────────────────────────────────────────────────────────

@app.get("/symbol/{symbol}/tick")
async def get_tick(symbol: str, x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)
    _require_mt5()

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        # Try enabling the symbol
        mt5.symbol_select(symbol, True)
        time.sleep(0.1)
        tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found or no tick data")

    return {
        "time_msc": tick.time_msc,
        "bid":      tick.bid,
        "ask":      tick.ask,
        "last":     tick.last,
        "volume":   tick.volume,
    }


# ── Positions ─────────────────────────────────────────────────────────────────

@app.get("/positions")
async def get_positions(x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)
    _require_mt5()

    positions = mt5.positions_get()
    if positions is None:
        return []
    return [
        {
            "ticket":  p.ticket,
            "symbol":  p.symbol,
            "type":    "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume":  p.volume,
            "price_open": p.price_open,
            "sl":      p.sl,
            "tp":      p.tp,
            "profit":  p.profit,
            "magic":   p.magic,
            "comment": p.comment,
        }
        for p in positions
    ]


@app.get("/position/{ticket}")
async def get_position(ticket: int, x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)
    _require_mt5()

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        raise HTTPException(status_code=404, detail="Position not found")
    p = positions[0]
    return {
        "ticket":  p.ticket,
        "symbol":  p.symbol,
        "type":    "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
        "volume":  p.volume,
        "price_open": p.price_open,
        "sl":      p.sl,
        "tp":      p.tp,
        "profit":  p.profit,
        "magic":   p.magic,
    }


@app.get("/deals/{ticket}")
async def get_deal_pnl(ticket: int, x_api_key: Optional[str] = Header(default=None)):
    """Return the total realised PnL for a closed position by summing its deal history."""
    _auth(x_api_key)
    _require_mt5()

    import datetime as _dt
    date_from = _dt.datetime.now() - _dt.timedelta(days=30)
    date_to   = _dt.datetime.now() + _dt.timedelta(days=1)

    deals = mt5.history_deals_get(date_from, date_to, position=ticket)
    if not deals:
        return {"profit": 0.0, "found": False}

    profit = sum(d.profit for d in deals)
    return {"profit": round(profit, 2), "found": True}


# ── Orders ────────────────────────────────────────────────────────────────────

class OrderRequest(BaseModel):
    symbol:     str
    order_type: str      # "BUY" or "SELL"
    volume:     float
    sl:         float
    tp:         float
    deviation:  int = 3
    magic:      int = 0
    comment:    str = ""


@app.post("/order")
async def place_order(req: OrderRequest, x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)
    _require_mt5()

    tick = mt5.symbol_info_tick(req.symbol)
    if tick is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{req.symbol}' not found")

    order_type = mt5.ORDER_TYPE_BUY if req.order_type == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick.ask if req.order_type == "BUY" else tick.bid

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    req.symbol,
        "volume":    req.volume,
        "type":      order_type,
        "price":     price,
        "sl":        req.sl,
        "tp":        req.tp,
        "deviation": req.deviation,
        "magic":     req.magic,
        "comment":   req.comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        return {"success": False, "retcode": -1, "retcode_description": "order_send returned None"}

    return {
        "success":              result.retcode == mt5.TRADE_RETCODE_DONE,
        "retcode":              result.retcode,
        "retcode_description":  result.comment,
        "order":                result.order,
        "price":                result.price,
        "volume":               result.volume,
    }


@app.delete("/position/{ticket}")
async def close_position(ticket: int, x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)
    _require_mt5()

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        raise HTTPException(status_code=404, detail="Position not found")

    p = positions[0]
    close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(p.symbol)
    if tick is None:
        raise HTTPException(status_code=404, detail="Cannot get tick for close")

    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

    request = {
        "action":   mt5.TRADE_ACTION_DEAL,
        "symbol":   p.symbol,
        "volume":   p.volume,
        "type":     close_type,
        "position": ticket,
        "price":    price,
        "deviation": 3,
        "magic":    p.magic,
        "comment":  "close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        return {"success": False, "retcode_description": "order_send returned None"}

    return {
        "success": result.retcode == mt5.TRADE_RETCODE_DONE,
        "retcode": result.retcode,
        "retcode_description": result.comment,
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MT5_SERVER_PORT", "8000"))
    print(f"Starting MT5 REST server on http://localhost:{port}")
    print("Make sure MetaTrader5 terminal is open and logged in.")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
