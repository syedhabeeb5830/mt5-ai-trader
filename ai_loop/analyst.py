"""
AI Loop — Trade Analyst
─────────────────────────────────────────────────────────────────────────────
Reads session trade logs and uses AI (Claude / GPT-4 / Gemini) to analyze
performance and suggest parameter improvements.

Run after a trading session:
  python -m ai_loop.analyst

Or use the Claude Code command: /analyze-trades
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import config
from ai_loop.prompts import SYSTEM_PROMPT, ANALYSIS_PROMPT
from db.logger import TradeLogger


class TradeAnalyst:
    """Analyzes trade session data using the configured AI provider."""

    def __init__(self):
        self.provider = config.AI_PROVIDER
        self.api_key  = config.AI_API_KEY
        self.model    = config.AI_MODEL

    def analyze(self, session_stats: dict, trades: list, current_config: dict) -> str:
        prompt = ANALYSIS_PROMPT.format(
            session_json=json.dumps(session_stats, indent=2),
            trades_json=json.dumps(trades[-20:], indent=2),  # last 20 trades
            config_json=json.dumps(current_config, indent=2),
        )

        if self.provider == "claude":
            return self._claude(prompt)
        elif self.provider == "openai":
            return self._openai(prompt)
        elif self.provider == "gemini":
            return self._gemini(prompt)
        else:
            raise ValueError(f"Unknown AI provider: {self.provider}. Set AI_PROVIDER=claude|openai|gemini in .env")

    def _claude(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model or "claude-sonnet-4-6",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model or "gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
        )
        return response.choices[0].message.content

    def _gemini(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model or "gemini-1.5-pro",
            system_instruction=SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt)
        return response.text


def _build_current_config() -> dict:
    return {
        "SYMBOL":           config.SYMBOL,
        "SL_POINTS":        config.SL_POINTS,
        "TP_POINTS":        config.TP_POINTS,
        "VOLUME":           config.VOLUME,
        "MOMENTUM_WINDOW":  config.MOMENTUM_WINDOW,
        "MIN_DIRECTION_PCT":config.MIN_DIRECTION_PCT,
        "MIN_MOVE_POINTS":  config.MIN_MOVE_POINTS,
        "MAX_SPREAD_POINTS":config.MAX_SPREAD_POINTS,
        "POLL_MS":          config.POLL_MS,
        "DAILY_LOSS_LIMIT": config.DAILY_LOSS_LIMIT,
        "MAX_DAILY_TRADES": config.MAX_DAILY_TRADES,
    }


async def run_analysis():
    logger  = TradeLogger()
    await logger.init()

    print("Loading trade data...")
    sessions = await logger.get_recent_sessions(days=7)
    trades   = await logger.get_recent_trades(days=7)

    if not trades and not sessions:
        print("No trade data found. Run scalper.py first to collect data.")
        return

    # Aggregate session stats
    total_trades = len(trades)
    wins  = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) < 0]
    total_pnl = sum(t.get("pnl", 0) for t in trades)

    session_stats = {
        "period_days":   7,
        "total_sessions": len(sessions),
        "total_trades":  total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct":  round(len(wins) / total_trades * 100, 1) if total_trades else 0,
        "total_pnl":     round(total_pnl, 2),
        "avg_win":       round(sum(t.get("pnl", 0) for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss":      round(sum(t.get("pnl", 0) for t in losses) / len(losses), 2) if losses else 0,
        "sessions":      sessions[-5:],  # last 5 sessions
    }

    print(f"\nAnalyzing {total_trades} trades over 7 days...")
    print(f"Win rate: {session_stats['win_rate_pct']}%  |  Total PnL: ${total_pnl:+.2f}\n")

    analyst = TradeAnalyst()

    print("Sending to AI for analysis...")
    try:
        analysis = analyst.analyze(session_stats, trades, _build_current_config())
    except Exception as e:
        print(f"AI analysis failed: {e}")
        return

    # Save report
    reports_dir = Path("logs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"analysis_{timestamp}.md"
    report_path.write_text(analysis, encoding="utf-8")

    print("\n" + "="*60)
    print(analysis)
    print("="*60)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_analysis())
