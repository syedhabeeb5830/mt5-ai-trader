---
description: Run the bot autonomously — trade a session, analyze results, suggest improvements, repeat
---

Autonomous AI trading loop. Each cycle:

1. Check MT5 server is running: `python scalper.py --status`
   - If not reachable, tell user to start `python mt5_server.py` and wait for confirmation
2. Ask user: "Run in paper mode or live mode?" (default: paper)
3. Start the bot: `python scalper.py --paper` (or without --paper for live)
4. Wait for the user to stop the bot with Ctrl+C
5. Run analysis: `python -m ai_loop.analyst`
6. Show the AI's recommendations clearly
7. Ask: "Would you like me to apply these changes and run another session?"
8. If yes: update `.env` with approved changes, then go back to step 3
9. If no: summarize what was learned this session and stop

Safety rules:
- Never apply parameter changes without explicit user approval
- Always show what will change in `.env` before applying
- Never change DAILY_LOSS_LIMIT to a less restrictive value without extra confirmation
- Remind user of risk warning before first live session
