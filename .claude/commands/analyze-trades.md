---
description: Analyze recent trades with AI and get parameter improvement recommendations
---

Run the AI trade analyst on the last 7 days of trading data.

Steps:
1. Run `python -m ai_loop.analyst` to generate the analysis
2. Read the output report from `logs/reports/`
3. Present the key findings clearly — what the AI recommends and why
4. Ask the user if they want to apply any parameter changes to `.env`
5. If yes, make the changes to `.env` and confirm what was changed
6. Remind them to restart the bot for changes to take effect
