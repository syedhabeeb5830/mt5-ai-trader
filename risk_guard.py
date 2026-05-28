"""
Risk Guard — halts trading if daily limits are breached.
Checked before every trade entry.
"""

from __future__ import annotations

import config
from order_manager import OrderManager


class RiskGuard:

    def __init__(self, manager: OrderManager):
        self._mgr = manager
        self.halted = False
        self.halt_reason: str = ""

    def check(self) -> bool:
        """Returns True if trading is allowed."""
        if self.halted:
            return False

        pnl    = self._mgr.realized_pnl
        trades = self._mgr.total_today

        if pnl <= config.DAILY_LOSS_LIMIT:
            self._halt(f"Daily loss limit hit: ${pnl:.2f} ≤ ${config.DAILY_LOSS_LIMIT}")
            return False

        if trades >= config.MAX_DAILY_TRADES:
            self._halt(f"Max daily trades reached: {trades}")
            return False

        return True

    def can_open_position(self) -> bool:
        if not self.check():
            return False
        return self._mgr.open_count < config.MAX_POSITIONS

    def _halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason
