"""
TITAN Memory System

Tracks recent step outcomes to provide context for recovery decisions.
"""

from typing import List, Optional


class Memory:
    """
    Maintains a sliding window of recent step summaries.
    """

    def __init__(self, max_size: int = 5) -> None:
        """
        Parameters
        ----------
        max_size : int
            Maximum number of step summaries to retain.
        """
        self.max_size = max_size
        self.entries: List[str] = []

    def add(
        self,
        step: int,
        fault: str,
        action: str,
        reward: float,
        hint: Optional[str] = None,
    ) -> None:
        """
        Add a step summary to memory.

        Parameters
        ----------
        step : int
            Step number.
        fault : str
            Diagnosed fault type.
        action : str
            Action taken.
        reward : float
            Reward received.
        hint : Optional[str]
            Additional hint (e.g., from diagnose action).
        """
        summary = f"Step {step}: fault={fault} → action={action} → reward={reward:.2f}"
        if hint is not None:
            summary += f" (diagnose revealed: {hint})"
        
        self.entries.append(summary)
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size:]

    def get(self) -> List[str]:
        """
        Get the list of step summaries.

        Returns
        -------
        List[str]
            List of summary strings.
        """
        return self.entries.copy()

    def get_formatted(self) -> str:
        """
        Get step summaries as a single formatted string.

        Returns
        -------
        str
            Summaries joined by newlines.
        """
        return "\n".join(self.entries)

    def clear(self) -> None:
        """Clear all entries."""
        self.entries = []
