from __future__ import annotations

from typing import Dict

COMMAND_TO_ACTION: Dict[str, int] = {
    "no_action": 0,
    "reset": 1,
    "memory_scrub": 2,
    "load_shedding": 3,
    "power_cycle": 4,
    "thermal_throttle": 5,
    "isolate": 6,
}

ACTION_TO_COMMAND: Dict[int, str] = {
    value: key for key, value in COMMAND_TO_ACTION.items()
}


def discrete_from_command(command: str) -> int:
    """Map a string command to a discrete action id with a safe fallback."""
    if not command:
        return COMMAND_TO_ACTION["no_action"]

    normalized = command.strip().lower()
    return COMMAND_TO_ACTION.get(normalized, COMMAND_TO_ACTION["no_action"])


def command_from_discrete(action_id: int) -> str:
    """Map a discrete action id back to a string command with a safe fallback."""
    return ACTION_TO_COMMAND.get(int(action_id), "no_action")
