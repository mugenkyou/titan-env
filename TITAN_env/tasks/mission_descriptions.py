"""
Mission descriptions for TITAN tasks.

This module provides human-readable mission descriptions for each TITAN task.
It maps task names to plain-English descriptions explaining the scenario and goals.

NOTE: This is a standalone file. It does not import or alter any existing task files.
"""

from __future__ import annotations

from typing import Dict

# Mission descriptions mapping task names to human-readable descriptions
MISSION_DESCRIPTIONS: Dict[str, str] = {
    "easy": (
        "Recover onboard memory after radiation-induced corruption before critical data loss. "
        "Goal: memory_integrity > 0.9 within 30 steps."
    ),
    "easy_single_fault_recovery": (
        "Recover onboard memory after radiation-induced corruption before critical data loss. "
        "Goal: memory_integrity > 0.9 within 30 steps."
    ),
    "medium": (
        "Stabilize CPU and power subsystem temperatures following a thermal fault cascade. "
        "Goal: cpu_temp and power_temp within safe thresholds within 40 steps."
    ),
    "medium_thermal_stabilization": (
        "Stabilize CPU and power subsystem temperatures following a thermal fault cascade. "
        "Goal: cpu_temp and power_temp within safe thresholds within 40 steps."
    ),
    "hard": (
        "Survive a compounding multi-fault scenario spanning memory, thermal, and latch-up anomalies. "
        "Goal: keep the system operational for 50 steps without full failure."
    ),
    "hard_multi_fault_survival": (
        "Survive a compounding multi-fault scenario spanning memory, thermal, and latch-up anomalies. "
        "Goal: keep the system operational for 50 steps without full failure."
    ),
}

# Convenience aliases for direct access
EASY_MISSION = MISSION_DESCRIPTIONS["easy_single_fault_recovery"]
MEDIUM_MISSION = MISSION_DESCRIPTIONS["medium_thermal_stabilization"]
HARD_MISSION = MISSION_DESCRIPTIONS["hard_multi_fault_survival"]


def get_mission_description(task_name: str) -> str:
    """
    Get the mission description for a given task name.
    
    Args:
        task_name: The name of the task (e.g., 'easy', 'easy_single_fault_recovery')
        
    Returns:
        Human-readable mission description string.
        
    Raises:
        KeyError: If task_name is not recognized.
    """
    key = task_name.strip().lower()
    if key not in MISSION_DESCRIPTIONS:
        raise KeyError(f"Unknown task: {task_name}")
    return MISSION_DESCRIPTIONS[key]
