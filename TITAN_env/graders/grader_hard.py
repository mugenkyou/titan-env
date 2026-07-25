from __future__ import annotations

from typing import Any, Dict, List


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def grade_hard(trajectory: List[Dict[str, Any]], target_survival_steps: int = 120) -> float:
    """Grade hard task by survival, battery stability, thermal stability, and active recovery.
    
    Scoring breakdown:
    - 40% survival (must survive close to target to score well)
    - 25% battery stability (maintain battery above 0.20)
    - 20% thermal stability (keep average temperature low)
    - 15% recovery actions (reward active fault management)
    """
    if len(trajectory) < 2:
        return 0.0

    # Support both 'cpu_temperature' (core) and 'cpu_temp' (OpenEnv wrapper) field names
    def get_temp(state: Dict) -> float:
        return float(state.get("cpu_temperature", state.get("cpu_temp", 1.0)))
    
    # Support both 'battery_soc' (core) and 'battery' (OpenEnv wrapper) field names
    def get_battery(state: Dict) -> float:
        return float(state.get("battery_soc", state.get("battery", 0.0)))

    survived_steps = len(trajectory) - 1
    # Stricter survival scoring - need to survive most steps
    survival_score = _clip01(float(survived_steps) / float(target_survival_steps))

    battery_values = [
        get_battery(step.get("state", {}))
        for step in trajectory
    ]
    min_battery = min(battery_values)
    battery_score = _clip01((min_battery - 0.20) / 0.80)

    cpu_temps = [
        get_temp(step.get("state", {}))
        for step in trajectory
    ]
    avg_temp = sum(cpu_temps) / len(cpu_temps)
    thermal_stability = _clip01(1.0 - avg_temp)
    
    # Reward active recovery - count non-no_action steps
    active_actions = sum(
        1 for step in trajectory 
        if step.get("action", "no_action") != "no_action"
    )
    action_ratio = _clip01(float(active_actions) / max(float(survived_steps), 1.0))
    # Bonus for taking some actions, but not too many (optimal is 10-30% active)
    recovery_score = min(action_ratio * 3.0, 1.0) if action_ratio > 0 else 0.0

    score = 0.40 * survival_score + 0.25 * battery_score + 0.20 * thermal_stability + 0.15 * recovery_score
    return _clip01(score)
