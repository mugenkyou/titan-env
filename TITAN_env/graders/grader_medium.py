from __future__ import annotations

from typing import Any, Dict, List


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def grade_medium(trajectory: List[Dict[str, Any]], energy_threshold: float = 0.55) -> float:
    """Grade medium task by thermal reduction and energy efficiency."""
    if len(trajectory) < 2:
        return 0.0

    # Support both 'cpu_temperature' (core) and 'cpu_temp' (OpenEnv wrapper) field names
    def get_temp(state: Dict) -> float:
        return float(state.get("cpu_temperature", state.get("cpu_temp", 1.0)))

    start_temp = get_temp(trajectory[0].get("state", {}))
    final_temp = get_temp(trajectory[-1].get("state", {}))
    target_temp = 0.70

    required_delta = max(start_temp - target_temp, 1e-6)
    achieved_delta = max(start_temp - final_temp, 0.0)
    thermal_score = _clip01(achieved_delta / required_delta)

    currents = [
        float(step.get("state", {}).get("current_draw", 1.0))
        for step in trajectory
    ]
    avg_current = sum(currents) / len(currents)
    if avg_current <= energy_threshold:
        energy_score = 1.0
    else:
        energy_score = _clip01(1.0 - (avg_current - energy_threshold) / (1.0 - energy_threshold))

    settle_step = len(trajectory) - 1
    for idx, step in enumerate(trajectory):
        if get_temp(step.get("state", {})) < target_temp:
            settle_step = idx
            break
    max_steps = max(len(trajectory) - 1, 1)
    efficiency_score = _clip01(1.0 - (float(settle_step) / float(max_steps)))

    score = 0.55 * thermal_score + 0.25 * energy_score + 0.20 * efficiency_score
    return _clip01(score)
