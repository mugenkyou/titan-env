from __future__ import annotations

from typing import Dict, Tuple, Union

from titan_env.core.environment.actions import ActionType, ACTION_COSTS


W1_UPTIME: float = 1.5
W2_FAULT_SEVERITY: float = 1.0
W3_ENERGY_USAGE: float = 0.1
W4_RECOVERY_LATENCY: float = 2.0  # -2.0 per instructions
W5_ACTION_COST: float = 0.05

FAILURE_PENALTY: float = -30.0

_MAX_FAULT_FLAGS: int = 5
_NOMINAL_CPU_TEMP: float = 0.30
_NOMINAL_PWR_TEMP: float = 0.25
_NOMINAL_BATTERY_SOC: float = 0.90
_MAX_LATENCY: float = (1.0 - _NOMINAL_CPU_TEMP) + (1.0 - _NOMINAL_PWR_TEMP) + _NOMINAL_BATTERY_SOC
_MAX_ACTION_COST: float = max(ACTION_COSTS.values()) if ACTION_COSTS else 1.0


def _is_fault_active(obs: dict, key: str) -> bool:
    return float(obs.get(key, 0.0)) > 0.0


def _compute_uptime(obs: dict) -> float:
    flags = (
        obs.get("seu_flag", 0.0)
        + obs.get("latchup_flag", 0.0)
        + obs.get("thermal_fault_flag", 0.0)
        + obs.get("memory_fault_flag", 0.0)
        + obs.get("power_fault_flag", 0.0)
    )
    return 1.0 if flags == 0.0 else 0.0


def _compute_fault_severity(obs: dict) -> float:
    flag_sum = (
        obs.get("seu_flag", 0.0)
        + obs.get("latchup_flag", 0.0)
        + obs.get("thermal_fault_flag", 0.0)
        + obs.get("memory_fault_flag", 0.0)
        + obs.get("power_fault_flag", 0.0)
    )
    flag_score = flag_sum / _MAX_FAULT_FLAGS

    cpu_excess = max(0.0, obs.get("cpu_temperature", 0.0) - _NOMINAL_CPU_TEMP)
    pwr_excess = max(0.0, obs.get("power_temperature", 0.0) - _NOMINAL_PWR_TEMP)
    temp_score = (
        0.5 * min(1.0, cpu_excess / (1.0 - _NOMINAL_CPU_TEMP))
        + 0.5 * min(1.0, pwr_excess / (1.0 - _NOMINAL_PWR_TEMP))
    )
    return min(1.0, (flag_score + temp_score) / 2.0)


def _compute_energy_usage(obs: dict) -> float:
    return float(min(1.0, max(0.0, obs.get("current_draw", 0.0))))


def _compute_recovery_latency(obs: dict) -> float:
    cpu_dev = abs(obs.get("cpu_temperature", 0.0) - _NOMINAL_CPU_TEMP)
    pwr_dev = abs(obs.get("power_temperature", 0.0) - _NOMINAL_PWR_TEMP)
    soc_dev = abs(obs.get("battery_soc", _NOMINAL_BATTERY_SOC) - _NOMINAL_BATTERY_SOC)
    return min(1.0, (cpu_dev + pwr_dev + soc_dev) / _MAX_LATENCY)


def _compute_action_cost(action: Union[ActionType, int]) -> float:
    if isinstance(action, int):
        action = ActionType(action)
    raw_cost = ACTION_COSTS.get(int(action), 0.0)
    return raw_cost / _MAX_ACTION_COST if _MAX_ACTION_COST > 0 else 0.0


def _severity_penalty(obs: dict) -> float:
    # -0.5 * severity_level per active fault (use *_severity_level, default 1)
    total = 0.0
    for key in [
        "seu_severity_level",
        "latchup_severity_level",
        "thermal_severity_level",
        "memory_severity_level",
        "power_severity_level",
    ]:
        level = obs.get(key, 0)
        if level:
            total += -0.5 * float(level)
    return total


def _contraindicated_penalty(obs: dict, action: ActionType) -> float:
    # -0.8 if action is contraindicated
    if action == ActionType.POWER_CYCLE and not _is_fault_active(obs, "power_fault_flag"):
        return -0.8
    if action == ActionType.MEMORY_SCRUB and not (
        _is_fault_active(obs, "memory_fault_flag") or _is_fault_active(obs, "seu_flag")
    ):
        return -0.8
    if action == ActionType.THERMAL_THROTTLING and not _is_fault_active(obs, "thermal_fault_flag"):
        return -0.8
    if action == ActionType.SUBSYSTEM_RESET and not any([
        _is_fault_active(obs, "seu_flag"),
        _is_fault_active(obs, "latchup_flag"),
        _is_fault_active(obs, "thermal_fault_flag"),
        _is_fault_active(obs, "memory_fault_flag"),
        _is_fault_active(obs, "power_fault_flag"),
    ]):
        return -0.8
    return 0.0


def _diagnose_healthy_penalty(obs: dict, action: ActionType) -> float:
    # -0.5 if DIAGNOSE used when system fully healthy (no active faults)
    healthy = _compute_uptime(obs) >= 1.0
    if action == ActionType.DIAGNOSE and healthy:
        return -0.5
    return 0.0


def compute_reward(
    obs: dict,
    action: Union[ActionType, int],
    terminated: bool,
) -> Tuple[float, Dict[str, float]]:
    if isinstance(action, int):
        action = ActionType(action)

    uptime = _compute_uptime(obs)
    severity = _compute_fault_severity(obs)
    energy = _compute_energy_usage(obs)
    latency = _compute_recovery_latency(obs)
    cost = _compute_action_cost(action)

    severity_level_penalty = _severity_penalty(obs)
    contraindicated_penalty = _contraindicated_penalty(obs, action)
    diagnose_penalty = _diagnose_healthy_penalty(obs, action)
    failure_penalty = FAILURE_PENALTY if terminated else 0.0

    total = (
        W1_UPTIME * uptime
        - W2_FAULT_SEVERITY * severity
        - W3_ENERGY_USAGE * energy
        - W4_RECOVERY_LATENCY * latency
        - W5_ACTION_COST * cost
        + severity_level_penalty
        + contraindicated_penalty
        + diagnose_penalty
        + failure_penalty
    )

    components: Dict[str, float] = {
        "reward_uptime": float(W1_UPTIME * uptime),
        "reward_fault_severity": float(-W2_FAULT_SEVERITY * severity),
        "reward_energy_usage": float(-W3_ENERGY_USAGE * energy),
        "reward_latency": float(-W4_RECOVERY_LATENCY * latency),
        "reward_action_cost": float(-W5_ACTION_COST * cost),
        "reward_severity_penalty": float(severity_level_penalty),
        "reward_contraindicated": float(contraindicated_penalty),
        "reward_diagnose_misuse": float(diagnose_penalty),
    }
    return float(total), components
