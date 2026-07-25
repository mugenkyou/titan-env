"""
TITAN — Reward Module: Version 2 (Research-Grade)
titan_env/rewards/reward_v2.py

Multi-objective normalized reward for RL training that encourages:
  - Long satellite uptime
  - Fast fault recovery
  - Low energy consumption
  - Minimal use of destructive/costly actions

Reward equation
---------------
  R = w1·uptime - w2·fault_severity - w3·energy_usage
        - w4·recovery_latency - w5·action_cost

On terminal failure an additional flat penalty FAILURE_PENALTY is added.

All five component terms are independently normalized to [0, 1] before
weighting, so the weight vector directly controls the relative importance
of each objective.

Typical range: [-3, +1] per step (failure: up to –53).

Usage
-----
    from titan_env.core.rewards.reward_v2 import compute_reward
    total, components = compute_reward(obs_dict, action, terminated)

Backward compatibility
----------------------
reward_v1 is untouched and still importable for baseline experiments.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

from titan_env.core.environment.actions import ActionType, ACTION_COSTS


# ---------------------------------------------------------------------------
# Optimized weights for 1000-step survival training
# ---------------------------------------------------------------------------

W1_UPTIME:            float = 1.5   # Increased: prioritize staying alive
W2_FAULT_SEVERITY:    float = 1.0   # Reduced: don't over-penalize faults
W3_ENERGY_USAGE:      float = 0.1   # Reduced: battery conservation matters less than survival
W4_RECOVERY_LATENCY:  float = 0.5   # Reduced: don't rush to nominal, survive first
W5_ACTION_COST:       float = 0.05  # Reduced: allow more actions when needed

FAILURE_PENALTY:      float = -30.0 # Reduced: less harsh, encourages exploration

# Max number of fault flags that can be simultaneously active (5 binary flags)
_MAX_FAULT_FLAGS: int = 5

# Nominal subsystem values (used for recovery_latency distance computation)
_NOMINAL_CPU_TEMP:   float = 0.30
_NOMINAL_PWR_TEMP:   float = 0.25
_NOMINAL_BATTERY_SOC: float = 0.90

# Max possible recovery latency (|cpu_temp - nom| + |pwr_temp - nom| + |soc - nom|)
# Worst case: cpu_temp=1.0, pwr_temp=1.0, soc=0.0
_MAX_LATENCY: float = (1.0 - _NOMINAL_CPU_TEMP) + (1.0 - _NOMINAL_PWR_TEMP) + _NOMINAL_BATTERY_SOC

# Max action cost from ACTION_COSTS table
_MAX_ACTION_COST: float = max(ACTION_COSTS.values()) if ACTION_COSTS else 1.0

# Max total weighted reward per step (no faults, no action cost, no failure)
MAX_REWARD_PER_STEP: float = W1_UPTIME


# ---------------------------------------------------------------------------
# Component functions (each returns a value in [0, 1])
# ---------------------------------------------------------------------------

def _compute_uptime(obs: dict) -> float:
    """
    1.0 if no fault flags are active, else 0.0.

    Fault flags: seu_flag, latchup_flag, thermal_fault_flag,
                 memory_fault_flag, power_fault_flag.
    """
    flags = (
        obs.get("seu_flag",            0.0)
        + obs.get("latchup_flag",      0.0)
        + obs.get("thermal_fault_flag",0.0)
        + obs.get("memory_fault_flag", 0.0)
        + obs.get("power_fault_flag",  0.0)
    )
    return 1.0 if flags == 0.0 else 0.0


def _compute_fault_severity(obs: dict) -> float:
    """
    Normalised severity score in [0, 1].

    severity = (active_flags / max_flags)
                + 0.5 * max(0, cpu_temperature - nominal) / (1 - nominal)
                + 0.5 * max(0, power_temperature - nominal) / (1 - nominal)

    The temperature excess terms each contribute up to 0.5,
    fault flags up to 1.0; total is divided by 2.0 to normalise.
    """
    flag_sum = (
        obs.get("seu_flag",            0.0)
        + obs.get("latchup_flag",      0.0)
        + obs.get("thermal_fault_flag",0.0)
        + obs.get("memory_fault_flag", 0.0)
        + obs.get("power_fault_flag",  0.0)
    )
    flag_score = flag_sum / _MAX_FAULT_FLAGS

    cpu_excess = max(0.0, obs.get("cpu_temperature",   0.0) - _NOMINAL_CPU_TEMP)
    pwr_excess = max(0.0, obs.get("power_temperature", 0.0) - _NOMINAL_PWR_TEMP)
    temp_norm  = 1.0 - _NOMINAL_CPU_TEMP          # both nominals similar; use cpu

    temp_score = 0.5 * min(1.0, cpu_excess / temp_norm) + \
                 0.5 * min(1.0, pwr_excess / (1.0 - _NOMINAL_PWR_TEMP))

    # Total severity in [0, 1]
    return min(1.0, (flag_score + temp_score) / 2.0)


def _compute_energy_usage(obs: dict) -> float:
    """
    current_draw from the state vector; already normalised to [0, 1].
    Higher current draw → higher penalty.
    """
    return float(min(1.0, max(0.0, obs.get("current_draw", 0.0))))


def _compute_recovery_latency(obs: dict) -> float:
    """
    Normalised distance from nominal subsystem operating point:

      latency = |cpu_temperature  - nominal_cpu_temp|
              + |power_temperature - nominal_pwr_temp|
              + |battery_soc      - nominal_soc|

    Scaled to [0, 1] by dividing by _MAX_LATENCY.
    """
    cpu_dev = abs(obs.get("cpu_temperature",   0.0)  - _NOMINAL_CPU_TEMP)
    pwr_dev = abs(obs.get("power_temperature", 0.0)  - _NOMINAL_PWR_TEMP)
    soc_dev = abs(obs.get("battery_soc",       _NOMINAL_BATTERY_SOC) - _NOMINAL_BATTERY_SOC)

    raw = cpu_dev + pwr_dev + soc_dev
    return min(1.0, raw / _MAX_LATENCY)


def _compute_action_cost(action: Union[ActionType, int]) -> float:
    """
    Normalised action cost from ACTION_COSTS table.

    Returns value in [0, 1] by dividing by _MAX_ACTION_COST.
    """
    if isinstance(action, int):
        action = ActionType(action)
    raw_cost = ACTION_COSTS.get(int(action), 0.0)
    return raw_cost / _MAX_ACTION_COST if _MAX_ACTION_COST > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward(
    obs:        dict,
    action:     Union[ActionType, int],
    terminated: bool,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the v2 multi-objective reward.

    Parameters
    ----------
    obs : dict
        Observation dictionary (keys from OBS_KEYS).
        Must contain at least the fault flag, temperature, energy, and soc keys.
    action : ActionType | int
        The action taken this step.
    terminated : bool
        True on subsystem failure (not timeout).

    Returns
    -------
    total_reward : float
    components   : Dict[str, float]
        Keys: reward_uptime, reward_fault_severity, reward_energy_usage,
              reward_latency, reward_action_cost
    """
    uptime   = _compute_uptime(obs)
    severity = _compute_fault_severity(obs)
    energy   = _compute_energy_usage(obs)
    latency  = _compute_recovery_latency(obs)
    cost     = _compute_action_cost(action)

    failure_penalty = FAILURE_PENALTY if terminated else 0.0

    total = (
        W1_UPTIME           * uptime
        - W2_FAULT_SEVERITY * severity
        - W3_ENERGY_USAGE   * energy
        - W4_RECOVERY_LATENCY * latency
        - W5_ACTION_COST    * cost
        + failure_penalty
    )

    components: Dict[str, float] = {
        "reward_uptime":          float(W1_UPTIME           * uptime),
        "reward_fault_severity":  float(-W2_FAULT_SEVERITY  * severity),
        "reward_energy_usage":    float(-W3_ENERGY_USAGE    * energy),
        "reward_latency":         float(-W4_RECOVERY_LATENCY * latency),
        "reward_action_cost":     float(-W5_ACTION_COST      * cost),
    }
    return float(total), components


# ---------------------------------------------------------------------------
# Convenience: component weights as a named dict for experiment logging
# ---------------------------------------------------------------------------

WEIGHTS: Dict[str, float] = {
    "w1_uptime":           W1_UPTIME,
    "w2_fault_severity":   W2_FAULT_SEVERITY,
    "w3_energy_usage":     W3_ENERGY_USAGE,
    "w4_recovery_latency": W4_RECOVERY_LATENCY,
    "w5_action_cost":      W5_ACTION_COST,
}


