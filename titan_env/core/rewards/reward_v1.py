
"""
TITAN — Reward Module: Version 1
TITAN/rewards/reward_v1.py

Canonical v1 reward function for supervised + RL training.

Reward components
-----------------
reward_survival        = +1.0  per step alive
reward_action_cost     = -ACTION_COSTS[action]   (penalty for costly interventions)
reward_failure_penalty = -50.0 on terminal failure, 0 otherwise

total = reward_survival + reward_action_cost + reward_failure_penalty

Usage
-----
    from titan_env.core.rewards.reward_v1 import compute_reward
    total, components = compute_reward(state, action, terminated)
"""

from __future__ import annotations

from typing import Dict, Tuple

from titan_env.core.environment.actions import ActionType, ACTION_COSTS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SURVIVAL_BONUS:   float = 1.0
FAILURE_PENALTY:  float = -50.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward(
    action: "ActionType | int",
    terminated: bool,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the v1 reward signal.

    Parameters
    ----------
    action : ActionType | int
        The action taken this step.
    terminated : bool
        True if the episode ended due to subsystem failure (not timeout).

    Returns
    -------
    total_reward : float
        Scalar reward to pass to the RL agent.
    components : Dict[str, float]
        Breakdown for logging: survival, action_cost, failure_penalty.
    """
    if isinstance(action, int):
        action = ActionType(action)

    cost             = ACTION_COSTS.get(int(action), 0.0)
    survival         = SURVIVAL_BONUS
    action_cost      = -cost
    failure_penalty  = FAILURE_PENALTY if terminated else 0.0

    total = survival + action_cost + failure_penalty

    components: Dict[str, float] = {
        "reward_survival":        survival,
        "reward_action_cost":     action_cost,
        "reward_failure_penalty": failure_penalty,
    }
    return float(total), components


