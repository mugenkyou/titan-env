"""
titan_env/rewards/__init__.py
Reward module package. Import compute_reward from reward_v1 for RL training.
"""

from .reward_v1 import compute_reward, SURVIVAL_BONUS, FAILURE_PENALTY

__all__ = ["compute_reward", "SURVIVAL_BONUS", "FAILURE_PENALTY"]


