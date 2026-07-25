"""Compatibility module for the core TITAN environment.

The concrete implementation lives in ``stratos_env.py``. This module exists so
imports like ``from titan_env.core.environment.titan_env import TITANEnv`` keep
working across the codebase.
"""

from .stratos_env import OBS_DIM, OBS_KEYS, TITANEnv

__all__ = ["TITANEnv", "OBS_KEYS", "OBS_DIM"]


