"""
TITAN — Phase 4 + 5: RL Interface
gym_env.py

Wraps TITANEnv as a Gymnasium-compatible environment for use with SB3 and
other RL frameworks. No physics, fault logic, or action definitions are changed.

Observation space (Box, float32, shape=(11,)) — in order:
  [0]  voltage              — regulated bus voltage
  [1]  current_draw         — load current draw
  [2]  battery_soc          — battery state-of-charge
  [3]  cpu_temperature      — CPU die temperature
  [4]  power_temperature    — power regulator temperature
  [5]  memory_integrity     — SRAM/flash integrity
  [6]  cpu_load             — processor utilisation
  [7]  delta_voltage        — |voltage(t) - voltage(t-1)|
  [8]  thermal_gradient     — |cpu_temperature - power_temperature|
  [9]  memory_drift         — |memory_integrity(t) - memory_integrity(t-1)|
  [10] step_fraction        — step / max_steps

Action space: Discrete(8) → ActionType.{NO_ACTION, SUBSYSTEM_RESET, MEMORY_SCRUB,
                                           LOAD_SHEDDING, POWER_CYCLE,
                                           THERMAL_THROTTLING, ISOLATE_SUBSYSTEM,
                                           DIAGNOSE}

Reward:
  +SURVIVAL_BONUS per step alive
  - ACTION_COST[action]
  - FAILURE_PENALTY on terminal step (failure only, not timeout)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .titan_env import TITANEnv
from .state_model import SubsystemState, StateBounds
from .fault_injection import FaultInjector, RadiationProfile
from .actions import ActionType, ACTION_COSTS
from ..rewards.reward_v1 import compute_reward as _reward_v1
from ..rewards.reward_v2 import compute_reward as _reward_v2
from ..rewards.reward_v3 import compute_reward as _reward_v3


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

SURVIVAL_BONUS: float = 1.0
FAILURE_PENALTY: float = -50.0
OBS_ARRAY_DIM: int = 11

# ACTION_COSTS is imported from actions.py (canonical source of truth)
# Re-exported here for backward compat with any external code that imported
# it from gym_env previously.


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class TITANGymEnv(gym.Env):
    """
    Gymnasium wrapper around TITANEnv for RL training and evaluation.

    Compatible with Stable-Baselines3 and any Gymnasium-compliant framework.

    Parameters
    ----------
    fault_injector : FaultInjector | None
        Radiation fault engine. None → deterministic Phase 1 behaviour.
    initial_state : SubsystemState | None
    bounds : StateBounds | None
    max_steps : int
        Episode truncation limit (default 1000).
    seed : int | None
        Physics noise RNG seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        fault_injector:  Optional[FaultInjector]  = None,
        initial_state:   Optional[SubsystemState]  = None,
        bounds:          Optional[StateBounds]      = None,
        max_steps:       int                        = 1000,
        seed:            Optional[int]              = None,
        reward_version:  str                        = "v2",
        training_mode:   bool                       = True,
    ) -> None:
        """
        Parameters
        ----------
        reward_version : str
            'v3'           — severity-aware reward with contraindication penalties (reward_v3).
            'v2' (default) — multi-objective structured reward (reward_v2).
            'v1'           — simple survival/action-cost/failure reward (reward_v1).
        """

        super().__init__()

        self._core = TITANEnv(
            initial_state=initial_state,
            bounds=bounds,
            max_steps=max_steps,
            fault_injector=fault_injector,
            seed=seed,
        )
        self._injector = fault_injector
        if self._injector is not None:
            self._injector.training_mode = training_mode
        self._max_steps = max_steps
        self._radiation_intensity: float = self._compute_radiation_intensity()
        self._reward_version: str = reward_version.lower()
        self._obs_rng = np.random.default_rng(seed)

        # --- Spaces ---
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_ARRAY_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(8)   # NO_ACTION … DIAGNOSE


        # Track last obs for trajectory logger support
        self._last_obs: np.ndarray = np.zeros(OBS_ARRAY_DIM, dtype=np.float32)
        self._prev_voltage: float = 0.0
        self._prev_memory_integrity: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial conditions.

        Parameters
        ----------
        seed : int | None
            Seeds numpy's global RNG for SB3 compatibility.
        options : dict | None
            Ignored. Present for API compliance.

        Returns
        -------
        obs : np.ndarray, shape (11,)
        info : dict
        """
        super().reset(seed=seed)

        obs_dict = self._core.reset()
        self._prev_voltage = float(obs_dict.get("voltage", 0.0))
        self._prev_memory_integrity = float(obs_dict.get("memory_integrity", 0.0))
        obs = self._obs_to_array(obs_dict, step=0)
        self._last_obs = obs

        info = {"step": 0}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply action and advance one timestep.

        Parameters
        ----------
        action : int
            Index into ActionType (0-7).

        Returns
        -------
        obs         : np.ndarray, shape (11,)
        reward      : float
        terminated  : bool — True on subsystem failure
        truncated   : bool — True on max_steps timeout
        info        : dict
        """
        obs_dict, done, core_info = self._core.step(action)
        step_idx = int(core_info.get("step", 0))
        obs = self._obs_to_array(obs_dict, step=step_idx)
        self._last_obs = obs

        # Determine termination type
        reason = core_info.get("failure_reason", "")
        terminated = done and not reason.startswith("MAX_STEPS")
        truncated  = done and reason.startswith("MAX_STEPS")

        # Compute reward via selected reward version
        if self._reward_version == "v3":
            reward, reward_components = _reward_v3(obs_dict, action, terminated)
        elif self._reward_version == "v2":
            reward, reward_components = _reward_v2(obs_dict, action, terminated)
        else:  # v1 (baseline)
            reward, reward_components = _reward_v1(action, terminated)

        info = {
            **core_info,
            "terminated":         terminated,
            "truncated":          truncated,
            "reward":             reward,
            "reward_version":     self._reward_version,
            **reward_components,
        }

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """No-op. Present for API compliance."""
        pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def core(self) -> TITANEnv:
        """Access the underlying TITANEnv (true state, history, fault log)."""
        return self._core

    @property
    def radiation_intensity(self) -> float:
        """Normalised radiation severity [0, 1] for the current episode."""
        return self._radiation_intensity

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _obs_to_array(self, obs_dict: Dict[str, float], step: int) -> np.ndarray:
        """Convert telemetry dict to the 11D policy observation vector."""
        voltage = float(obs_dict.get("voltage", 0.0))
        current_draw = float(obs_dict.get("current_draw", 0.0))
        battery_soc = float(obs_dict.get("battery_soc", 0.0))
        cpu_temperature = float(obs_dict.get("cpu_temperature", 0.0))
        power_temperature = float(obs_dict.get("power_temperature", 0.0))
        memory_integrity = float(obs_dict.get("memory_integrity", 0.0))
        cpu_load = float(obs_dict.get("cpu_load", 0.0))

        delta_voltage = abs(voltage - self._prev_voltage)
        thermal_gradient = abs(cpu_temperature - power_temperature)
        memory_drift = abs(memory_integrity - self._prev_memory_integrity)
        step_fraction = float(step) / float(max(self._max_steps, 1))

        arr = np.array(
            [
                voltage,
                current_draw,
                battery_soc,
                cpu_temperature,
                power_temperature,
                memory_integrity,
                cpu_load,
                delta_voltage,
                thermal_gradient,
                memory_drift,
                step_fraction,
            ],
            dtype=np.float32,
        )
        arr = self._add_observation_noise(arr)

        self._prev_voltage = voltage
        self._prev_memory_integrity = memory_integrity
        return arr

    def _add_observation_noise(self, arr: np.ndarray) -> np.ndarray:
        """Apply Gaussian sensor noise scaled by episode radiation intensity."""
        std = 0.01 * self._radiation_intensity
        if std <= 0.0:
            return np.clip(arr, 0.0, 1.0).astype(np.float32)
        noise = self._obs_rng.normal(0.0, std, size=arr.shape).astype(np.float32)
        return np.clip(arr + noise, 0.0, 1.0).astype(np.float32)

    def _compute_reward(self, action: int, terminated: bool) -> float:
        """
        Delegates to reward_v1.compute_reward — canonical reward module.
        Returns the total scalar reward.
        """
        total, _ = _reward_v1(action, terminated)
        return total

    def _compute_radiation_intensity(self) -> float:
        """
        Map the active FaultInjector profile to a normalised intensity scalar.

        Uses the mean of per-step fault probabilities as a proxy for severity.
        Range [0, 1]. 0.0 if no injector is attached.
        """
        if self._injector is None:
            return 0.0
        p = self._injector.profile
        raw = (p.p_seu + p.p_latchup + p.p_telemetry) / 3.0
        return float(min(raw, 1.0))


