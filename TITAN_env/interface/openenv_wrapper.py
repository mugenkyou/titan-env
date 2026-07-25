from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from TITAN_env.core.environment.TITAN_env import TITANEnv as CoreTITANEnv
from TITAN_env.core.rewards.reward_v2 import compute_reward as compute_reward_v2

from TITAN_env.interface.action_mapping import discrete_from_command
from TITAN_env.interface.models import Action, Observation, Reward


class TITANEnv:
    """OpenEnv-compatible wrapper around the existing TITAN core environment."""

    def __init__(self, core_env: Optional[CoreTITANEnv] = None) -> None:
        self.core_env = core_env if core_env is not None else CoreTITANEnv()
        self._last_observation: Optional[Observation] = None

    def reset(self) -> Observation:
        raw_obs = self.core_env.reset()
        structured = self._observation_from_dict(raw_obs)
        self._last_observation = structured
        return structured

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        discrete_action = discrete_from_command(action.command)
        raw_obs, done, info = self.core_env.step(discrete_action)

        structured_obs = self._observation_from_dict(raw_obs)
        reward_value, _ = compute_reward_v2(raw_obs, discrete_action, bool(done))
        structured_reward = Reward(value=float(reward_value))

        self._last_observation = structured_obs
        return structured_obs, structured_reward, bool(done), info

    def state(self) -> Observation:
        if self._last_observation is not None:
            return self._last_observation

        state = self.core_env.state
        raw_obs = {
            "voltage": state.voltage,
            "current_draw": state.current_draw,
            "battery_soc": state.battery_soc,
            "cpu_temperature": state.cpu_temperature,
            "power_temperature": state.power_temperature,
            "memory_integrity": state.memory_integrity,
            "cpu_load": state.cpu_load,
            "seu_flag": float(state.seu_flag),
            "latchup_flag": float(state.latchup_flag),
            "thermal_fault_flag": float(state.thermal_fault_flag),
            "memory_fault_flag": float(state.memory_fault_flag),
            "power_fault_flag": float(state.power_fault_flag),
            "recent_fault_count": state.recent_fault_count,
        }
        structured = self._observation_from_dict(raw_obs)
        self._last_observation = structured
        return structured

    @staticmethod
    def _safe01(value: float) -> float:
        if value is None:
            return 0.0
        numeric = float(value)
        if not math.isfinite(numeric):
            return 0.0
        return max(0.0, min(1.0, numeric))

    def _observation_from_dict(self, raw_obs: Dict[str, float]) -> Observation:
        faults = self._fault_list(raw_obs)
        signal = self._safe01(1.0 - raw_obs.get("current_draw", 0.0))

        return Observation(
            voltage=self._safe01(raw_obs.get("voltage", 0.0)),
            current_draw=self._safe01(raw_obs.get("current_draw", 0.0)),
            battery=self._safe01(raw_obs.get("battery_soc", 0.0)),
            cpu_temp=self._safe01(raw_obs.get("cpu_temperature", 0.0)),
            power_temp=self._safe01(raw_obs.get("power_temperature", 0.0)),
            memory=self._safe01(raw_obs.get("memory_integrity", 0.0)),
            cpu_load=self._safe01(raw_obs.get("cpu_load", 0.0)),
            signal=signal,
            recent_fault_count=self._safe01(raw_obs.get("recent_fault_count", 0.0)),
            faults=faults,
        )

    @staticmethod
    def _fault_list(raw_obs: Dict[str, float]) -> List[str]:
        faults: List[str] = []
        if raw_obs.get("seu_flag", 0.0) >= 0.5:
            faults.append("seu")
        if raw_obs.get("latchup_flag", 0.0) >= 0.5:
            faults.append("latchup")
        if raw_obs.get("thermal_fault_flag", 0.0) >= 0.5:
            faults.append("thermal")
        if raw_obs.get("memory_fault_flag", 0.0) >= 0.5:
            faults.append("memory")
        if raw_obs.get("power_fault_flag", 0.0) >= 0.5:
            faults.append("power")
        return faults
