from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from titan_env.core.environment.titan_env import TITANEnv as CoreTITANEnv
from titan_env.core.rewards.reward_v2 import compute_reward as compute_reward_v2

from titan_env.interface.action_mapping import discrete_from_command
from titan_env.interface.models import Action, Observation, Reward


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
        command = (action.command or "").strip().lower()
        discrete_action = discrete_from_command(command)
        raw_obs, done, info = self.core_env.step(discrete_action)

        structured_obs = self._observation_from_dict(raw_obs)
        reward_value, _ = compute_reward_v2(raw_obs, discrete_action, bool(done))
        structured_reward = Reward(value=float(reward_value))

        safe_info = self._sanitize_info(info)
        if command == "diagnose":
            safe_info.update(self._diagnostic_info(raw_obs))

        self._last_observation = structured_obs
        return structured_obs, structured_reward, bool(done), safe_info

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
            recent_fault_count=0.0,
            faults=faults,
        )

    @staticmethod
    def _fault_list(raw_obs: Dict[str, float]) -> List[str]:
        return []

    @staticmethod
    def _sanitize_info(info: Optional[Dict]) -> Dict:
        if not info:
            return {}
        safe: Dict = {}
        for key, value in info.items():
            if "fault" in str(key).lower():
                continue
            safe[key] = value
        return safe

    @staticmethod
    def _diagnostic_info(raw_obs: Dict[str, float]) -> Dict[str, float | str | None]:
        severity_map = {
            "seu": float(raw_obs.get("seu_severity", raw_obs.get("seu_flag", 0.0))),
            "latchup": float(raw_obs.get("latchup_severity", raw_obs.get("latchup_flag", 0.0))),
            "thermal": float(raw_obs.get("thermal_severity", raw_obs.get("thermal_fault_flag", 0.0))),
            "memory": float(raw_obs.get("memory_severity", raw_obs.get("memory_fault_flag", 0.0))),
            "power": float(raw_obs.get("power_severity", raw_obs.get("power_fault_flag", 0.0))),
        }
        active = [(name, level) for name, level in severity_map.items() if level > 0.0]
        if not active:
            return {"diagnose_fault": None, "diagnose_severity": 0.0}
        fault_name, severity = max(active, key=lambda item: item[1])
        return {"diagnose_fault": fault_name, "diagnose_severity": float(severity)}


