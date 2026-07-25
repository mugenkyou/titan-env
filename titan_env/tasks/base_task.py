from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from titan_env.core.environment.actions import ActionType
from titan_env.core.environment.titan_env import TITANEnv as CoreTITANEnv
from titan_env.interface.openenv_wrapper import TITANEnv as OpenEnvTITANEnv

StateDict = Dict[str, float]
Trajectory = List[Dict[str, Any]]
InitFn = Callable[[CoreTITANEnv, int], None]
SuccessFn = Callable[[StateDict, Trajectory], bool]
ConstraintFn = Callable[[Trajectory], bool]


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True)
class BaseTask:
    """Deterministic, evaluatable task specification for TITAN."""

    name: str
    max_steps: int
    init_fn: InitFn
    success_fn: SuccessFn
    constraint_fn: ConstraintFn

    def build_core_env(self, seed: int) -> CoreTITANEnv:
        """Create task environment for deterministic runs."""
        return CoreTITANEnv(max_steps=self.max_steps, seed=seed)

    def build_openenv_wrapper(self, seed: int) -> OpenEnvTITANEnv:
        """Create OpenEnv wrapper bound to task-configured core env."""
        return OpenEnvTITANEnv(core_env=self.build_core_env(seed))

    def reset(self, env: CoreTITANEnv, seed: int) -> StateDict:
        """Reset env then apply deterministic task-specific initialization."""
        _ = env.reset()
        self.init_fn(env, int(seed))
        self._clamp_state_in_place(env)
        return dict(env._get_observation(env.state, fault_event=None))

    def evaluate(self, trajectory: Trajectory) -> Dict[str, bool]:
        """Evaluate success and constraints from trajectory only."""
        final_state = trajectory[-1]["state"] if trajectory else {}
        return {
            "success": bool(self.success_fn(final_state, trajectory)),
            "constraints_ok": bool(self.constraint_fn(trajectory)),
        }

    def run_actions(
        self,
        actions: Sequence[ActionType | int],
        seed: int,
        env: Optional[CoreTITANEnv] = None,
    ) -> Trajectory:
        """Run a fixed action sequence and return trajectory records."""
        active_env = env if env is not None else self.build_core_env(seed)
        current_obs = self.reset(active_env, seed)

        trajectory: Trajectory = [
            {
                "step": 0,
                "state": current_obs,
                "action": None,
                "done": False,
                "info": {"task": self.name},
            }
        ]

        for idx, action in enumerate(actions, start=1):
            obs, done, info = active_env.step(int(action))
            trajectory.append(
                {
                    "step": idx,
                    "state": dict(obs),
                    "action": int(action),
                    "done": bool(done),
                    "info": info,
                }
            )
            if done or idx >= self.max_steps:
                break

        return trajectory

    def run_policy(
        self,
        policy_fn: Callable[[StateDict, int], ActionType | int],
        seed: int,
        env: Optional[CoreTITANEnv] = None,
    ) -> Trajectory:
        """Run policy callback over at most max_steps and return trajectory."""
        active_env = env if env is not None else self.build_core_env(seed)
        current_obs = self.reset(active_env, seed)

        trajectory: Trajectory = [
            {
                "step": 0,
                "state": current_obs,
                "action": None,
                "done": False,
                "info": {"task": self.name},
            }
        ]

        for step in range(1, self.max_steps + 1):
            action = policy_fn(current_obs, step - 1)
            obs, done, info = active_env.step(int(action))
            current_obs = dict(obs)
            trajectory.append(
                {
                    "step": step,
                    "state": current_obs,
                    "action": int(action),
                    "done": bool(done),
                    "info": info,
                }
            )
            if done:
                break

        return trajectory

    @staticmethod
    def deterministic_uniform(seed: int, low: float, high: float, salt: int = 0) -> float:
        rng = np.random.default_rng(int(seed) + int(salt))
        return float(rng.uniform(low, high))

    @staticmethod
    def _clamp_state_in_place(env: CoreTITANEnv) -> None:
        state = env.state.clamp_all()
        env.state.battery_level = _clip01(state.battery_level)
        env.state.temperature = _clip01(state.temperature)
        env.state.cpu_health = _clip01(state.cpu_health)
        env.state.communication_health = _clip01(state.communication_health)
        env.state.voltage = _clip01(state.voltage)
        env.state.current_draw = _clip01(state.current_draw)
        env.state.battery_soc = _clip01(state.battery_soc)
        env.state.cpu_temperature = _clip01(state.cpu_temperature)
        env.state.power_temperature = _clip01(state.power_temperature)
        env.state.memory_integrity = _clip01(state.memory_integrity)
        env.state.cpu_load = _clip01(state.cpu_load)
        env.state.seu_flag = int(bool(state.seu_flag))
        env.state.latchup_flag = int(bool(state.latchup_flag))
        env.state.thermal_fault_flag = int(bool(state.thermal_fault_flag))
        env.state.memory_fault_flag = int(bool(state.memory_fault_flag))
        env.state.power_fault_flag = int(bool(state.power_fault_flag))
        env.state.recent_fault_count = _clip01(state.recent_fault_count)


