from __future__ import annotations

from dataclasses import replace

from titan_env.core.environment.fault_injection import FaultInjector, INTENSITY_PROFILES
from titan_env.core.environment.titan_env import TITANEnv as CoreTITANEnv
from titan_env.tasks.base_task import BaseTask, StateDict, Trajectory


def _hard_init(env, seed: int) -> None:
    _ = seed  # deterministic constant setup by design
    env.state.battery_soc = 0.56
    env.state.battery_level = 0.56
    env.state.voltage = 0.68
    env.state.current_draw = 0.66
    env.state.cpu_temperature = 0.86
    env.state.power_temperature = 0.82
    env.state.temperature = 0.86
    env.state.memory_integrity = 0.72
    env.state.cpu_load = 0.88
    env.state.cpu_health = 0.72
    env.state.communication_health = 0.76

    env.state.seu_flag = 1
    env.state.thermal_fault_flag = 1
    env.state.power_fault_flag = 1
    env.state.latchup_flag = 0
    env.state.memory_fault_flag = 0
    env.state.recent_fault_count = 0.5


def _hard_success(_state: StateDict, trajectory: Trajectory) -> bool:
    steps_survived = len(trajectory) - 1
    if steps_survived < 100:
        return False
    if trajectory and bool(trajectory[-1].get("done", False)):
        return False
    return True


def _hard_constraints(trajectory: Trajectory) -> bool:
    if len(trajectory) <= 1:
        return False

    min_battery = min(float(step["state"].get("battery_soc", 0.0)) for step in trajectory)
    if min_battery < 0.20:
        return False

    for step in trajectory:
        if bool(step.get("done", False)):
            reason = (step.get("info") or {}).get("failure_reason", "")
            if reason and "MAX_STEPS" not in reason:
                return False
    return True


class HardMultiFaultSurvivalTask(BaseTask):
    def __init__(self) -> None:
        super().__init__(
            name="hard_multi_fault_survival",
            max_steps=120,
            init_fn=_hard_init,
            success_fn=_hard_success,
            constraint_fn=_hard_constraints,
        )

    def build_core_env(self, seed: int) -> CoreTITANEnv:
        profile = replace(
            INTENSITY_PROFILES["high"],
            radiation_intensity=0.50,
            p_seu=0.0,
            p_latchup=0.0,
            p_telemetry=0.0,
            # Balanced fault rates - challenging but survivable
            base_seu_rate=0.02,
            base_latchup_rate=0.01,
            base_thermal_rate=0.04,
            base_memory_rate=0.02,
            base_power_rate=0.02,
            seed=int(seed),
        )
        injector = FaultInjector(profile)
        return CoreTITANEnv(max_steps=self.max_steps, seed=seed, fault_injector=injector)


HARD_TASK = HardMultiFaultSurvivalTask()


