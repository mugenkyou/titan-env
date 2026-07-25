from __future__ import annotations

from TITAN_env.tasks.base_task import BaseTask, StateDict, Trajectory


def _easy_init(env, seed: int) -> None:
    _ = seed  # deterministic constant setup by design
    env.state.battery_soc = 0.60
    env.state.battery_level = 0.60
    env.state.memory_integrity = 0.88  # start closer to goal
    env.state.cpu_temperature = 0.42
    env.state.power_temperature = 0.35
    env.state.cpu_load = 0.35
    env.state.seu_flag = 1
    env.state.latchup_flag = 0
    env.state.thermal_fault_flag = 0
    env.state.memory_fault_flag = 0
    env.state.power_fault_flag = 0
    env.state.recent_fault_count = 0.1


def _easy_success(state: StateDict, _trajectory: Trajectory) -> bool:
    return float(state.get("memory_integrity", 0.0)) > 0.90


def _easy_constraints(trajectory: Trajectory) -> bool:
    if len(trajectory) <= 1:
        return False
    if (len(trajectory) - 1) > 30:
        return False
    last = trajectory[-1]
    if bool(last.get("done", False)):
        reason = (last.get("info") or {}).get("failure_reason", "")
        if reason and "MAX_STEPS" not in reason:
            return False
    return True


EASY_TASK = BaseTask(
    name="easy_single_fault_recovery",
    max_steps=30,
    init_fn=_easy_init,
    success_fn=_easy_success,
    constraint_fn=_easy_constraints,
)
