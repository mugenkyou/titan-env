from __future__ import annotations

from titan_env.tasks.base_task import BaseTask, StateDict, Trajectory


def _medium_init(env, seed: int) -> None:
    _ = seed  # deterministic constant setup by design
    env.state.battery_soc = 0.78
    env.state.battery_level = 0.78
    env.state.cpu_temperature = 0.88
    env.state.power_temperature = 0.80
    env.state.temperature = 0.88
    env.state.cpu_load = 0.90
    env.state.current_draw = 0.68
    env.state.thermal_fault_flag = 1
    env.state.seu_flag = 0
    env.state.latchup_flag = 0
    env.state.memory_fault_flag = 0
    env.state.power_fault_flag = 0
    env.state.recent_fault_count = 0.2


def _medium_success(state: StateDict, _trajectory: Trajectory) -> bool:
    return float(state.get("cpu_temperature", 1.0)) < 0.70


def _medium_constraints(trajectory: Trajectory) -> bool:
    if len(trajectory) <= 1:
        return False
    if (len(trajectory) - 1) > 50:
        return False

    currents = [
        float(step["state"].get("current_draw", 1.0))
        for step in trajectory
        if isinstance(step.get("state"), dict)
    ]
    if not currents:
        return False
    avg_current = sum(currents) / len(currents)
    return avg_current <= 0.55


MEDIUM_TASK = BaseTask(
    name="medium_thermal_stabilization",
    max_steps=50,
    init_fn=_medium_init,
    success_fn=_medium_success,
    constraint_fn=_medium_constraints,
)


