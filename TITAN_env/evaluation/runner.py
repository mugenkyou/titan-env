from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from TITAN_env.core.environment.state_model import OBS_KEYS
from TITAN_env.core.rewards.reward_v2 import compute_reward as compute_reward_v2
from TITAN_env.interface.action_mapping import discrete_from_command
from TITAN_env.interface.llm_interface import parse_action, render_observation, get_available_commands
from TITAN_env.interface.models import Observation
from TITAN_env.evaluation.scoring import score_trajectory
from TITAN_env.evaluation.trajectory import EvaluationTrajectory
from TITAN_env.tasks.registry import resolve_task_bundle, list_registered_tasks


TASK_DESCRIPTIONS: Dict[str, str] = {
    "easy_single_fault_recovery": "Recover from a single SEU fault efficiently.",
    "medium_thermal_stabilization": "Reduce thermal stress under high load conditions.",
    "hard_multi_fault_survival": "Survive a high-radiation multi-fault scenario.",
}


def _ensure_observation_model(raw_observation: Dict[str, Any]) -> Observation:
    faults = []
    if raw_observation.get("seu_flag", 0.0) >= 0.5:
        faults.append("seu")
    if raw_observation.get("latchup_flag", 0.0) >= 0.5:
        faults.append("latchup")
    if raw_observation.get("thermal_fault_flag", 0.0) >= 0.5:
        faults.append("thermal")
    if raw_observation.get("memory_fault_flag", 0.0) >= 0.5:
        faults.append("memory")
    if raw_observation.get("power_fault_flag", 0.0) >= 0.5:
        faults.append("power")

    return Observation(
        voltage=float(raw_observation.get("voltage", 0.0)),
        current_draw=float(raw_observation.get("current_draw", 0.0)),
        battery=float(raw_observation.get("battery_soc", 0.0)),
        cpu_temp=float(raw_observation.get("cpu_temperature", 0.0)),
        power_temp=float(raw_observation.get("power_temperature", 0.0)),
        memory=float(raw_observation.get("memory_integrity", 0.0)),
        cpu_load=float(raw_observation.get("cpu_load", 0.0)),
        signal=float(max(0.0, min(1.0, 1.0 - float(raw_observation.get("current_draw", 0.0))))),
        recent_fault_count=float(raw_observation.get("recent_fault_count", 0.0)),
        faults=faults,
    )


def _build_prompt(task_name: str, observation_text: str, step: int, max_steps: int) -> str:
    description = TASK_DESCRIPTIONS.get(task_name, task_name)
    commands = ", ".join(get_available_commands())
    return (
        f"Task: {task_name}\n"
        f"Goal: {description}\n"
        f"Step: {step}/{max_steps}\n"
        f"Valid commands: {commands}\n\n"
        f"Observation:\n{observation_text}\n\n"
        "Return exactly one command name."
    )


def _call_model(model: Callable[[str], str], prompt: str) -> str:
    try:
        response = model(prompt)
    except Exception:
        return "no_action"
    return str(response)


def _compute_reward(action: int, next_observation: Dict[str, Any], done: bool, terminated: bool) -> float:
    total, _ = compute_reward_v2(next_observation, action, terminated and done)
    return float(total)


def run_task_with_trajectory(
    task_name: str,
    model: Callable[[str], str],
    seed: int,
    max_steps: Optional[int] = None,
) -> tuple[float, EvaluationTrajectory]:
    """Run a registered task with an LLM-like model and return score plus trajectory."""
    bundle = resolve_task_bundle(task_name)
    task = bundle.task
    active_max_steps = min(task.max_steps, int(max_steps)) if max_steps is not None else task.max_steps

    env = task.build_core_env(seed)
    initial_observation = task.reset(env, seed)

    trajectory = EvaluationTrajectory()
    trajectory.start(initial_observation)

    for step_index in range(1, active_max_steps + 1):
        observation_model = _ensure_observation_model(trajectory.observations[-1])
        observation_text = render_observation(observation_model)
        prompt = _build_prompt(task.name, observation_text, step_index, active_max_steps)
        model_output = _call_model(model, prompt)
        action_command = parse_action(model_output).command
        action_value = discrete_from_command(action_command)

        next_observation, done, info = env.step(action_value)
        failure_reason = str(info.get("failure_reason", ""))
        terminated = bool(failure_reason) and not failure_reason.startswith("MAX_STEPS")
        reward = _compute_reward(action_value, next_observation, done, terminated)

        trajectory.append_step(
            action=action_command,
            reward=reward,
            done=bool(done),
            next_observation=next_observation,
            info=info,
        )

        if done:
            break

    score = score_trajectory(task.name, trajectory)
    return score, trajectory


def run_task(task_name: str, model: Callable[[str], str], seed: int) -> float:
    """Run a task and return only the final score."""
    score, _ = run_task_with_trajectory(task_name, model, seed)
    return score


def run_all_tasks(model: Callable[[str], str], seed: int) -> Dict[str, float]:
    """Run all registered tasks with a shared model and return per-task scores."""
    results: Dict[str, float] = {}
    for task_name in list_registered_tasks():
        results[task_name] = run_task(task_name, model, seed)
    return results


__all__ = ["run_task", "run_task_with_trajectory", "run_all_tasks"]
