#!/usr/bin/env python
"""
No-Op Baseline Agent

Runs all registered tasks using only 'no_action' at every step.
Uses the existing OpenEnv wrapper interface.
Prints scores in the same [START][STEP][END] log format as inference.py.

This script uses only titan_env modules and avoids the main inference entry point.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Tuple

from titan_env.evaluation.scoring import score_trajectory
from titan_env.evaluation.trajectory import EvaluationTrajectory
from titan_env.interface.models import Action, Observation
from titan_env.interface.openenv_wrapper import TITANEnv as OpenEnvWrapper
from titan_env.tasks.registry import available_task_names, resolve_task_bundle


ENV_NAME = "titan_env.interface.openenv_wrapper.TITANEnv"
MODEL_NAME = "baseline-noop"


def _normalize_dict(model: Any) -> Dict[str, Any]:
    """Convert a model object to a dictionary."""
    if hasattr(model, "model_dump"):
        return dict(model.model_dump())
    if hasattr(model, "dict"):
        return dict(model.dict())
    return dict(model)


def _format_error(error: Exception | str | None) -> str:
    """Format an error for logging."""
    if error is None:
        return "null"
    text = str(error).strip().replace("\n", " ")
    return text if text else "null"


def _score_trajectory(task_name: str, trajectory: EvaluationTrajectory) -> float:
    """Score a trajectory using the task's grader."""
    return score_trajectory(task_name, trajectory)


def run_noop_task(task_alias: str, seed: int) -> Tuple[float, List[float]]:
    """
    Run a single task with the no-op agent.
    
    Args:
        task_alias: The task name (e.g., 'easy', 'medium', 'hard').
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (final_score, list_of_rewards).
    """
    bundle = resolve_task_bundle(task_alias)
    task = bundle.task
    core_env = task.build_core_env(seed)
    wrapper = OpenEnvWrapper(core_env=core_env)
    
    task.reset(wrapper.core_env, seed)
    observation = wrapper.state()
    
    trajectory = EvaluationTrajectory()
    trajectory.start(_normalize_dict(observation))
    
    print(f"[START] task={task_alias} env={ENV_NAME} model={MODEL_NAME}")
    
    score = 0.0
    success = False
    rewards: List[float] = []
    
    try:
        for step_index in range(1, task.max_steps + 1):
            action = Action(command="no_action")
            step_error: str | None = None
            
            try:
                next_observation, reward, done, info = wrapper.step(action)
            except Exception as exc:
                next_observation = observation
                reward = None
                done = True
                info = {"error": _format_error(exc)}
                step_error = _format_error(exc)
            
            reward_value = float(getattr(reward, "value", 0.0)) if reward is not None else 0.0
            rewards.append(reward_value)
            
            trajectory.append_step(
                action=action.command,
                reward=reward_value,
                done=bool(done),
                next_observation=_normalize_dict(next_observation),
                info=info,
            )
            
            print(
                f"[STEP] step={step_index} action={action.command} reward={reward_value:.2f} "
                f"done={str(bool(done)).lower()} error={_format_error(step_error)}"
            )
            
            observation = next_observation
            if done:
                break
        
        score = _score_trajectory(task_alias, trajectory)
        task_result = bundle.task.evaluate(trajectory.to_grader_records())
        success = bool(task_result.get("success", False)) and bool(task_result.get("constraints_ok", False))
    finally:
        close_fn = getattr(wrapper, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        rewards_str = ",".join(f"{value:.2f}" for value in rewards)
        print(f"[END] success={str(success).lower()} steps={len(rewards)} rewards={rewards_str}")
    
    return score, rewards


def main() -> int:
    """Main entry point for the no-op baseline agent."""
    parser = argparse.ArgumentParser(
        description="Run TITAN tasks with no-op baseline agent."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run a specific task (default: run all tasks)",
    )
    args = parser.parse_args()
    
    tasks = [args.task] if args.task else list(available_task_names())
    
    all_scores: Dict[str, float] = {}
    for task_alias in tasks:
        score, _ = run_noop_task(task_alias, seed=args.seed)
        all_scores[task_alias] = score
    
    print("\n--- NOOP Baseline Summary ---")
    for task, score in all_scores.items():
        print(f"{task}: {score:.4f}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

