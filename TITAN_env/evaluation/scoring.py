from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from TITAN_env.evaluation.trajectory import EvaluationTrajectory
from TITAN_env.tasks.registry import resolve_task_bundle


def _trajectory_to_records(trajectory: Any) -> List[Dict[str, Any]]:
    if isinstance(trajectory, EvaluationTrajectory):
        return trajectory.to_grader_records()
    if isinstance(trajectory, list):
        return trajectory
    raise TypeError("trajectory must be an EvaluationTrajectory or list of records")


def score_trajectory(task_name: str, trajectory: Any) -> float:
    """Score a trajectory using the grader associated with a registered task."""
    bundle = resolve_task_bundle(task_name)
    records = _trajectory_to_records(trajectory)
    score = float(bundle.grader(records))
    return max(0.0, min(1.0, score))


__all__ = ["score_trajectory"]
