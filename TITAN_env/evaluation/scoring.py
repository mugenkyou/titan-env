from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence

from titan_env.evaluation.trajectory import EvaluationTrajectory
from titan_env.tasks.registry import resolve_task_bundle


EPS = 1e-6


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
    score_value = bundle.grader(records)
    if score_value is None:
        score = EPS
    else:
        score = float(score_value)
        if math.isnan(score):
            score = EPS
    return float(max(EPS, min(1.0 - EPS, score)))


__all__ = ["score_trajectory"]


