from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence

from titan_env.evaluation.trajectory import EvaluationTrajectory
from titan_env.tasks.registry import resolve_task_bundle


def safe_score(score: float) -> float:
    return min(max(round(float(score), 4), 0.01), 0.99)


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
        score = safe_score(0.0)
    else:
        score = float(score_value)
        if math.isnan(score):
            score = safe_score(0.0)
    score = safe_score(score)
    assert 0.0 < score < 1.0, f"Invalid score: {score}"
    return score


__all__ = ["score_trajectory"]


