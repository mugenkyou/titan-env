from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from TITAN_env.graders.grader_easy import grade_easy
from TITAN_env.graders.grader_hard import grade_hard
from TITAN_env.graders.grader_medium import grade_medium
from TITAN_env.tasks.base_task import BaseTask
from TITAN_env.tasks.task_easy import EASY_TASK
from TITAN_env.tasks.task_hard import HARD_TASK
from TITAN_env.tasks.task_medium import MEDIUM_TASK


@dataclass(frozen=True)
class TaskBundle:
    task: BaseTask
    grader: Callable


_TASK_BUNDLES: Dict[str, TaskBundle] = {
    "easy": TaskBundle(task=EASY_TASK, grader=grade_easy),
    "easy_single_fault_recovery": TaskBundle(task=EASY_TASK, grader=grade_easy),
    "medium": TaskBundle(task=MEDIUM_TASK, grader=grade_medium),
    "medium_thermal_stabilization": TaskBundle(task=MEDIUM_TASK, grader=grade_medium),
    "hard": TaskBundle(task=HARD_TASK, grader=grade_hard),
    "hard_multi_fault_survival": TaskBundle(task=HARD_TASK, grader=grade_hard),
}


def available_task_names() -> Tuple[str, ...]:
    """Return the supported task names in deterministic order."""
    return tuple(("easy", "medium", "hard"))


def resolve_task_bundle(name: str) -> TaskBundle:
    """Resolve a task name to its task/grader pair."""
    key = name.strip().lower()
    if key not in _TASK_BUNDLES:
        raise KeyError(f"Unknown TITAN task: {name}")
    return _TASK_BUNDLES[key]


def get_task(name: str) -> BaseTask:
    return resolve_task_bundle(name).task


def get_grader(name: str):
    return resolve_task_bundle(name).grader


def list_registered_tasks() -> Tuple[str, ...]:
    """Return canonical long-form task names for display or configuration."""
    return (
        EASY_TASK.name,
        MEDIUM_TASK.name,
        HARD_TASK.name,
    )
