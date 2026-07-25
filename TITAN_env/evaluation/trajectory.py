from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


StateDict = Dict[str, Any]


@dataclass
class EvaluationTrajectory:
    """Deterministic trajectory container used by the inference pipeline."""

    observations: List[StateDict] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    next_observations: List[StateDict] = field(default_factory=list)

    def start(self, observation: StateDict) -> None:
        if self.observations:
            raise ValueError("trajectory already started")
        self.observations.append(dict(observation))

    def append_step(
        self,
        action: str,
        reward: float,
        done: bool,
        next_observation: StateDict,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.observations:
            raise ValueError("trajectory must be started before appending steps")

        self.actions.append(str(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.infos.append(dict(info or {}))
        self.next_observations.append(dict(next_observation))
        self.observations.append(dict(next_observation))

    @property
    def step_count(self) -> int:
        return len(self.actions)

    def to_grader_records(self) -> List[Dict[str, Any]]:
        """Convert the trajectory into grader-compatible transition records."""
        if not self.observations:
            return []

        records: List[Dict[str, Any]] = [
            {
                "step": 0,
                "state": dict(self.observations[0]),
                "action": None,
                "reward": 0.0,
                "done": False,
                "info": {"phase": "initial"},
            }
        ]

        for index, action in enumerate(self.actions, start=1):
            records.append(
                {
                    "step": index,
                    "state": dict(self.next_observations[index - 1]),
                    "action": action,
                    "reward": float(self.rewards[index - 1]),
                    "done": bool(self.dones[index - 1]),
                    "info": dict(self.infos[index - 1]),
                }
            )

        return records


__all__ = ["EvaluationTrajectory"]
