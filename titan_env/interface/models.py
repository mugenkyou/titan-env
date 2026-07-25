from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Structured OpenEnv observation mapped from TITAN telemetry."""

    voltage: float = Field(ge=0.0, le=1.0)
    current_draw: float = Field(ge=0.0, le=1.0)
    battery: float = Field(ge=0.0, le=1.0)
    cpu_temp: float = Field(ge=0.0, le=1.0)
    power_temp: float = Field(ge=0.0, le=1.0)
    memory: float = Field(ge=0.0, le=1.0)
    cpu_load: float = Field(ge=0.0, le=1.0)
    signal: float = Field(ge=0.0, le=1.0)
    recent_fault_count: float = Field(ge=0.0, le=1.0)
    faults: List[str] = Field(default_factory=list)


class Action(BaseModel):
    """OpenEnv action model."""

    command: str = Field(min_length=1)


class Reward(BaseModel):
    """OpenEnv reward model."""

    value: float
