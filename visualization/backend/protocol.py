"""
WebSocket protocol definitions for TITAN visualization
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import json


@dataclass
class TelemetryMessage:
    """Telemetry data sent from backend to frontend."""
    step: int
    episode: int
    telemetry: Dict[str, float]
    faults: Dict[str, bool]
    action: Optional[int] = None
    action_name: Optional[str] = None
    reason: Optional[str] = None
    reward: float = 0.0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TelemetryMessage":
        return cls(**data)


@dataclass  
class CommandMessage:
    """Command sent from frontend to backend."""
    action: str  # "reset", "step", "pause", "resume", "inject_fault", "set_speed"
    action_id: Optional[int] = None
    fault_type: Optional[str] = None
    speed: Optional[float] = None
    profile: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps({k: v for k, v in self.__dict__.items() if v is not None})
    
    @classmethod
    def from_json(cls, data: str) -> "CommandMessage":
        d = json.loads(data)
        return cls(**d)


# Observation key mapping (index -> name)
OBS_KEYS = [
    "voltage",
    "current_draw", 
    "battery_soc",
    "cpu_temperature",
    "power_temperature",
    "memory_integrity",
    "cpu_load",
    "seu_flag",
    "latchup_flag",
    "thermal_fault_flag",
    "memory_fault_flag",
    "power_fault_flag",
    "recent_fault_count",
]

# Action mapping (id -> name)
ACTION_NAMES = {
    0: "NO_ACTION",
    1: "SUBSYSTEM_RESET",
    2: "MEMORY_SCRUB",
    3: "LOAD_SHEDDING",
    4: "POWER_CYCLE",
    5: "THERMAL_THROTTLING",
    6: "ISOLATE_SUBSYSTEM",
}
