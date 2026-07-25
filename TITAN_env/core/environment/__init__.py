from .state_model import SubsystemState, StateBounds, StateTransition
from .titan_env import TITANEnv
from .fault_injection import (
    FaultType,
    FaultEvent,
    RadiationProfile,
    FaultInjector,
    INTENSITY_PROFILES,
)
from .actions import ActionType, ActionProcessor, ActionEffect

# Phase 4: Gymnasium wrapper — optional, gracefully absent if gymnasium not installed
try:
    from .gym_env import TITANGymEnv
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

__all__ = [
    # Phase 1
    "SubsystemState",
    "StateBounds",
    "StateTransition",
    "TITANEnv",
    # Phase 2
    "FaultType",
    "FaultEvent",
    "RadiationProfile",
    "FaultInjector",
    "INTENSITY_PROFILES",
    # Phase 3
    "ActionType",
    "ActionProcessor",
    "ActionEffect",
    # Phase 4
    "TITANGymEnv",
]


