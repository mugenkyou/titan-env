"""
TITAN — Phase 3: 7-Action Control System
actions.py

Expands action space from 4 → 7 discrete actions.  Each action now carries
a richer ActionEffect that maps directly onto the Phase 2 physics engine
(load_shedding, memory_scrub, thermal_throttle, power_cycle, recovery_effect).

Backward-compat aliases keep all Phase 1–3 tests green without editing them:
    DO_NOTHING    = NO_ACTION
    RESET_CPU     = SUBSYSTEM_RESET
    REDUCE_LOAD   = LOAD_SHEDDING
    COOLING_MODE  = THERMAL_THROTTLING

Action integer mapping (Discrete(8)):
    0  NO_ACTION         — no intervention
    1  SUBSYSTEM_RESET   — partial reboot; restores CPU and comms
    2  MEMORY_SCRUB      — ECC scrub; repairs memory integrity, small CPU cost
    3  LOAD_SHEDDING     — shed 30 % of compute; lowers thermal + current draw
    4  POWER_CYCLE       — hard power cycle; resets voltage/current, battery cost
    5  THERMAL_THROTTLING — aggressive cooling + load shed; highest thermal relief
    6  ISOLATE_SUBSYSTEM — quarantine faulty subsystem; large recovery, load cut
    7  DIAGNOSE          — passive diagnostic action (no direct state mutation)

Step cycle position:
    1. FaultInjector.sample(state, t)       [Phase 2]
    2. ActionProcessor.apply(action, state) [Phase 3 — THIS MODULE]
    3. StateTransition.step(state)          [Phase 1 + 2 physics]
    4. check_failure(...)
    5. _get_observation()
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Tuple

from .state_model import SubsystemState, _clamp


__all__ = [
    "ActionType",
    "ActionEffect",
    "ActionProcessor",
    "ACTION_COSTS",
]


# ---------------------------------------------------------------------------
# Action cost table — canonical source of truth
# ---------------------------------------------------------------------------

ACTION_COSTS: Dict[int, float] = {}   # populated after ActionType is defined
# (see bottom of module — populated once ActionType enum exists)


# ---------------------------------------------------------------------------
# Action taxonomy — 7 actions + backward-compat aliases
# ---------------------------------------------------------------------------

class ActionType(enum.IntEnum):
    """
    Discrete recovery actions available to a controller.

    IntEnum allows actions to be passed as integers (Gymnasium Discrete space).
    Aliases preserve backward compatibility with tests written for the
    4-action Phase 1 spec.
    """
    NO_ACTION          = 0
    SUBSYSTEM_RESET    = 1
    MEMORY_SCRUB       = 2
    LOAD_SHEDDING      = 3
    POWER_CYCLE        = 4
    THERMAL_THROTTLING = 5
    ISOLATE_SUBSYSTEM  = 6
    DIAGNOSE           = 7

    # -------- backward-compat aliases --------
    DO_NOTHING  = 0   # alias for NO_ACTION
    RESET_CPU   = 1   # alias for SUBSYSTEM_RESET
    REDUCE_LOAD = 3   # alias for LOAD_SHEDDING
    COOLING_MODE = 5  # alias for THERMAL_THROTTLING


# ---------------------------------------------------------------------------
# Action effect record — full physics contract
# ---------------------------------------------------------------------------

@dataclass
class ActionEffect:
    """
    Logged record of what an action applied to the satellite state.

    Legacy fields (delta_battery, delta_temperature, delta_cpu, delta_comms)
    are retained for Phase 1 causal chain compatibility.

    Phase 3 physics fields are consumed by StateTransition.step():
        load_shedding    (float 0–1) : fraction of cpu_load / current shed
        recovery_effect  (float 0–1) : latchup recovery rate boost
        memory_scrub     (bool)      : True triggers memory integrity repair
        thermal_throttle (float 0–1) : direct cpu_temperature reduction
        power_cycle      (bool)      : True resets voltage/current to nominal
    """

    action_type:       ActionType
    step:              int

    # Phase 1 legacy deltas (applied to true state by ActionProcessor)
    delta_battery:     float = 0.0
    delta_temperature: float = 0.0
    delta_cpu:         float = 0.0
    delta_comms:       float = 0.0

    # Phase 3 physics contract (consumed by StateTransition.step)
    load_shedding:     float = 0.0
    recovery_effect:   float = 0.0
    memory_scrub:      bool  = False
    thermal_throttle:  float = 0.0
    power_cycle:       bool  = False

    def as_dict(self) -> dict:
        return {
            "action":            self.action_type.name,
            "step":              self.step,
            "delta_battery":     round(self.delta_battery,    6),
            "delta_temperature": round(self.delta_temperature, 6),
            "delta_cpu":         round(self.delta_cpu,         6),
            "delta_comms":       round(self.delta_comms,       6),
            # Phase 3 physics fields
            "load_shedding":     self.load_shedding,
            "recovery_effect":   self.recovery_effect,
            "memory_scrub":      self.memory_scrub,
            "thermal_throttle":  self.thermal_throttle,
            "power_cycle":       self.power_cycle,
        }


# ---------------------------------------------------------------------------
# Action coefficients — ALL tuning in one place
# ---------------------------------------------------------------------------

class _Coefficients:
    """
    Named constants for every action delta.
    Centralised here so tuning never touches application logic.

    Trade-off summary
    -----------------
    NO_ACTION          : No effect. Passive degradation continues.

    SUBSYSTEM_RESET    : Strong CPU + comms restoration.
                         Temporary comms penalty, battery spike.
                         Best when CPU is critically low.

    MEMORY_SCRUB       : ECC pass repairs memory integrity.
                         Small CPU load penalty during scrub.
                         Best after SEU events.

    LOAD_SHEDDING      : Sheds 30 % of compute.
                         Reduces heat generation and current draw.
                         Best when temperature is climbing.

    POWER_CYCLE        : Hard power cycle. Resets voltage + current.
                         Significant battery cost per invocation.
                         Best after latch-up accumulation.

    THERMAL_THROTTLING : Combines load shedding + aggressive cooling.
                         Highest thermal relief; moderate battery impact.
                         Best when cpu_temperature is near-critical.

    ISOLATE_SUBSYSTEM  : Quarantines a faulty subsystem.
                         Large recovery_effect to counter latchup.
                         Cuts 25 % of cpu_load as isolation overhead.
    """

    # NO_ACTION (no physical effect via ActionProcessor)

    # SUBSYSTEM_RESET (legacy: was RESET_CPU)
    SR_BATTERY_COST:   float = -0.015
    SR_TEMP_DELTA:     float =  0.000
    SR_CPU_RECOVERY:   float = +0.200
    SR_COMMS_PENALTY:  float = -0.100
    # Phase 3 physics
    SR_LOAD_SHEDDING:  float =  0.10
    SR_RECOVERY:       float =  0.20

    # MEMORY_SCRUB
    MS_BATTERY_COST:   float = -0.005
    MS_TEMP_DELTA:     float =  0.000
    MS_CPU_DELTA:      float =  0.000   # CPU load penalty applied via physics contract
    MS_COMMS_DELTA:    float =  0.000
    # Phase 3 physics
    MS_CPU_LOAD_COST:  float =  0.05   # scrub increases cpu_load briefly

    # LOAD_SHEDDING (legacy: was REDUCE_LOAD)
    LS_BATTERY_COST:   float = -0.005
    LS_TEMP_REDUCTION: float = -0.080
    LS_CPU_RECOVERY:   float = +0.040
    LS_COMMS_DELTA:    float =  0.000
    # Phase 3 physics
    LS_LOAD_SHEDDING:  float =  0.30

    # POWER_CYCLE
    PC_BATTERY_COST:   float = -0.100   # hard reboot draws significant power
    PC_TEMP_DELTA:     float =  0.000
    PC_CPU_DELTA:      float =  0.000
    PC_COMMS_PENALTY:  float = -0.050   # brief link drop during cycle

    # THERMAL_THROTTLING (legacy: was COOLING_MODE)
    # Temperature reduction is now INDIRECT: load shedding reduces cpu_load,
    # which in turn reduces heat generation in the thermal equation.
    # No direct thermal_throttle applied — physics handles it.
    TT_BATTERY_COST:   float = -0.010
    TT_TEMP_REDUCTION: float = -0.120   # applied to legacy temperature field
    TT_CPU_DELTA:      float =  0.000
    TT_COMMS_DELTA:    float =  0.000
    # Phase 3 physics
    TT_LOAD_SHEDDING:    float = 0.30
    TT_THERMAL_THROTTLE: float = 0.0    # 0 — indirect cooling via cpu_load only

    # ISOLATE_SUBSYSTEM
    IS_BATTERY_COST:   float = -0.008
    IS_TEMP_DELTA:     float =  0.000
    IS_CPU_DELTA:      float =  0.000
    IS_COMMS_PENALTY:  float = -0.100   # bus isolation disrupts comms link
    # Phase 3 physics
    IS_LOAD_SHEDDING:  float =  0.30   # multiplicative: 0.30 → cpu_load *= 0.70
    IS_RECOVERY:       float =  0.40


# ---------------------------------------------------------------------------
# Action processor — stateless, pure
# ---------------------------------------------------------------------------

class ActionProcessor:
    """
    Stateless, deterministic action application engine.

    Usage
    -----
    new_state, effect = ActionProcessor.apply(action, state, step=t)

    The returned state has all Phase 1 variables clamped to [0, 1].
    Phase 3 physics fields on effect are consumed by StateTransition.step().
    """

    @classmethod
    def apply(
        cls,
        action: ActionType | int | None,
        state:  SubsystemState,
        step:   int = 0,
    ) -> Tuple[SubsystemState, ActionEffect]:
        """
        Apply a recovery action to the satellite state.

        Parameters
        ----------
        action : ActionType | int | None
            The action to apply. None → NO_ACTION.
        state  : SubsystemState
            Current true state (copied, not mutated).
        step   : int
            Current timestep (logged in ActionEffect).
        """
        if action is None:
            action = ActionType.NO_ACTION
        if isinstance(action, int):
            action = ActionType(action)

        dispatch = {
            ActionType.NO_ACTION:          cls._no_action,
            ActionType.SUBSYSTEM_RESET:    cls._subsystem_reset,
            ActionType.MEMORY_SCRUB:       cls._memory_scrub,
            ActionType.LOAD_SHEDDING:      cls._load_shedding,
            ActionType.POWER_CYCLE:        cls._power_cycle,
            ActionType.THERMAL_THROTTLING: cls._thermal_throttling,
            ActionType.ISOLATE_SUBSYSTEM:  cls._isolate_subsystem,
            ActionType.DIAGNOSE:           cls._diagnose,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"Unknown action: {action!r}")
        return handler(state, step)

    # ------------------------------------------------------------------
    # 0. NO_ACTION
    # ------------------------------------------------------------------

    @classmethod
    def _no_action(cls, state: SubsystemState, step: int) -> Tuple[SubsystemState, ActionEffect]:
        """No intervention. State unchanged."""
        new_state = SubsystemState(
            battery_level=        state.battery_level,
            temperature=          state.temperature,
            cpu_health=           state.cpu_health,
            communication_health= state.communication_health,
        )
        effect = ActionEffect(
            action_type=ActionType.NO_ACTION, step=step,
        )
        return new_state, effect

    # ------------------------------------------------------------------
    # 1. SUBSYSTEM_RESET  (was RESET_CPU)
    # ------------------------------------------------------------------

    @classmethod
    def _subsystem_reset(cls, state: SubsystemState, step: int) -> Tuple[SubsystemState, ActionEffect]:
        """
        Partial reboot: strong CPU recovery, brief comms penalty, battery spike.
        Best when CPU health is critically low.
        """
        c = _Coefficients
        new_state = SubsystemState(
            battery_level=        _clamp(state.battery_level        + c.SR_BATTERY_COST),
            temperature=          _clamp(state.temperature          + c.SR_TEMP_DELTA),
            cpu_health=           _clamp(state.cpu_health           + c.SR_CPU_RECOVERY),
            communication_health= _clamp(state.communication_health + c.SR_COMMS_PENALTY),
        )
        effect = ActionEffect(
            action_type=ActionType.SUBSYSTEM_RESET, step=step,
            delta_battery=c.SR_BATTERY_COST, delta_temperature=c.SR_TEMP_DELTA,
            delta_cpu=c.SR_CPU_RECOVERY,     delta_comms=c.SR_COMMS_PENALTY,
            load_shedding=c.SR_LOAD_SHEDDING, recovery_effect=c.SR_RECOVERY,
        )
        return new_state, effect

    # ------------------------------------------------------------------
    # 2. MEMORY_SCRUB
    # ------------------------------------------------------------------

    @classmethod
    def _memory_scrub(cls, state: SubsystemState, step: int) -> Tuple[SubsystemState, ActionEffect]:
        """
        ECC memory scrub: repairs memory_integrity (via physics contract).
        Small battery cost; increases cpu_load briefly.
        Best after SEU events when memory integrity is degraded.
        """
        c = _Coefficients
        new_state = SubsystemState(
            battery_level=        _clamp(state.battery_level        + c.MS_BATTERY_COST),
            temperature=          _clamp(state.temperature          + c.MS_TEMP_DELTA),
            cpu_health=           _clamp(state.cpu_health           + c.MS_CPU_DELTA),
            communication_health= _clamp(state.communication_health + c.MS_COMMS_DELTA),
        )
        effect = ActionEffect(
            action_type=ActionType.MEMORY_SCRUB, step=step,
            delta_battery=c.MS_BATTERY_COST, delta_temperature=c.MS_TEMP_DELTA,
            delta_cpu=c.MS_CPU_DELTA,         delta_comms=c.MS_COMMS_DELTA,
            memory_scrub=True,
            # cpu_load increase applied by StateTransition via physics contract
            load_shedding= -c.MS_CPU_LOAD_COST,  # negative = increase
        )
        return new_state, effect

    # ------------------------------------------------------------------
    # 3. LOAD_SHEDDING  (was REDUCE_LOAD)
    # ------------------------------------------------------------------

    @classmethod
    def _load_shedding(cls, state: SubsystemState, step: int) -> Tuple[SubsystemState, ActionEffect]:
        """
        Shed 30 % of compute: lowers temperature and current draw.
        Best when temperature is climbing but CPU is still healthy.
        """
        c = _Coefficients
        new_state = SubsystemState(
            battery_level=        _clamp(state.battery_level        + c.LS_BATTERY_COST),
            temperature=          _clamp(state.temperature          + c.LS_TEMP_REDUCTION),
            cpu_health=           _clamp(state.cpu_health           + c.LS_CPU_RECOVERY),
            communication_health= _clamp(state.communication_health + c.LS_COMMS_DELTA),
        )
        effect = ActionEffect(
            action_type=ActionType.LOAD_SHEDDING, step=step,
            delta_battery=c.LS_BATTERY_COST, delta_temperature=c.LS_TEMP_REDUCTION,
            delta_cpu=c.LS_CPU_RECOVERY,      delta_comms=c.LS_COMMS_DELTA,
            load_shedding=c.LS_LOAD_SHEDDING,
        )
        return new_state, effect

    # ------------------------------------------------------------------
    # 4. POWER_CYCLE
    # ------------------------------------------------------------------

    @classmethod
    def _power_cycle(cls, state: SubsystemState, step: int) -> Tuple[SubsystemState, ActionEffect]:
        """
        Hard power cycle: resets voltage and current_draw to nominal.
        High battery cost. Comms briefly disrupted.
        Best after latch-up accumulation or persistent over-current.
        """
        c = _Coefficients
        new_state = SubsystemState(
            battery_level=        _clamp(state.battery_level        + c.PC_BATTERY_COST),
            temperature=          _clamp(state.temperature          + c.PC_TEMP_DELTA),
            cpu_health=           _clamp(state.cpu_health           + c.PC_CPU_DELTA),
            communication_health= _clamp(state.communication_health + c.PC_COMMS_PENALTY),
        )
        effect = ActionEffect(
            action_type=ActionType.POWER_CYCLE, step=step,
            delta_battery=c.PC_BATTERY_COST, delta_temperature=c.PC_TEMP_DELTA,
            delta_cpu=c.PC_CPU_DELTA,          delta_comms=c.PC_COMMS_PENALTY,
            power_cycle=True,
        )
        return new_state, effect

    # ------------------------------------------------------------------
    # 5. THERMAL_THROTTLING  (was COOLING_MODE)
    # ------------------------------------------------------------------

    @classmethod
    def _thermal_throttling(cls, state: SubsystemState, step: int) -> Tuple[SubsystemState, ActionEffect]:
        """
        Thermal throttling: sheds 30 % of compute load.
        Temperature reduction is INDIRECT — lower cpu_load reduces heat generation
        in the thermal equation next step (no direct temperature delta applied).
        This is physically more realistic than instant temperature cuts.
        Best when cpu_temperature is near-critical.
        """
        c = _Coefficients
        new_state = SubsystemState(
            battery_level=        _clamp(state.battery_level        + c.TT_BATTERY_COST),
            temperature=          _clamp(state.temperature          + c.TT_TEMP_REDUCTION),
            cpu_health=           _clamp(state.cpu_health           + c.TT_CPU_DELTA),
            communication_health= _clamp(state.communication_health + c.TT_COMMS_DELTA),
        )
        effect = ActionEffect(
            action_type=ActionType.THERMAL_THROTTLING, step=step,
            delta_battery=c.TT_BATTERY_COST, delta_temperature=c.TT_TEMP_REDUCTION,
            delta_cpu=c.TT_CPU_DELTA,          delta_comms=c.TT_COMMS_DELTA,
            load_shedding=c.TT_LOAD_SHEDDING,
            thermal_throttle=c.TT_THERMAL_THROTTLE,   # 0.0 — indirect only
        )
        return new_state, effect

    # ------------------------------------------------------------------
    # 6. ISOLATE_SUBSYSTEM
    # ------------------------------------------------------------------

    @classmethod
    def _isolate_subsystem(cls, state: SubsystemState, step: int) -> Tuple[SubsystemState, ActionEffect]:
        """
        Quarantine a faulty subsystem from the bus.
        - cpu_load *= (1 - 0.30) = 0.70 via multiplicative load shedding.
        - communication_health -= IS_COMMS_PENALTY (bus isolation → comms disruption).
        - Large recovery_effect counters latch-up current surges.
        - Small battery cost from isolation relay switching.
        """
        c = _Coefficients
        new_state = SubsystemState(
            battery_level=        _clamp(state.battery_level        + c.IS_BATTERY_COST),
            temperature=          _clamp(state.temperature          + c.IS_TEMP_DELTA),
            cpu_health=           _clamp(state.cpu_health           + c.IS_CPU_DELTA),
            communication_health= _clamp(state.communication_health + c.IS_COMMS_PENALTY),
        )
        effect = ActionEffect(
            action_type=ActionType.ISOLATE_SUBSYSTEM, step=step,
            delta_battery=c.IS_BATTERY_COST, delta_temperature=c.IS_TEMP_DELTA,
            delta_cpu=c.IS_CPU_DELTA,          delta_comms=c.IS_COMMS_PENALTY,
            load_shedding=c.IS_LOAD_SHEDDING,
            recovery_effect=c.IS_RECOVERY,
        )
        return new_state, effect

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @classmethod
    def _diagnose(cls, state: SubsystemState, step: int) -> Tuple[SubsystemState, ActionEffect]:
        """Diagnostic probe action. No direct intervention on plant state."""
        new_state = SubsystemState(
            battery_level=        state.battery_level,
            temperature=          state.temperature,
            cpu_health=           state.cpu_health,
            communication_health= state.communication_health,
        )
        effect = ActionEffect(
            action_type=ActionType.DIAGNOSE, step=step,
        )
        return new_state, effect

    @staticmethod
    def action_space_size() -> int:
        """Number of discrete actions (matches gym Discrete(8))."""
        # Count only canonical values (not aliases) — IntEnum dedups them
        return len(set(ActionType))


# ---------------------------------------------------------------------------
# ACTION_COSTS — populated here so ActionType enum is fully resolved
# ---------------------------------------------------------------------------

ACTION_COSTS = {
    int(ActionType.NO_ACTION):          0.0,
    int(ActionType.SUBSYSTEM_RESET):    0.5,
    int(ActionType.MEMORY_SCRUB):       0.4,
    int(ActionType.LOAD_SHEDDING):      0.3,
    int(ActionType.POWER_CYCLE):        0.8,
    int(ActionType.THERMAL_THROTTLING): 0.2,
    int(ActionType.ISOLATE_SUBSYSTEM):  1.0,
    int(ActionType.DIAGNOSE):           0.1,
}
"""
Canonical action cost table — single source of truth.

Costs are subtracted from the survival bonus (+1.0) each step.
Higher-cost actions must deliver proportionally more benefit to be worth using.

    NO_ACTION          0.0  — free; passive baseline
    SUBSYSTEM_RESET    0.5  — moderate; boot cost
    MEMORY_SCRUB       0.4  — light; ECC scan overhead
    LOAD_SHEDDING      0.3  — very light; just compute reduction
    POWER_CYCLE        0.8  — expensive; inrush + reboot overhead
    THERMAL_THROTTLING 0.2  — cheapest active action
    ISOLATE_SUBSYSTEM  1.0  — most expensive; bus reconfiguration cost
"""
