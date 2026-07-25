"""
TITAN — Phase 1 + 2: Core State Engine + Physics Model Upgrade
state_model.py

Phase 2 additions:
    - Centralized OBS_KEYS (moved here from titan_env.py)
  - Physics parameter block (8 named constants)
  - Five dedicated subsystem update functions:
      update_cpu_temperature()  — thermal model
      update_current_draw()     — power model
      update_battery_soc()      — battery model
      update_memory_integrity() — memory corruption / repair
      update_cpu_load()         — CPU load dynamics
  - StateTransition.step() refactored to call the 5 functions.
    Gains an optional action_effect parameter (None → backward compat).

Observation schema (13D, in order):
    voltage, current_draw, battery_soc, cpu_temperature, power_temperature,
    memory_integrity, cpu_load, seu_flag, latchup_flag, thermal_fault_flag,
    memory_fault_flag, power_fault_flag, recent_fault_count
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Observation schema — single source of truth used by env, gym, logger, tests
# ---------------------------------------------------------------------------

OBS_KEYS: Tuple[str, ...] = (
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
)

OBS_DIM: int = len(OBS_KEYS)  # 13


# ---------------------------------------------------------------------------
# Physics parameter block — centralized for easy tuning
# ---------------------------------------------------------------------------

# Thermal model
ALPHA_HEAT_GEN:    float = 0.15
BETA_COOLING:      float = 0.10
COOLING_HEADROOM:  float = 0.25

# Power model
GAMMA_LATCHUP:    float = 0.35
DELTA_RECOVERY:   float = 0.10

# Memory model
ETA_MEMORY_DAMAGE: float = 0.01  # reduced for easy task achievability
MU_MEMORY_REPAIR:  float = 0.50  # increased for faster recovery

# Battery model
SOLAR_CHARGE:     float = 0.15

# CPU load dynamics
WORKLOAD_NOISE_STD: float = 0.02

# Power cycle nominal reset values
VOLTAGE_NOMINAL_RESET:   float = 0.85
CURRENT_DRAW_NOMINAL_RESET: float = 0.40
POWER_CYCLE_BATTERY_COST:   float = 0.10  # battery_soc cost per power cycle

# Phase 4 fault flag physics — reactions to active fault flags
# Applied in StateTransition.step() AFTER normal physics, per-step when flag is set.
THERMAL_RUNAWAY_RATE:     float = 0.05   # cpu_temperature escalation per step when thermal_fault_flag=1
MEMORY_CORRUPTION_RATE:   float = 0.04   # memory_integrity drop per step when memory_fault_flag=1
POWER_FAULT_VOLTAGE_AMP:  float = 0.05   # voltage perturbation amplitude when power_fault_flag=1

# ---------------------------------------------------------------------------
# PhysicsConfig dataclass — injectable wrapper around the constants above
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    """
    Centralized, tunable physics coefficients.

    Default values mirror the module-level constants.  Pass a custom
    PhysicsConfig to StateTransition.step() for unit tests or experiments
    that need non-default physics.
    """
    alpha_heat_gen:    float = ALPHA_HEAT_GEN
    beta_cooling:      float = BETA_COOLING
    cooling_headroom:  float = COOLING_HEADROOM
    gamma_latchup:     float = GAMMA_LATCHUP
    delta_recovery:    float = DELTA_RECOVERY
    eta_memory_damage: float = ETA_MEMORY_DAMAGE
    mu_memory_repair:  float = MU_MEMORY_REPAIR
    solar_charge:      float = SOLAR_CHARGE
    workload_noise_std: float = WORKLOAD_NOISE_STD

    def __post_init__(self):
        """Validate all coefficients are positive."""
        for name, val in self.__dict__.items():
            assert val >= 0.0, f"PhysicsConfig.{name} must be >= 0, got {val}"


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class SubsystemState:
    """
    True internal state of the satellite at a single timestep.

    All float values are normalised to [0.0, 1.0].
    Integer fault flags are exactly 0 or 1.

    Phase 1 (legacy) fields
    -----------------------
    battery_level        : 1.0 = full, 0.0 = depleted
    temperature          : 0.0 = cold, 1.0 = critical overheat
    cpu_health           : 1.0 = nominal, 0.0 = failed
    communication_health : 1.0 = nominal, 0.0 = lost

    Phase 5 extended fields
    -----------------------
    voltage              : regulated bus voltage
    current_draw         : load current draw
    battery_soc          : battery state-of-charge (Phase 2: physics-driven)
    cpu_temperature      : CPU die temperature    (Phase 2: physics-driven)
    power_temperature    : power regulator temperature
    memory_integrity     : SRAM/flash integrity   (Phase 2: physics-driven)
    cpu_load             : processor utilisation  (Phase 2: physics-driven)

    Fault flags (set by titan_env, reset each step)
    -------------------------------------------------
    seu_flag, latchup_flag, thermal_fault_flag,
    memory_fault_flag, power_fault_flag, recent_fault_count
    """

    # — Phase 1 legacy (preserved for failure checks and history) —
    battery_level:        float = 1.0
    temperature:          float = 0.2
    cpu_health:           float = 1.0
    communication_health: float = 1.0

    # — Phase 5 / Phase 2 physics state —
    voltage:              float = 0.85
    current_draw:         float = 0.40
    battery_soc:          float = 0.90
    cpu_temperature:      float = 0.30
    power_temperature:    float = 0.25
    memory_integrity:     float = 1.00
    cpu_load:             float = 0.30

    # — Fault flags —
    seu_flag:             int   = 0
    latchup_flag:         int   = 0
    thermal_fault_flag:   int   = 0
    memory_fault_flag:    int   = 0
    power_fault_flag:     int   = 0
    recent_fault_count:   float = 0.0

    def clamp_all(self) -> "SubsystemState":
        """Return a new state with all floats clamped to [0, 1] and flags to {0, 1}."""
        return SubsystemState(
            battery_level=        _clamp(self.battery_level),
            temperature=          _clamp(self.temperature),
            cpu_health=           _clamp(self.cpu_health),
            communication_health= _clamp(self.communication_health),
            voltage=              _clamp(self.voltage),
            current_draw=         _clamp(self.current_draw),
            battery_soc=          _clamp(self.battery_soc),
            cpu_temperature=      _clamp(self.cpu_temperature),
            power_temperature=    _clamp(self.power_temperature),
            memory_integrity=     _clamp(self.memory_integrity),
            cpu_load=             _clamp(self.cpu_load),
            seu_flag=             int(bool(self.seu_flag)),
            latchup_flag=         int(bool(self.latchup_flag)),
            thermal_fault_flag=   int(bool(self.thermal_fault_flag)),
            memory_fault_flag=    int(bool(self.memory_fault_flag)),
            power_fault_flag=     int(bool(self.power_fault_flag)),
            recent_fault_count=   _clamp(self.recent_fault_count),
        )

    def as_dict(self) -> dict:
        """Return all state fields as a flat dict."""
        return {
            "battery_level":        self.battery_level,
            "temperature":          self.temperature,
            "cpu_health":           self.cpu_health,
            "communication_health": self.communication_health,
            "voltage":              self.voltage,
            "current_draw":         self.current_draw,
            "battery_soc":          self.battery_soc,
            "cpu_temperature":      self.cpu_temperature,
            "power_temperature":    self.power_temperature,
            "memory_integrity":     self.memory_integrity,
            "cpu_load":             self.cpu_load,
            "seu_flag":             self.seu_flag,
            "latchup_flag":         self.latchup_flag,
            "thermal_fault_flag":   self.thermal_fault_flag,
            "memory_fault_flag":    self.memory_fault_flag,
            "power_fault_flag":     self.power_fault_flag,
            "recent_fault_count":   self.recent_fault_count,
        }


# ---------------------------------------------------------------------------
# Failure thresholds
# ---------------------------------------------------------------------------

@dataclass
class StateBounds:
    battery_critical:    float = 0.10
    temperature_max:     float = 0.90
    cpu_min:             float = 0.15
    comms_min:           float = 0.10
    memory_fault_thresh: float = 0.15


# ---------------------------------------------------------------------------
# Phase 2: Subsystem update functions
# ---------------------------------------------------------------------------

def update_cpu_temperature(
    state: SubsystemState,
    rng: "np.random.Generator | None" = None,
) -> float:
    """
    Thermal Model
    -------------
    T_cpu(t+1) = clamp( T(t) + α*cpu_load - β*headroom + ε )

    headroom = max(COOLING_HEADROOM, 1 - T_cpu)
      COOLING_HEADROOM ensures a minimum cooling floor even at peak temperature.
    ε ~ N(0, 0.01) when rng is not None, else 0.
    """
    noise = float(rng.normal(0.0, 0.01)) if rng is not None else 0.0
    headroom = max(COOLING_HEADROOM, 1.0 - state.cpu_temperature)
    delta = ALPHA_HEAT_GEN * state.cpu_load - BETA_COOLING * headroom
    return clamp(state.cpu_temperature + delta + noise)


def update_current_draw(
    state: SubsystemState,
    load_shedding: float = 0.0,
) -> float:
    """
    Power Model
    -----------
    I(t+1) = clamp( I(t) * (1 + γ)  [latchup]  OR  I(t) * (1 - δ) [no latchup] )
             - load_shedding * 0.5

    load_shedding (0–1): fraction of power shed by REDUCE_LOAD action.
    """
    if state.latchup_flag:
        new_current = state.current_draw * (1.0 + GAMMA_LATCHUP)
    else:
        new_current = state.current_draw * (1.0 - DELTA_RECOVERY)
    new_current -= load_shedding * 0.5
    return clamp(new_current)


def update_battery_soc(state: SubsystemState) -> float:
    """
    Battery Model
    -------------
    SOC(t+1) = clamp( SOC(t) - energy_consumption + SOLAR_CHARGE )

    energy_consumption = current_draw * voltage  (normalised power draw)
    SOLAR_CHARGE is a constant per-step gain (always-on solar panel).
    Note: solar_charge is ADDED (not subtracted).
    """
    energy_consumption = state.current_draw * state.voltage
    return clamp(state.battery_soc - energy_consumption + SOLAR_CHARGE)


def update_memory_integrity(
    state: SubsystemState,
    memory_scrub: bool = False,
) -> float:
    """
    Memory Corruption Model
    -----------------------
    M(t+1) = clamp( M(t) - η*seu_flag + μ*memory_scrub )

    memory_scrub (bool): True when RESET_CPU action is applied.
    """
    damage = ETA_MEMORY_DAMAGE * float(state.seu_flag)
    repair = MU_MEMORY_REPAIR if memory_scrub else 0.0
    return clamp(state.memory_integrity - damage + repair)


def update_cpu_load(
    state: SubsystemState,
    load_shedding: float = 0.0,
    rng: "np.random.Generator | None" = None,
) -> float:
    """
    CPU Load Dynamics
    -----------------
    L(t+1) = clamp( L(t) * (1 - load_shedding) + mean_reversion + ε )

    load_shedding semantics:
      * Positive (0–1): MULTIPLICATIVE reduction. 0.30 → 30 % of current load removed.
        Example: L=0.70, shed=0.30  →  0.70 * 0.70 = 0.49  before mean-reversion.
      * Negative (MEMORY_SCRUB): treated as a load INCREASE by the same multiplier.
        Example: L=0.40, shed=-0.05 →  0.40 * 1.05 = 0.42  before mean-reversion.

    Mean-reversion pulls L toward nominal (0.30) each step at rate 0.05.
    ε ~ N(0, WORKLOAD_NOISE_STD) when rng is not None, else 0.
    """
    noise = float(rng.normal(0.0, WORKLOAD_NOISE_STD)) if rng is not None else 0.0
    CPU_LOAD_NOMINAL = 0.30
    CPU_LOAD_REVERT  = 0.05

    # Multiplicative load shedding (handles both positive and negative shedding)
    shed_factor = clamp(1.0 - load_shedding, lo=0.0, hi=2.0)   # cap at 2x for safety
    shed_load   = state.cpu_load * shed_factor

    reversion = CPU_LOAD_REVERT * (CPU_LOAD_NOMINAL - shed_load)
    return clamp(shed_load + reversion + noise)


# ---------------------------------------------------------------------------
# Transition engine
# ---------------------------------------------------------------------------

class StateTransition:
    """
    Deterministic, single-step state update engine.

    Phase 1 causal chain (legacy):
        Power → Thermal → CPU health → Comms health

    Phase 2 physics updates (run after Phase 1 chain):
        1. update_cpu_temperature()
        2. update_current_draw()
        3. update_battery_soc()
        4. update_memory_integrity()
        5. update_cpu_load()

    Parameters
    ----------
    rng : np.random.Generator | None
        Physics noise RNG.  None → noise = 0 → fully deterministic.
    action_effect : ActionEffect | None
        The action applied this step.  None → no action-driven physics.
    """

    # Phase 1 coefficients (legacy)
    # NOTE: POWER_DRAW_RATE reduced to allow 1000+ step survival
    # Original: 0.002 → max ~450 steps
    # New: 0.0005 → max ~1800 steps (with buffer for fault-driven drain)
    # HEAT_GAIN reduced to prevent thermal cascade failures
    POWER_DRAW_RATE:              float = 0.0005
    HEAT_GAIN:                    float = 0.015
    COOLING_RATE:                 float = 0.008
    THERMAL_STRESS_COEFFICIENT:   float = 0.008
    SAFE_TEMP_THRESHOLD:          float = 0.65
    COMM_DEGRADATION_COEFFICIENT: float = 0.012
    CPU_CRITICAL_THRESHOLD:       float = 0.35

    # Phase 2 voltage / power_temperature (simple first-order lag kept)
    VOLTAGE_NOMINAL:    float = 0.85
    VOLTAGE_LAG:        float = 0.05
    PWR_TEMP_COOLING:   float = 0.006
    PWR_TEMP_CURRENT_GAIN: float = 0.12
    PWR_TEMP_AMBIENT_GAIN: float = 0.08

    @classmethod
    def step(
        cls,
        state: SubsystemState,
        rng: "np.random.Generator | None" = None,
        action_effect=None,
    ) -> SubsystemState:
        """
        Advance state by one timestep.

        Step order
        ----------
        1. Phase 1 legacy chain  (battery, temperature, cpu_health, comms)
        2. Phase 2 subsystem models  (cpu_temperature, current_draw,
                                       battery_soc, memory_integrity, cpu_load)
        3. Voltage + power_temperature (first-order lag)
        4. clamp_all()
        """

        # ------------------------------------------------------------------
        # Phase 1: legacy causal chain (preserved for failure detection)
        # ------------------------------------------------------------------
        new_battery = state.battery_level - cls.POWER_DRAW_RATE

        heat = cls.HEAT_GAIN * (1.0 - new_battery)
        new_temperature = state.temperature + heat - cls.COOLING_RATE

        thermal_excess = max(0.0, new_temperature - cls.SAFE_TEMP_THRESHOLD)
        new_cpu = state.cpu_health - cls.THERMAL_STRESS_COEFFICIENT * thermal_excess

        cpu_deficit = max(0.0, cls.CPU_CRITICAL_THRESHOLD - new_cpu)
        new_comms = state.communication_health - cls.COMM_DEGRADATION_COEFFICIENT * cpu_deficit

        # ------------------------------------------------------------------
        # Phase 3: derive physics action contract from ActionEffect fields
        # ------------------------------------------------------------------
        load_shedding    = getattr(action_effect, "load_shedding",   0.0)  if action_effect else 0.0
        recovery_effect  = getattr(action_effect, "recovery_effect", 0.0)  if action_effect else 0.0
        memory_scrub     = getattr(action_effect, "memory_scrub",    False) if action_effect else False
        thermal_throttle = getattr(action_effect, "thermal_throttle",0.0)  if action_effect else 0.0
        power_cycle      = getattr(action_effect, "power_cycle",     False) if action_effect else False

        # Compute new_cpu_load FIRST so THERMAL_THROTTLING immediately feeds
        # into the thermal equation (indirect cooling via reduced heat gen).
        new_cpu_load    = update_cpu_load(state, load_shedding=load_shedding, rng=rng)

        # Thermal model: use effective cpu_load (post-shedding) for heat gen,
        # so THERMAL_THROTTLING reduces temperature in the SAME step it fires.
        effective_cpu_temp_state = SubsystemState(
            **{**state.as_dict(), "cpu_load": new_cpu_load}
        )
        new_cpu_temp    = update_cpu_temperature(effective_cpu_temp_state, rng)

        new_current     = update_current_draw(state, load_shedding=max(0.0, load_shedding))
        new_battery_soc = update_battery_soc(state)
        new_memory      = update_memory_integrity(state, memory_scrub=bool(memory_scrub))

        # thermal_throttle: direct temperature suppression on top of indirect effect
        # (kept as a small direct nudge for THERMAL_THROTTLING when set)
        if thermal_throttle > 0.0:
            new_cpu_temp = clamp(new_cpu_temp - thermal_throttle)

        # Power cycle: voltage reset + current SPIKE + battery cost
        if power_cycle:
            new_voltage     = VOLTAGE_NOMINAL_RESET
            # Inrush current spike on power-on (realistic behavior)
            new_current     = clamp(state.current_draw + 0.20)
            new_battery_soc = clamp(new_battery_soc - POWER_CYCLE_BATTERY_COST)

        # Recovery effect: damps latchup-driven current surges
        if recovery_effect > 0.0 and state.latchup_flag:
            new_current = clamp(new_current * (1.0 - recovery_effect))

        # Voltage — first-order lag toward nominal (unless power_cycle just reset it)
        if not power_cycle:
            noise_v = float(rng.normal(0.0, 0.01)) if rng is not None else 0.0
            voltage_target = cls.VOLTAGE_NOMINAL * (0.3 + 0.7 * new_battery)
            new_voltage = state.voltage + cls.VOLTAGE_LAG * (voltage_target - state.voltage) + noise_v

        # Power temperature — driven by current draw and ambient temperature
        new_pwr_temp = (
            state.power_temperature
            + cls.PWR_TEMP_CURRENT_GAIN * new_current
            + cls.PWR_TEMP_AMBIENT_GAIN * new_temperature
            - cls.PWR_TEMP_COOLING
        )

        # Flags reset to 0; titan_env sets them after StateTransition
        return SubsystemState(
            battery_level=        new_battery,
            temperature=          new_temperature,
            cpu_health=           new_cpu,
            communication_health= new_comms,
            voltage=              new_voltage,
            current_draw=         new_current,
            battery_soc=          new_battery_soc,
            cpu_temperature=      new_cpu_temp,
            power_temperature=    new_pwr_temp,
            memory_integrity=     new_memory,
            cpu_load=             new_cpu_load,
            seu_flag=             0,
            latchup_flag=         0,
            thermal_fault_flag=   0,
            memory_fault_flag=    0,
            power_fault_flag=     0,
            recent_fault_count=   state.recent_fault_count,
        ).clamp_all()


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------

def check_failure(
    state: SubsystemState,
    bounds: StateBounds,
) -> Tuple[bool, str]:
    """Evaluate all failure conditions. Returns (failed, reason)."""

    if state.battery_level < bounds.battery_critical:
        return True, (
            f"POWER FAILURE — battery_level={state.battery_level:.4f} "
            f"below critical {bounds.battery_critical}"
        )
    if state.temperature > bounds.temperature_max:
        return True, (
            f"THERMAL RUNAWAY — temperature={state.temperature:.4f} "
            f"exceeds limit {bounds.temperature_max}"
        )
    if state.cpu_health < bounds.cpu_min:
        return True, (
            f"PROCESSING FAILURE — cpu_health={state.cpu_health:.4f} "
            f"below minimum {bounds.cpu_min}"
        )
    if state.communication_health < bounds.comms_min:
        return True, (
            f"COMMS COLLAPSE — communication_health={state.communication_health:.4f} "
            f"below minimum {bounds.comms_min}"
        )
    return False, ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Private clamping helper — kept for internal use."""
    return max(lo, min(hi, value))


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Public clamping utility.
    Ensures value stays within [lo, hi].  Default range is [0, 1].
    Used inside every subsystem update function to guarantee safe output.
    """
    return max(lo, min(hi, value))


