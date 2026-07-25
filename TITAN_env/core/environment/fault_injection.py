"""
TITAN — Phase 4: Expanded Fault Injection System
fault_injection.py

Revises fault injection from 2→5 active fault types with
state-dependent probabilities driven by radiation intensity,
cpu_temperature, and power_temperature.

Fault types
-----------
  SEU                — Single Event Upset: memory integrity impulse drop
  LATCH_UP           — Radiation latchup: power surge
  THERMAL_RUNAWAY    — Thermal instability: escalating cpu_temperature
  MEMORY_CORRUPTION  — Severe bit corruption beyond standard SEU
  POWER_FAULT        — Voltage regulator instability
  TELEMETRY_CORRUPTION — Sensor noise only (obs-layer, no true-state effect)

State-dependent probability model
----------------------------------
  p_seu              = base_seu_rate    * radiation_intensity
  p_latchup          = base_latchup_rate * radiation_intensity * (1 + power_temperature)
  p_thermal_runaway  = base_thermal_rate * max(0, cpu_temperature - 0.70)
  p_memory_corrupt   = base_memory_rate  * radiation_intensity * (1 + seu_flag)
  p_power_fault      = base_power_rate   * (1 + power_temperature)

Multiple faults CAN fire in the same step (independent Bernoulli draws).
All effects are applied before StateTransition runs, so physics propagates
naturally from the perturbed state.

Step cycle (inside TITANEnv.step):
  1. FaultInjector.sample(state, step) ← THIS MODULE
  2. ActionProcessor.apply(...)
  3. StateTransition.step(...)
  4. check_failure(...)
  5. _get_observation(...)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .state_model import SubsystemState, _clamp


# ---------------------------------------------------------------------------
# Fault taxonomy
# ---------------------------------------------------------------------------

class FaultType(enum.Enum):
    SEU                  = "SEU"
    LATCH_UP             = "LATCH_UP"
    THERMAL_RUNAWAY      = "THERMAL_RUNAWAY"
    MEMORY_CORRUPTION    = "MEMORY_CORRUPTION"
    POWER_FAULT          = "POWER_FAULT"
    TELEMETRY_CORRUPTION = "TELEMETRY_CORRUPTION"


# ---------------------------------------------------------------------------
# Fault event record
# ---------------------------------------------------------------------------

@dataclass
class FaultEvent:
    """
    Immutable log record of a single fault occurrence.

    Attributes
    ----------
    fault_type        : FaultType
    step              : int       — simulation timestep (= fault_timestamp)
    subsystem         : str       — primary affected subsystem label
    magnitude         : float     — sampled intensity (0–1 normalised)
    fault_severity    : float     — alias for magnitude (dataset logging)
    fault_timestamp   : int       — alias for step (dataset logging)
    fault_subsystem   : str       — alias for subsystem (dataset logging)
    """

    fault_type:     FaultType
    step:           int
    subsystem:      str
    magnitude:      float

    @property
    def fault_severity(self) -> float:
        return self.magnitude

    @property
    def fault_timestamp(self) -> int:
        return self.step

    @property
    def fault_subsystem(self) -> str:
        return self.subsystem

    def as_dict(self) -> dict:
        return {
            "fault_type":       self.fault_type.value,
            "step":             self.step,
            "subsystem":        self.subsystem,
            "magnitude":        round(self.magnitude, 6),
            # Dataset logging fields
            "fault_severity":   round(self.magnitude, 6),
            "fault_timestamp":  self.step,
            "fault_subsystem":  self.subsystem,
        }


# ---------------------------------------------------------------------------
# Radiation profile
# ---------------------------------------------------------------------------

@dataclass
class RadiationProfile:
    """
    Per-step fault probabilities, magnitude ranges, and base rates for
    state-dependent probability computation.

    Parameters
    ----------
    radiation_intensity : float
        Scalar multiplier representing environment radiation level.
        1.0 = baseline (medium). Used in state-dependent probability formulas.

    p_seu, p_latchup, p_telemetry : float
        Legacy static per-step probabilities (fallback if base rates are 0).

    base_seu_rate, base_latchup_rate : float
        Base rates for SEU/latchup (scaled by radiation_intensity in sample()).

    base_thermal_rate : float
        Base rate for thermal runaway (scaled by cpu_temperature excess above 0.70).

    base_memory_rate : float
        Base rate for memory corruption (scaled by radiation_intensity and seu_flag).

    base_power_rate : float
        Base rate for power faults (scaled by power_temperature).

    thermal_runaway_rate : float
        Per-step cpu_temperature increase when THERMAL_RUNAWAY fires.

    memory_corruption_rate : float
        Per-step memory_integrity drop when MEMORY_CORRUPTION fires.

    power_fault_voltage_noise : float
        Max voltage noise swing when POWER_FAULT fires.

    power_fault_battery_drain : float
        battery_soc drain when POWER_FAULT fires.
    """

    # Radiation scalar
    radiation_intensity:       float = 1.0

    # Legacy static probs (used in get_telemetry_noise; SEU/latchup use base_rates)
    p_seu:               float = 0.05
    p_latchup:           float = 0.02
    p_telemetry:         float = 0.08

    # Magnitude ranges (existing fault types)
    seu_mag_range:       Tuple[float, float] = (0.02, 0.12)
    latchup_drain_range: Tuple[float, float] = (0.04, 0.12)
    latchup_heat_range:  Tuple[float, float] = (0.05, 0.15)
    telemetry_noise_max: float = 0.05

    # State-dependent base rates (new)
    base_seu_rate:      float = 0.06
    base_latchup_rate:  float = 0.03
    base_thermal_rate:  float = 0.20    # scales with cpu_temp excess above 0.70
    base_memory_rate:   float = 0.08    # scales with radiation + seu_flag
    base_power_rate:    float = 0.04    # scales with power_temperature

    # New fault effect magnitudes
    thermal_runaway_rate:       float = 0.05   # cpu_temperature per-step spike
    memory_corruption_rate:     float = 0.06   # memory_integrity per-step drop
    power_fault_voltage_noise:  float = 0.10   # voltage swing amplitude
    power_fault_battery_drain:  float = 0.03   # battery_soc drain

    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Preset intensity profiles — REBALANCED for 1000-step survival target
# ---------------------------------------------------------------------------

INTENSITY_PROFILES: Dict[str, RadiationProfile] = {
    "low": RadiationProfile(
        radiation_intensity=0.10,
        p_seu=0.001,       p_latchup=0.0005,  p_telemetry=0.002,
        seu_mag_range=(0.002, 0.006),
        latchup_drain_range=(0.002, 0.005),
        latchup_heat_range=(0.002, 0.006),
        telemetry_noise_max=0.005,
        base_seu_rate=0.002, base_latchup_rate=0.001,
        base_thermal_rate=0.005, base_memory_rate=0.002, base_power_rate=0.002,
        thermal_runaway_rate=0.003, memory_corruption_rate=0.003,
        power_fault_voltage_noise=0.006, power_fault_battery_drain=0.0005,
        seed=None,
    ),
    "medium": RadiationProfile(
        radiation_intensity=0.18,
        p_seu=0.003,       p_latchup=0.0015,  p_telemetry=0.005,
        seu_mag_range=(0.003, 0.009),
        latchup_drain_range=(0.003, 0.008),
        latchup_heat_range=(0.003, 0.01),
        telemetry_noise_max=0.008,
        base_seu_rate=0.004, base_latchup_rate=0.002,
        base_thermal_rate=0.012, base_memory_rate=0.004, base_power_rate=0.003,
        thermal_runaway_rate=0.005, memory_corruption_rate=0.005,
        power_fault_voltage_noise=0.01, power_fault_battery_drain=0.001,
        seed=None,
    ),
    "high": RadiationProfile(
        radiation_intensity=0.28,
        p_seu=0.005,       p_latchup=0.003,   p_telemetry=0.01,
        seu_mag_range=(0.005, 0.014),
        latchup_drain_range=(0.005, 0.012),
        latchup_heat_range=(0.005, 0.015),
        telemetry_noise_max=0.012,
        base_seu_rate=0.007, base_latchup_rate=0.004,
        base_thermal_rate=0.02, base_memory_rate=0.007, base_power_rate=0.005,
        thermal_runaway_rate=0.008, memory_corruption_rate=0.008,
        power_fault_voltage_noise=0.015, power_fault_battery_drain=0.002,
        seed=None,
    ),
    "storm": RadiationProfile(
        # Solar particle event — same as HIGH for baseline 1000 step target
        radiation_intensity=0.28,
        p_seu=0.005,       p_latchup=0.003,   p_telemetry=0.01,
        seu_mag_range=(0.005, 0.014),
        latchup_drain_range=(0.005, 0.012),
        latchup_heat_range=(0.005, 0.015),
        telemetry_noise_max=0.012,
        base_seu_rate=0.007, base_latchup_rate=0.004,
        base_thermal_rate=0.02, base_memory_rate=0.007, base_power_rate=0.005,
        thermal_runaway_rate=0.008, memory_corruption_rate=0.008,
        power_fault_voltage_noise=0.015, power_fault_battery_drain=0.002,
        seed=None,
    ),
}


# ---------------------------------------------------------------------------
# Fault injector
# ---------------------------------------------------------------------------

class FaultInjector:
    """
    Stochastic, seed-controlled fault injection engine (Phase 4 expanded).

    Usage
    -----
    injector = FaultInjector(INTENSITY_PROFILES["medium"])
    injector.reset()
    perturbed_state, events = injector.sample(state, step=t)
    noise = injector.get_telemetry_noise(step=t)

    Parameters
    ----------
    profile : RadiationProfile
    """

    def __init__(self, profile: RadiationProfile) -> None:
        self._profile = profile
        self._rng: np.random.Generator = self._make_rng()
        self._fault_log: List[FaultEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Re-seed the RNG and clear the fault log. Call at episode start."""
        self._rng = self._make_rng()
        self._fault_log = []

    def sample(
        self,
        state: SubsystemState,
        step:  int,
    ) -> Tuple[SubsystemState, Optional[FaultEvent]]:
        """
        Sample all fault events for this timestep using state-dependent
        probabilities and apply true-state mutations.

        Multiple faults may fire in the same step (independent Bernoulli draws).
        The PRIMARY fault event (highest-priority) is returned for the info dict.

        Priority: LATCH_UP > SEU > THERMAL_RUNAWAY > MEMORY_CORRUPTION > POWER_FAULT

        Parameters
        ----------
        state : SubsystemState  (copied, not mutated)
        step  : int

        Returns
        -------
        (perturbed_state, primary_fault_event | None)
        """
        p    = self._profile
        ri   = p.radiation_intensity
        t_cpu  = float(state.cpu_temperature)
        t_pwr  = float(state.power_temperature)
        s_flag = int(state.seu_flag)

        # State-dependent probabilities — take the MAX of the computed rate
        # and the legacy static probability, so:
        #   - Old tests that set p_seu=1.0 still guarantee fires (legacy path)
        #   - New state-driven experiments use the base_rate path
        prob_seu_dynamic     = _clamp(p.base_seu_rate     * ri,                           lo=0, hi=1)
        prob_latchup_dynamic = _clamp(p.base_latchup_rate * ri * (1.0 + t_pwr),           lo=0, hi=1)

        prob_seu     = max(prob_seu_dynamic,     p.p_seu)
        prob_latchup = max(prob_latchup_dynamic, p.p_latchup)
        prob_thermal = _clamp(p.base_thermal_rate * max(0.0, t_cpu - 0.70),               lo=0, hi=1)
        prob_mem_cor = _clamp(p.base_memory_rate  * ri * (1.0 + float(s_flag)),           lo=0, hi=1)
        prob_power   = _clamp(p.base_power_rate   * (1.0 + t_pwr),                       lo=0, hi=1)

        # Independent Bernoulli draws
        roll_seu     = self._rng.random() < prob_seu
        roll_latchup = self._rng.random() < prob_latchup
        roll_thermal = self._rng.random() < prob_thermal
        roll_mem_cor = self._rng.random() < prob_mem_cor
        roll_power   = self._rng.random() < prob_power

        # Copy state (preserve full Phase 2/3/5 fields)
        new_state = SubsystemState(
            battery_level=        state.battery_level,
            temperature=          state.temperature,
            cpu_health=           state.cpu_health,
            communication_health= state.communication_health,
            voltage=              state.voltage,
            current_draw=         state.current_draw,
            battery_soc=          state.battery_soc,
            cpu_temperature=      state.cpu_temperature,
            power_temperature=    state.power_temperature,
            memory_integrity=     state.memory_integrity,
            cpu_load=             state.cpu_load,
            seu_flag=             state.seu_flag,
            latchup_flag=         state.latchup_flag,
            thermal_fault_flag=   state.thermal_fault_flag,
            memory_fault_flag=    state.memory_fault_flag,
            power_fault_flag=     state.power_fault_flag,
            recent_fault_count=   state.recent_fault_count,
        )

        primary_event: Optional[FaultEvent] = None

        # Apply faults in priority order (all that rolled)
        if roll_latchup:
            new_state, event = self._apply_latchup(new_state, step)
            self._fault_log.append(event)
            if primary_event is None:
                primary_event = event

        if roll_seu:
            new_state, event = self._apply_seu(new_state, step)
            self._fault_log.append(event)
            if primary_event is None:
                primary_event = event

        if roll_thermal:
            new_state, event = self._apply_thermal_runaway(new_state, step)
            self._fault_log.append(event)
            if primary_event is None:
                primary_event = event

        if roll_mem_cor:
            new_state, event = self._apply_memory_corruption(new_state, step)
            self._fault_log.append(event)
            if primary_event is None:
                primary_event = event

        if roll_power:
            new_state, event = self._apply_power_fault(new_state, step)
            self._fault_log.append(event)
            if primary_event is None:
                primary_event = event

        return new_state, primary_event

    def get_telemetry_noise(self, step: int) -> Optional[Dict[str, float]]:
        """
        Sample sensor-level noise (TELEMETRY_CORRUPTION).
        True state is never modified here — obs-layer only.
        """
        if self._rng.random() >= self._profile.p_telemetry:
            return None

        noise_max = self._profile.telemetry_noise_max
        noise = {
            "battery_level":        float(self._rng.uniform(-noise_max, noise_max)),
            "temperature":          float(self._rng.uniform(-noise_max, noise_max)),
            "cpu_health":           float(self._rng.uniform(-noise_max, noise_max)),
            "communication_health": float(self._rng.uniform(-noise_max, noise_max)),
        }

        event = FaultEvent(
            fault_type=FaultType.TELEMETRY_CORRUPTION,
            step=step,
            subsystem="telemetry",
            magnitude=noise_max,
        )
        self._fault_log.append(event)
        return noise

    def inject_manual(
        self,
        fault_type: FaultType,
        state: SubsystemState,
        step: Optional[int] = None,
        magnitude: float = 1.0,
    ) -> Tuple[SubsystemState, FaultEvent]:
        """
        Manually inject a fault into the state (for user-triggered events).
        
        Parameters
        ----------
        fault_type : FaultType
            The type of fault to inject
        state : SubsystemState
            The current state to perturb
        step : int, optional
            Simulation step (defaults to current fault log length)
        magnitude : float, optional
            Per with the magnitude factor (0.0-1.0), affects fault severity
            
        Returns
        -------
        (new_state, fault_event) : Tuple[SubsystemState, FaultEvent]
        """
        if step is None:
            step = len(self._fault_log)
        
        # Copy state
        new_state = SubsystemState(
            battery_level=        state.battery_level,
            temperature=          state.temperature,
            cpu_health=           state.cpu_health,
            communication_health= state.communication_health,
            voltage=              state.voltage,
            current_draw=         state.current_draw,
            battery_soc=          state.battery_soc,
            cpu_temperature=      state.cpu_temperature,
            power_temperature=    state.power_temperature,
            memory_integrity=     state.memory_integrity,
            cpu_load=             state.cpu_load,
            seu_flag=             state.seu_flag,
            latchup_flag=         state.latchup_flag,
            thermal_fault_flag=   state.thermal_fault_flag,
            memory_fault_flag=    state.memory_fault_flag,
            power_fault_flag=     state.power_fault_flag,
            recent_fault_count=   state.recent_fault_count,
        )
        
        # Apply based on type
        if fault_type == FaultType.SEU:
            lo, hi = self._profile.seu_mag_range
            mag = float(self._rng.uniform(lo, hi)) * magnitude
            new_state.cpu_health = _clamp(new_state.cpu_health - mag)
            new_state.seu_flag = 1
            event = FaultEvent(
                fault_type=FaultType.SEU, step=step,
                subsystem="processing", magnitude=mag,
            )
        elif fault_type == FaultType.LATCH_UP:
            d_lo, d_hi = self._profile.latchup_drain_range
            h_lo, h_hi = self._profile.latchup_heat_range
            drain = float(self._rng.uniform(d_lo, d_hi)) * magnitude
            heat = float(self._rng.uniform(h_lo, h_hi)) * magnitude
            new_state.battery_level = _clamp(new_state.battery_level - drain)
            new_state.temperature = _clamp(new_state.temperature + heat)
            new_state.latchup_flag = 1
            event = FaultEvent(
                fault_type=FaultType.LATCH_UP, step=step,
                subsystem="power+thermal", magnitude=max(drain, heat),
            )
        elif fault_type == FaultType.THERMAL_RUNAWAY:
            rate = self._profile.thermal_runaway_rate * magnitude
            mag = float(self._rng.uniform(rate * 0.5, rate * 1.5))
            new_state.cpu_temperature = _clamp(new_state.cpu_temperature + mag)
            new_state.thermal_fault_flag = 1
            event = FaultEvent(
                fault_type=FaultType.THERMAL_RUNAWAY, step=step,
                subsystem="thermal", magnitude=mag,
            )
        elif fault_type == FaultType.MEMORY_CORRUPTION:
            rate = self._profile.memory_corruption_rate * magnitude
            mag = float(self._rng.uniform(rate * 0.5, rate * 1.5))
            new_state.memory_integrity = _clamp(new_state.memory_integrity - mag)
            new_state.memory_fault_flag = 1
            event = FaultEvent(
                fault_type=FaultType.MEMORY_CORRUPTION, step=step,
                subsystem="memory", magnitude=mag,
            )
        elif fault_type == FaultType.POWER_FAULT:
            vn = self._profile.power_fault_voltage_noise * magnitude
            drain = self._profile.power_fault_battery_drain * magnitude
            noise = float(self._rng.uniform(-vn, vn))
            new_state.voltage = _clamp(new_state.voltage + noise)
            new_state.battery_soc = _clamp(new_state.battery_soc - drain)
            new_state.power_fault_flag = 1
            event = FaultEvent(
                fault_type=FaultType.POWER_FAULT, step=step,
                subsystem="power", magnitude=abs(noise) + drain,
            )
        else:
            # Fallback: treat as SEU
            lo, hi = self._profile.seu_mag_range
            mag = float(self._rng.uniform(lo, hi)) * magnitude
            new_state.cpu_health = _clamp(new_state.cpu_health - mag)
            event = FaultEvent(
                fault_type=fault_type, step=step,
                subsystem="unknown", magnitude=mag,
            )
        
        self._fault_log.append(event)
        return new_state, event

    def compute_probabilities(self, state: SubsystemState) -> Dict[str, float]:
        """
        Return the current state-dependent fault probabilities (for logging/debug).
        """
        p   = self._profile
        ri  = p.radiation_intensity
        t_c = float(state.cpu_temperature)
        t_p = float(state.power_temperature)
        sf  = int(state.seu_flag)
        return {
            "p_seu":            _clamp(p.base_seu_rate    * ri,                         lo=0, hi=1),
            "p_latchup":        _clamp(p.base_latchup_rate * ri * (1.0 + t_p),          lo=0, hi=1),
            "p_thermal_runaway":_clamp(p.base_thermal_rate * max(0.0, t_c - 0.70),      lo=0, hi=1),
            "p_memory_corrupt": _clamp(p.base_memory_rate  * ri * (1.0 + float(sf)),    lo=0, hi=1),
            "p_power_fault":    _clamp(p.base_power_rate   * (1.0 + t_p),              lo=0, hi=1),
        }

    @property
    def fault_log(self) -> List[FaultEvent]:
        """All fault events recorded this episode (read-only copy)."""
        return list(self._fault_log)

    @property
    def profile(self) -> RadiationProfile:
        """The active radiation profile."""
        return self._profile

    # ------------------------------------------------------------------
    # Private fault mechanics
    # ------------------------------------------------------------------

    def _apply_seu(
        self,
        state: SubsystemState,
        step:  int,
    ) -> Tuple[SubsystemState, FaultEvent]:
        """
        Single Event Upset: impulse drop to cpu_health.
        Sets seu_flag=1 so state_model physics adds memory damage next step.
        """
        lo, hi    = self._profile.seu_mag_range
        magnitude = float(self._rng.uniform(lo, hi))
        state.cpu_health = _clamp(state.cpu_health - magnitude)
        state.seu_flag   = 1
        event = FaultEvent(
            fault_type=FaultType.SEU, step=step,
            subsystem="processing", magnitude=magnitude,
        )
        return state, event

    def _apply_latchup(
        self,
        state: SubsystemState,
        step:  int,
    ) -> Tuple[SubsystemState, FaultEvent]:
        """
        Latch-up: power surge + thermal spike.
        Sets latchup_flag=1 so state_model physics surges current_draw.
        """
        d_lo, d_hi = self._profile.latchup_drain_range
        h_lo, h_hi = self._profile.latchup_heat_range
        extra_drain = float(self._rng.uniform(d_lo, d_hi))
        heat_spike  = float(self._rng.uniform(h_lo, h_hi))
        state.battery_level  = _clamp(state.battery_level - extra_drain)
        state.temperature    = _clamp(state.temperature   + heat_spike)
        state.latchup_flag   = 1
        magnitude = max(extra_drain, heat_spike)
        event = FaultEvent(
            fault_type=FaultType.LATCH_UP, step=step,
            subsystem="power+thermal", magnitude=magnitude,
        )
        return state, event

    def _apply_thermal_runaway(
        self,
        state: SubsystemState,
        step:  int,
    ) -> Tuple[SubsystemState, FaultEvent]:
        """
        Thermal runaway: direct cpu_temperature spike.
        Sets thermal_fault_flag=1 for physics propagation.
        Effect: cpu_temperature += runaway_rate (absorbed by thermal equation).
        """
        rate      = self._profile.thermal_runaway_rate
        magnitude = float(self._rng.uniform(rate * 0.5, rate * 1.5))
        state.cpu_temperature    = _clamp(state.cpu_temperature    + magnitude)
        state.thermal_fault_flag = 1
        event = FaultEvent(
            fault_type=FaultType.THERMAL_RUNAWAY, step=step,
            subsystem="thermal", magnitude=magnitude,
        )
        return state, event

    def _apply_memory_corruption(
        self,
        state: SubsystemState,
        step:  int,
    ) -> Tuple[SubsystemState, FaultEvent]:
        """
        Severe memory corruption: direct memory_integrity drop.
        Sets memory_fault_flag=1 for physics propagation.
        Effect: memory_integrity -= corruption_rate (more severe than SEU).
        """
        rate      = self._profile.memory_corruption_rate
        magnitude = float(self._rng.uniform(rate * 0.5, rate * 1.5))
        state.memory_integrity = _clamp(state.memory_integrity - magnitude)
        state.memory_fault_flag = 1
        event = FaultEvent(
            fault_type=FaultType.MEMORY_CORRUPTION, step=step,
            subsystem="memory", magnitude=magnitude,
        )
        return state, event

    def _apply_power_fault(
        self,
        state: SubsystemState,
        step:  int,
    ) -> Tuple[SubsystemState, FaultEvent]:
        """
        Voltage regulator fault: voltage fluctuation + battery drain.
        Sets power_fault_flag=1 for physics propagation.
        Effect: voltage += noise, battery_soc -= drain.
        """
        vn    = self._profile.power_fault_voltage_noise
        drain = self._profile.power_fault_battery_drain
        noise = float(self._rng.uniform(-vn, vn))
        state.voltage      = _clamp(state.voltage      + noise)
        state.battery_soc  = _clamp(state.battery_soc  - drain)
        state.power_fault_flag = 1
        magnitude = abs(noise) + drain
        event = FaultEvent(
            fault_type=FaultType.POWER_FAULT, step=step,
            subsystem="power", magnitude=magnitude,
        )
        return state, event

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_rng(self) -> np.random.Generator:
        return np.random.default_rng(self._profile.seed)
