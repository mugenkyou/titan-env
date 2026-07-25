"""
TITAN — Phase 1 + 2 + 3 + 5: Core State Engine + Fault Injection + Action System
titan_env.py

Phase 5 additions:
  - StateTransition.step() receives the environment RNG for reproducible noise
  - Circular buffer (deque, maxlen=10) tracks the last 10 fault events
  - recent_fault_count = len(non-None entries in buffer) / 10
  - Fault flags (seu_flag, latchup_flag, thermal_fault_flag,
    memory_fault_flag, power_fault_flag) are set on the true state after
    StateTransition.step() using fault context from this step
  - _get_observation() returns an ordered 13-key dict matching OBS_KEYS
  - The 13 observation channels (in order):
      voltage, current_draw, battery_soc, cpu_temperature, power_temperature,
      memory_integrity, cpu_load, seu_flag, latchup_flag, thermal_fault_flag,
      memory_fault_flag, power_fault_flag, recent_fault_count
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Tuple

import numpy as np

from .state_model import (
    SubsystemState,
    StateBounds,
    StateTransition,
    check_failure,
    _clamp,
    OBS_KEYS,
    OBS_DIM,
)
from .fault_injection import FaultInjector, FaultEvent, FaultType
from .actions import ActionType, ActionProcessor, ActionEffect

# Re-export for any code that imported OBS_KEYS from titan_env directly
    # Re-export for any code that imported OBS_KEYS from TITAN directly
__all__ = ["TITANEnv", "OBS_KEYS", "OBS_DIM"]


class TITANEnv:
    """
    Discrete-time satellite subsystem simulation environment.

    Phase 1 + 2 + 3 + 5 capabilities
    -----------------------------------
    - Deterministic causal state evolution (Phase 1)
    - Pluggable stochastic fault injection (Phase 2)
    - Telemetry noise (Phase 2, observation-only)
    - Discrete recovery actions (Phase 3)
    - Extended 13D observation space with fault flags and fault history (Phase 5)
    - Full history logging with fault and action metadata
    - Gym-compatible reset / step interface

    Parameters
    ----------
    initial_state : SubsystemState, optional
    bounds : StateBounds, optional
    max_steps : int
        Hard upper limit on episode length (default 5000).
    fault_injector : FaultInjector | None
        Phase 2 fault injection engine. None → deterministic behaviour.
    seed : int | None
        Seed for the physics noise RNG. None → non-reproducible.
    """

    def __init__(
        self,
        initial_state: Optional[SubsystemState] = None,
        bounds: Optional[StateBounds] = None,
        max_steps: int = 5000,
        fault_injector: Optional[FaultInjector] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._initial_state: SubsystemState = initial_state or SubsystemState()
        self._sbounds: StateBounds = bounds or StateBounds()
        self._max_steps: int = max_steps
        self._injector: Optional[FaultInjector] = fault_injector

        # Physics noise RNG (Phase 5)
        # Use seed+1 so the physics RNG is independent from the fault injector RNG
        # (both would otherwise start with the same seed). seed=None → deterministic Phase 1.
        self._seed = seed
        physics_seed = (seed + 1) if seed is not None else None
        self._rng: np.random.Generator = np.random.default_rng(physics_seed)

        # Runtime state
        self._state: SubsystemState = SubsystemState()
        self._step_count: int = 0
        self._done: bool = False
        self._history: List[dict] = []

        # Phase 5: circular fault buffer for recent_fault_count
        self._fault_buffer: collections.deque = collections.deque(maxlen=10)

        # Episode metrics — reset each episode
        self._faults_triggered: int   = 0
        self._actions_taken:    int   = 0
        self._max_temperature:  float = 0.0
        self._min_battery_soc:  float = 1.0

        self.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """
        Reset to initial conditions.
        Re-seeds fault injector RNG and physics RNG for reproducibility.
        """
        self._state = SubsystemState(
            battery_level=        self._initial_state.battery_level,
            temperature=          self._initial_state.temperature,
            cpu_health=           self._initial_state.cpu_health,
            communication_health= self._initial_state.communication_health,
            voltage=              self._initial_state.voltage,
            current_draw=         self._initial_state.current_draw,
            battery_soc=          self._initial_state.battery_soc,
            cpu_temperature=      self._initial_state.cpu_temperature,
            power_temperature=    self._initial_state.power_temperature,
            memory_integrity=     self._initial_state.memory_integrity,
            cpu_load=             self._initial_state.cpu_load,
            seu_flag=             0,
            latchup_flag=         0,
            thermal_fault_flag=   0,
            memory_fault_flag=    0,
            power_fault_flag=     0,
            recent_fault_count=   0.0,
        )
        self._step_count = 0
        self._done = False
        self._history = []
        self._fault_buffer.clear()

        # Reset episode metrics
        self._faults_triggered = 0
        self._actions_taken    = 0
        self._max_temperature  = self._initial_state.temperature
        self._min_battery_soc  = self._initial_state.battery_soc

        # Re-seed physics RNG
        self._rng = np.random.default_rng(self._seed)

        if self._injector is not None:
            self._injector.reset()

        obs = self._get_observation(self._state, fault_event=None)
        self._log_step(obs, done=False, failure_reason="",
                       fault_event=None, action_effect=None)
        return obs

    def step(self, action=None) -> Tuple[dict, bool, dict]:
        """
        Advance the simulation by one discrete timestep.

        Step sequence (Phase 1 + 2 + 3 + 5)
        --------------------------------------
        1. Fault injection    — radiation perturbs true state   [Phase 2]
        2. Action application — controller intervenes           [Phase 3]
        3. Causal transition  — physics evolves                 [Phase 1+5]
        4. Flag computation   — fault/threshold flags set       [Phase 5]
        5. Failure check                                        [Phase 1]
        6. Observation + telemetry noise                        [Phase 2]

        Parameters
        ----------
        action : ActionType | int | None
            Recovery action to apply. None or 0 → DO_NOTHING.

        Returns
        -------
        observation : dict  (13 keys in OBS_KEYS order)
        done : bool
        info : dict
        """
        if self._done:
            raise RuntimeError(
                "Episode has ended. Call reset() before stepping again."
            )

        self._step_count += 1

        # --- 1. Fault injection [Phase 2] ---
        fault_event: Optional[FaultEvent] = None
        working_state = self._state

        if self._injector is not None:
            working_state, fault_event = self._injector.sample(
                self._state, self._step_count
            )

        # Update fault circular buffer
        self._fault_buffer.append(fault_event)

        # --- 2. Action application [Phase 3] ---
        if action is None:
            resolved_action = ActionType.DO_NOTHING
        elif isinstance(action, ActionType):
            resolved_action = action
        else:
            resolved_action = ActionType(int(action))

        working_state, action_effect = ActionProcessor.apply(
            resolved_action, working_state, self._step_count
        )

        # --- 3. Causal + physics transition [Phase 1 + 2 + 5] ---
        # Only apply Gaussian noise when env was constructed with an explicit seed.
        # rng=None → StateTransition uses noise()=0 → fully deterministic (Phase 1 compat).
        physics_rng = self._rng if self._seed is not None else None
        self._state = StateTransition.step(
            working_state,
            rng=physics_rng,
            action_effect=action_effect,   # Phase 2: power + memory models use this
        )

        # --- 4. Flag computation [Phase 4 + 5] ---
        # a) Radiation fault flags — fault injection can set these directly;
        #    we also OR with threshold-based checks so both paths set the flag.
        seu_flag     = 1 if (fault_event is not None and fault_event.fault_type == FaultType.SEU)          else 0
        latchup_flag = 1 if (fault_event is not None and fault_event.fault_type == FaultType.LATCH_UP)      else 0

        # b) New Phase 4 fault types from injection
        thermal_inj = 1 if (fault_event is not None and fault_event.fault_type == FaultType.THERMAL_RUNAWAY)    else 0
        memory_inj  = 1 if (fault_event is not None and fault_event.fault_type == FaultType.MEMORY_CORRUPTION)  else 0
        power_inj   = 1 if (fault_event is not None and fault_event.fault_type == FaultType.POWER_FAULT)         else 0

        # c) Threshold-based flags (OR with injection)
        thermal_fault = max(thermal_inj, 1 if self._state.temperature      > self._sbounds.temperature_max    else 0)
        memory_fault  = max(memory_inj,  1 if self._state.memory_integrity < self._sbounds.memory_fault_thresh else 0)
        power_fault   = max(power_inj,   1 if self._state.battery_level    < self._sbounds.battery_critical   else 0)

        # d) Recent fault count
        recent_faults = sum(1 for e in self._fault_buffer if e is not None)
        recent_fault_count = recent_faults / 10.0

        # Inject flags onto the state
        self._state.seu_flag           = seu_flag
        self._state.latchup_flag       = latchup_flag
        self._state.thermal_fault_flag = thermal_fault
        self._state.memory_fault_flag  = memory_fault
        self._state.power_fault_flag   = power_fault
        self._state.recent_fault_count = recent_fault_count

        # --- 5. Failure check ---
        failed, reason = check_failure(self._state, self._sbounds)
        timeout = self._step_count >= self._max_steps
        done = failed or timeout

        if timeout and not failed:
            reason = f"MAX_STEPS reached ({self._max_steps})"

        self._done = done

        # --- 5b. Episode metrics update ---
        if fault_event is not None:
            self._faults_triggered += 1
        if resolved_action != ActionType.NO_ACTION:
            self._actions_taken += 1
        self._max_temperature = max(self._max_temperature, self._state.temperature)
        self._min_battery_soc = min(self._min_battery_soc, self._state.battery_soc)

        # --- 6. Observation [Phase 2 + 5] ---
        obs = self._get_observation(self._state, fault_event=fault_event)

        self._log_step(obs, done=done, failure_reason=reason,
                       fault_event=fault_event, action_effect=action_effect)

        info = {
            "step":            self._step_count,
            "failure_reason":  reason,
            "done":            done,
            "fault_event":     fault_event,
            "fault_type":      fault_event.fault_type.name if fault_event else None,
            "fault_subsystem": fault_event.subsystem if fault_event else None,
            "fault_severity_level": fault_event.severity_level if fault_event else 0,
            "action":          resolved_action.name,
            # Reward components (for RL debugging)
            "reward_survival":        1.0 if not done or not failed else 0.0,
            "reward_failure_penalty": -50.0 if failed else 0.0,
            # Episode metrics
            "episode_faults_triggered": self._faults_triggered,
            "episode_actions_taken":    self._actions_taken,
            "episode_max_temperature":  round(self._max_temperature, 4),
            "episode_min_battery_soc":  round(self._min_battery_soc, 4),
            "episode_steps_survived":   self._step_count,
        }
        return obs, done, info

    def run(
        self,
        max_steps: int = 1000,
        policy=None,
    ) -> List[dict]:
        """
        Convenience runner: execute until done or max_steps.

        Parameters
        ----------
        max_steps : int
        policy : callable | None
            Optional policy function: `policy(obs) -> ActionType`.
            None → DO_NOTHING every step.

        Returns
        -------
        List[dict]
            Full episode history.
        """
        original_max = self._max_steps
        self._max_steps = max_steps
        obs = self.reset()

        try:
            while not self._done:
                action = policy(obs) if policy is not None else ActionType.DO_NOTHING
                obs, _, _ = self.step(action)
        finally:
            self._max_steps = original_max

        return list(self._history)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> SubsystemState:
        return self._state

    @property
    def history(self) -> List[dict]:
        return list(self._history)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def done(self) -> bool:
        return self._done

    @property
    def fault_log(self) -> List[dict]:
        if self._injector is None:
            return []
        return [e.as_dict() for e in self._injector.fault_log]

    @property
    def action_space_size(self) -> int:
        """Number of available discrete actions."""
        return ActionProcessor.action_space_size()

    @property
    def obs_dim(self) -> int:
        """Observation vector dimension (13)."""
        return OBS_DIM

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_observation(
        self,
        state: SubsystemState,
        fault_event: Optional[FaultEvent] = None,
    ) -> dict:
        """
        Generate observable telemetry.

        Returns an ordered dict with:
        - 13 new Phase 5 channels (matching OBS_KEYS) for the gym array
        - Legacy Phase 1 key aliases for backward compatibility with existing code/tests.
          (battery_level, temperature, cpu_health, communication_health)

        Applies telemetry noise to physical channels only.
        Fault flags and recent_fault_count are never noised.
        True state is never modified here.
        """
        obs = {
            # Phase 5 channels (13D spec)
            "voltage":            state.voltage,
            "current_draw":       state.current_draw,
            "battery_soc":        state.battery_soc,
            "cpu_temperature":    state.cpu_temperature,
            "power_temperature":  state.power_temperature,
            "memory_integrity":   state.memory_integrity,
            "cpu_load":           state.cpu_load,
            "seu_flag":           float(state.seu_flag),
            "latchup_flag":       float(state.latchup_flag),
            "thermal_fault_flag": float(state.thermal_fault_flag),
            "memory_fault_flag":  float(state.memory_fault_flag),
            "power_fault_flag":   float(state.power_fault_flag),
            "recent_fault_count": state.recent_fault_count,
            # Per-fault severity levels (not part of the gym array)
            "seu_severity":       0.0,
            "latchup_severity":   0.0,
            "thermal_severity":   0.0,
            "memory_severity":    0.0,
            "power_severity":     0.0,
            "severity_level_total": 0.0,
            # Legacy Phase 1 aliases (backward compat — not in gym obs array)
            "battery_level":        state.battery_level,
            "temperature":          state.temperature,
            "cpu_health":           state.cpu_health,
            "communication_health": state.communication_health,
        }
        if self._injector is not None:
            active_severity = self._injector.active_fault_severity
            obs["seu_severity"] = float(active_severity.get(FaultType.SEU, 0))
            obs["latchup_severity"] = float(active_severity.get(FaultType.LATCH_UP, 0))
            obs["thermal_severity"] = float(active_severity.get(FaultType.THERMAL_RUNAWAY, 0))
            obs["memory_severity"] = float(active_severity.get(FaultType.MEMORY_CORRUPTION, 0))
            obs["power_severity"] = float(active_severity.get(FaultType.POWER_FAULT, 0))
            obs["severity_level_total"] = float(
                obs["seu_severity"]
                + obs["latchup_severity"]
                + obs["thermal_severity"]
                + obs["memory_severity"]
                + obs["power_severity"]
            )

        # Apply telemetry noise to physical channels only
        if self._injector is not None:
            noise = self._injector.get_telemetry_noise(self._step_count)
            if noise is not None:
                # Map old noise channel names → new physical obs channels
                noise_map = {
                    "battery_level":        "battery_soc",
                    "temperature":          "cpu_temperature",
                    "cpu_health":           "cpu_load",
                    "communication_health": "memory_integrity",
                }
                for old_key, new_key in noise_map.items():
                    if old_key in noise:
                        obs[new_key] = _clamp(obs[new_key] + noise[old_key])
                # Also apply noise to legacy keys for consistency
                for old_key in ("battery_level", "temperature", "cpu_health", "communication_health"):
                    if old_key in noise:
                        obs[old_key] = _clamp(obs[old_key] + noise[old_key])

        return obs


    def _log_step(
        self,
        obs: dict,
        done: bool,
        failure_reason: str,
        fault_event: Optional[FaultEvent],
        action_effect: Optional[ActionEffect],
    ) -> None:
        record = {
            "step":              self._step_count,
            **obs,
            "done":              done,
            "failure_reason":    failure_reason,
            # Fault metadata
            "fault_type":        fault_event.fault_type.name if fault_event else None,
            "fault_subsystem":   fault_event.subsystem      if fault_event else None,
            "fault_magnitude":   fault_event.magnitude      if fault_event else None,
            # Action metadata
            "action":            action_effect.action_type.name if action_effect else None,
            "action_delta_battery":  action_effect.delta_battery     if action_effect else None,
            "action_delta_temp":     action_effect.delta_temperature  if action_effect else None,
            "action_delta_cpu":      action_effect.delta_cpu          if action_effect else None,
            "action_delta_comms":    action_effect.delta_comms        if action_effect else None,
        }
        self._history.append(record)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TITANEnv("
            f"step={self._step_count}, done={self._done}, "
            f"battery={self._state.battery_level:.3f}, "
            f"temp={self._state.temperature:.3f}, "
            f"cpu={self._state.cpu_health:.3f}, "
            f"comms={self._state.communication_health:.3f})"
        )


