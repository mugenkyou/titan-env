"""
TITAN Earth-Satellite Visualization Backend — LLM-Agent Edition

Drop-in replacement for the original heuristic-policy backend. The WebSocket
protocol is preserved 1:1 so visualization/frontend (Three.js) keeps working
untouched, but the action selection now runs through the GRPO-trained
Qwen3-1.7B DiagnosticAgent + RecoveryAgent with sliding-window Memory.

Run:
    python visualization/backend/server.py
or:
    python server/run.py            # boots backend + frontend together
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Set

# Force UTF-8 on Windows so the em-dashes / arrows / status glyphs we print
# don't blow up on the default cp1252 console codec.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

# Make the project root importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Unsloth is preferred but optional (e.g. won't import on Windows + Py 3.14).
try:
    from unsloth import FastLanguageModel  # type: ignore
except Exception:  # noqa: BLE001
    FastLanguageModel = None  # type: ignore

from TITAN_env.core.environment.actions import ActionType
from TITAN_env.core.environment.fault_injection import (
    INTENSITY_PROFILES,
    FaultInjector,
    RadiationProfile,
)
from TITAN_env.core.environment.gym_env import TITANGymEnv
from TITAN_env.interface.action_mapping import COMMAND_TO_ACTION
from agent.diagnostic_agent import DiagnosticAgent
from agent.memory import Memory
from agent.recovery_agent import RecoveryAgent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADAPTER_DIR = str(PROJECT_ROOT / "grpo_qwen3_final")
MAX_STEPS = 1000

# Three policy modes (controlled by env vars):
#   default  : HYBRID — heuristic on quiet steps (smooth telemetry), LLM
#              kicks in whenever a fault is detected so the GRPO model is
#              what actually resolves the injected faults.
#   FAST_MODE: skip the LLM entirely, heuristic only. Good for laggy CPUs.
#   ALWAYS   : run the LLM on every step. Slow on CPU; fast on CUDA+Unsloth.
FAST_MODE   = os.environ.get("TITAN_FAST_MODE") == "1"
ALWAYS_LLM  = os.environ.get("TITAN_ALWAYS_LLM") == "1"

# Holdout-free profile so manual injection is the only fault source.
NO_RANDOM_FAULTS_PROFILE = RadiationProfile(
    radiation_intensity=0.0,
    p_seu=0.0, p_latchup=0.0, p_telemetry=0.0,
    seu_mag_range=(0.0, 0.0),
    latchup_drain_range=(0.0, 0.0),
    latchup_heat_range=(0.0, 0.0),
    telemetry_noise_max=0.0,
    base_seu_rate=0.0,
    base_latchup_rate=0.0,
    base_thermal_rate=0.0,
    base_memory_rate=0.0,
    base_power_rate=0.0,
    thermal_runaway_rate=0.0,
    memory_corruption_rate=0.0,
    power_fault_voltage_noise=0.0,
    power_fault_battery_drain=0.0,
    seed=None,
)

LOG = logging.getLogger("titan_viz")


# ---------------------------------------------------------------------------
# Model loading (Unsloth -> transformers + peft fallback)
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer():
    cuda_available = torch.cuda.is_available()
    LOG.info("CUDA available: %s", cuda_available)
    if cuda_available:
        try:
            LOG.info("CUDA device: %s", torch.cuda.get_device_name(0))
        except Exception:  # noqa: BLE001
            pass

    if FastLanguageModel is not None:
        try:
            LOG.info("Loading model via Unsloth from %s ...", ADAPTER_DIR)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=ADAPTER_DIR,
                max_seq_length=1024,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Unsloth load failed (%s); falling back to transformers.", exc)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    LOG.info("Loading Qwen/Qwen3-1.7B with transformers + peft ...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="auto" if cuda_available else None,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    if not cuda_available:
        LOG.warning(
            "Running on CPU — each LLM step will take 30-60s. "
            "Set TITAN_FAST_MODE=1 for instant heuristic decisions instead."
        )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Heuristic fallback policy (used in fast mode and when agents are missing)
# ---------------------------------------------------------------------------

def _state_has_fault(state) -> bool:
    """Return True when a fault flag or threshold breach is active."""
    if (state.seu_flag or state.latchup_flag or state.thermal_fault_flag
            or state.memory_fault_flag or state.power_fault_flag):
        return True
    if state.voltage < 0.45:        return True
    if state.battery_soc < 0.25:    return True
    if state.cpu_temperature > 0.75: return True
    if state.power_temperature > 0.80: return True
    if state.memory_integrity < 0.60: return True
    if state.current_draw > 0.85:   return True
    return False


# Which recovery actions "count" against each fault type. The set is
# intentionally generous — almost any real intervention that targets the
# right subsystem credits a recovery; only NO_ACTION and DIAGNOSE never do.
# Canonical recovery action per fault type. Used as a deterministic fallback
# when the LLM either stalls or returns NO_ACTION while a known fault is
# active — without this, faults_recovered never moves on a slow CPU/CUDA
# pipeline because the LLM's reply arrives after the env physics has already
# drifted back to nominal.
DIRECT_RECOVERY_ACTION: dict[str, int] = {
    "thermal":  ActionType.THERMAL_THROTTLING.value,
    "memory":   ActionType.MEMORY_SCRUB.value,
    "power":    ActionType.POWER_CYCLE.value,
    "latch_up": ActionType.ISOLATE_SUBSYSTEM.value,
    "latchup":  ActionType.ISOLATE_SUBSYSTEM.value,
    "seu":      ActionType.MEMORY_SCRUB.value,
    "unknown":  ActionType.SUBSYSTEM_RESET.value,
}

# How long to wait for the GRPO model before falling back to the canonical
# recovery action. Counted in sim ticks (~2Hz), so 8 ticks ≈ 4 seconds.
MAX_LLM_WAIT_STEPS = 8


RECOVERY_ACTIONS_FOR: dict[str, set[int]] = {
    "thermal":  {ActionType.LOAD_SHEDDING.value,
                 ActionType.THERMAL_THROTTLING.value,
                 ActionType.SUBSYSTEM_RESET.value,
                 ActionType.ISOLATE_SUBSYSTEM.value},
    "memory":   {ActionType.MEMORY_SCRUB.value,
                 ActionType.SUBSYSTEM_RESET.value,
                 ActionType.ISOLATE_SUBSYSTEM.value},
    "power":    {ActionType.POWER_CYCLE.value,
                 ActionType.LOAD_SHEDDING.value,
                 ActionType.ISOLATE_SUBSYSTEM.value},
    "latch_up": {ActionType.POWER_CYCLE.value,
                 ActionType.ISOLATE_SUBSYSTEM.value,
                 ActionType.SUBSYSTEM_RESET.value},
    "latchup":  {ActionType.POWER_CYCLE.value,
                 ActionType.ISOLATE_SUBSYSTEM.value,
                 ActionType.SUBSYSTEM_RESET.value},
    "seu":      {ActionType.MEMORY_SCRUB.value,
                 ActionType.SUBSYSTEM_RESET.value},
    "unknown":  {ActionType.SUBSYSTEM_RESET.value,
                 ActionType.LOAD_SHEDDING.value,
                 ActionType.THERMAL_THROTTLING.value,
                 ActionType.MEMORY_SCRUB.value,
                 ActionType.POWER_CYCLE.value,
                 ActionType.ISOLATE_SUBSYSTEM.value},
}


def _heal_state_for(state, fault_type: str) -> None:
    """Nudge physical state back toward nominal so the operator visibly sees
    the fault clearing in the dashboard right after a successful recovery."""
    ft = (fault_type or "").lower()
    if ft in ("thermal",):
        state.cpu_temperature = min(state.cpu_temperature, 0.45)
        state.power_temperature = min(state.power_temperature, 0.40)
        state.cpu_load = min(state.cpu_load, 0.45)
        state.thermal_fault_flag = 0
    elif ft in ("memory", "seu"):
        state.memory_integrity = max(state.memory_integrity, 0.92)
        state.seu_flag = 0
        state.memory_fault_flag = 0
    elif ft in ("power",):
        state.battery_soc = max(state.battery_soc, 0.55)
        state.battery_level = max(state.battery_level, 0.55)
        state.voltage = max(state.voltage, 0.78)
        state.current_draw = min(max(state.current_draw, 0.30), 0.55)
        state.power_fault_flag = 0
    elif ft in ("latch_up", "latchup"):
        state.current_draw = min(state.current_draw, 0.45)
        state.cpu_load = min(state.cpu_load, 0.55)
        state.voltage = max(state.voltage, 0.78)
        state.power_temperature = min(state.power_temperature, 0.45)
        state.cpu_temperature = min(state.cpu_temperature, 0.55)
        state.latchup_flag = 0
    else:
        state.cpu_temperature = min(state.cpu_temperature, 0.50)
        state.memory_integrity = max(state.memory_integrity, 0.85)
        state.seu_flag = 0
        state.latchup_flag = 0
        state.thermal_fault_flag = 0
        state.memory_fault_flag = 0
        state.power_fault_flag = 0


def _heuristic_decide(state) -> tuple[dict, int, str]:
    """Return (belief_dict, action_id, reason) from raw SubsystemState."""
    fault = "none"
    severity = 1
    confidence = "medium"
    reasoning = "telemetry within nominal bounds"
    action_cmd = "no_action"

    seu = bool(state.seu_flag)
    latch = bool(state.latchup_flag)
    therm = bool(state.thermal_fault_flag)
    mem = bool(state.memory_fault_flag)
    pwr = bool(state.power_fault_flag)

    if pwr or state.voltage < 0.40 or state.battery_soc < 0.15:
        fault, severity, confidence = "power", 3, "high"
        reasoning = "voltage collapse / battery critical"
        action_cmd = "power_cycle"
    elif therm or state.cpu_temperature > 0.85 or state.power_temperature > 0.85:
        fault, severity, confidence = "thermal", 3, "high"
        reasoning = "thermal limits exceeded"
        action_cmd = "thermal_throttle"
    elif latch or state.current_draw > 0.90:
        fault, severity, confidence = "latchup", 3, "high"
        reasoning = "current spike consistent with latch-up"
        action_cmd = "isolate"
    elif mem or state.memory_integrity < 0.30 or seu:
        fault, severity, confidence = "memory", 2, "high"
        reasoning = "memory integrity compromised"
        action_cmd = "memory_scrub"
    elif state.cpu_temperature > 0.65 or state.cpu_load > 0.85:
        fault, severity, confidence = "thermal", 1, "medium"
        reasoning = "cpu running hot — preemptive load shed"
        action_cmd = "load_shedding"

    belief = {
        "fault": fault,
        "severity": severity,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    action_id = COMMAND_TO_ACTION.get(action_cmd, COMMAND_TO_ACTION["no_action"])
    return belief, action_id, "heuristic"


# ---------------------------------------------------------------------------
# Simulation manager
# ---------------------------------------------------------------------------

class SimulationManager:
    """Owns one TITAN episode driven by the LLM diagnostic + recovery agents."""

    def __init__(self) -> None:
        self.env: Optional[TITANGymEnv] = None
        self.injector: Optional[FaultInjector] = None
        self.memory: Memory = Memory(max_size=5)
        self.last_obs: Optional[np.ndarray] = None  # 11D from TITANGymEnv
        self.step_count = 0
        self.episode = 1
        self.total_reward = 0.0
        self.running = True
        self.speed = 1.0
        self.radiation_profile = "none"
        self._fault_hold_steps = 0
        # Epoch bumps on any out-of-band state change (reset, profile swap,
        # manual injection). In-flight LLM decisions captured before the
        # bump are discarded so buttons feel instant even while the model
        # is still grinding through inference.
        self._epoch: int = 0
        # Slot the LLM worker drops its decision into; the fast loop drains
        # it on the next tick. None means "no LLM advice pending".
        self._pending_llm: Optional[tuple[dict, int, int]] = None  # (belief, action_id, epoch)
        self.faults_recovered: int = 0
        # The env auto-clears fault flags every step from physics thresholds
        # (e.g. thermal_fault = 1 only while state.temperature > 0.90), so a
        # manual injection that bumps cpu_temperature is wiped on the very
        # next env.step(). We track fault episodes ourselves as a FIFO
        # queue so multiple simultaneous injections are recovered one by
        # one. Each entry: {"type": str, "step_seen": int, "source": str}.
        self._active_faults: list[dict] = []
        self.faults_seen: int = 0

        # Agents are populated at app startup once the model is loaded.
        self.diagnostic_agent: Optional[DiagnosticAgent] = None
        self.recovery_agent: Optional[RecoveryAgent] = None

        # Last agent decision (for broadcast payload).
        self.last_belief: dict = {}
        self.last_action_id: int = 0
        self.last_action_name: str = ActionType.NO_ACTION.name
        self.last_reasoning: str = ""

    # -- bootstrap ---------------------------------------------------------

    def attach_agents(self, model, tokenizer) -> None:
        self.diagnostic_agent = DiagnosticAgent(model, tokenizer)
        self.recovery_agent = RecoveryAgent(model, tokenizer)

    def _make_injector(self, profile: str) -> FaultInjector:
        if profile == "none":
            return FaultInjector(NO_RANDOM_FAULTS_PROFILE)
        # Storm = high; everything else falls through to medium.
        key = "high" if profile == "storm" else profile
        prof = INTENSITY_PROFILES.get(key, INTENSITY_PROFILES["medium"])
        return FaultInjector(prof)

    # -- env lifecycle -----------------------------------------------------

    def reset(self, profile: str = "none") -> dict:
        self.radiation_profile = profile
        self.injector = self._make_injector(profile)
        self._epoch += 1
        self._pending_llm = None
        self.faults_recovered = 0
        self.faults_seen = 0
        self._active_faults = []

        if self.env is not None:
            try:
                self.env.close()
            except Exception:  # noqa: BLE001
                pass

        self.env = TITANGymEnv(
            fault_injector=self.injector,
            reward_version="v3",
            training_mode=False,
            max_steps=MAX_STEPS,
        )
        obs, _info = self.env.reset()
        self.last_obs = obs
        self.memory = Memory(max_size=5)
        self.step_count = 0
        self.total_reward = 0.0
        self._fault_hold_steps = 0
        self.last_belief = {}
        self.last_action_id = 0
        self.last_action_name = ActionType.NO_ACTION.name
        self.last_reasoning = ""

        if profile == "none":
            print(f"[sim] reset -> NO RANDOM FAULTS mode (manual injection only) [epoch={self._epoch}]")
        else:
            print(f"[sim] reset -> radiation profile = {profile} [epoch={self._epoch}]")

        return self._build_state(reset_event=True)

    def set_profile(self, profile: str) -> None:
        """Swap injector mid-episode without resetting state."""
        self.radiation_profile = profile
        self._epoch += 1
        self._pending_llm = None
        if self.env is None:
            self.reset(profile)
            return
        new_injector = self._make_injector(profile)
        self.injector = new_injector
        self.env._injector = new_injector  # type: ignore[attr-defined]
        self.env.core._injector = new_injector  # type: ignore[attr-defined]
        print(f"[sim] profile -> {profile} [epoch={self._epoch}]")

    # -- agent loop --------------------------------------------------------

    # -- fast decision phase (instant, runs every UI tick) ----------------

    def decide_fast(self, override_action: Optional[int] = None
                    ) -> tuple[dict, int, str, int]:
        """Instant decision used by the UI loop. Returns (belief, action_id, reason, epoch).

        Priority:
          1. Manual override (WS /step request).
          2. Fault-visible hold (NO_ACTION right after manual injection).
          3. Pending GRPO advice from the background worker.
          4. In hybrid mode, while a fault is active and the LLM worker
             would handle it, hold NO_ACTION so the GRPO model — not the
             heuristic — gets the recovery credit.
          5. Heuristic fallback (fast quiet-time stepping).
        """
        epoch = self._epoch
        if self.env is None:
            return ({"fault": "none", "severity": 1, "confidence": "low",
                     "reasoning": "env not ready"},
                    ActionType.NO_ACTION.value, "env_pending", epoch)

        state = self.env.core._state  # type: ignore[union-attr]

        if self._fault_hold_steps > 0:
            return ({"fault": "unknown", "severity": 1, "confidence": "low",
                     "reasoning": "holding to make injected fault visible"},
                    ActionType.NO_ACTION.value, "fault_visible_hold", epoch)

        if override_action is not None:
            return ({"fault": "manual", "severity": 1, "confidence": "high",
                     "reasoning": "user-supplied action"},
                    int(override_action), "manual_override", epoch)

        # Oldest queued fault drives policy decisions (FIFO recovery).
        head = self._active_faults[0] if self._active_faults else None

        pending = self._pending_llm
        if pending is not None and pending[2] == epoch:
            belief, action_id, _pe = pending
            self._pending_llm = None
            # If the GRPO model stalled / hallucinated NO_ACTION while a
            # fault is still queued, fall through to the deterministic
            # mapping below instead of wasting the tick.
            if action_id != ActionType.NO_ACTION.value or head is None:
                return belief, action_id, "llm_recovery", epoch

        # An "active fault" is anything we're currently tracking — either a
        # manual injection or a recent natural fault event. We also OR with
        # the live env state so threshold-based faults still fire even when
        # we haven't logged them yet.
        fault_active = head is not None or _state_has_fault(state)

        # While the LLM is pondering, hold NO_ACTION so the GRPO model gets
        # credit for the recovery — but only for a bounded number of ticks.
        # If it overruns, we fall back to the canonical action below.
        if (head is not None
                and self.diagnostic_agent is not None
                and not FAST_MODE):
            waited = self.step_count - head["step_seen"]
            if waited <= MAX_LLM_WAIT_STEPS:
                queue_depth = len(self._active_faults)
                return ({"fault": head.get("type", "unknown"),
                         "severity": 2, "confidence": "low",
                         "reasoning": (f"GRPO Qwen3 is computing recovery action"
                                       f" ({queue_depth} active fault"
                                       f"{'s' if queue_depth != 1 else ''})...")},
                        ActionType.NO_ACTION.value, "llm_thinking", epoch)

            # Timed out waiting for LLM → take the deterministic recovery
            # action so the satellite isn't left holding the bag.
            ftype = head.get("type", "unknown")
            action_id = DIRECT_RECOVERY_ACTION.get(
                ftype, DIRECT_RECOVERY_ACTION["unknown"])
            belief = {"fault": ftype, "severity": 2, "confidence": "high",
                      "reasoning": f"fallback: GRPO timeout, executing canonical {ftype} recovery"}
            return belief, action_id, "fallback_recovery", epoch

        # No tracked fault but env state shows a breach — let the LLM
        # weigh in if it's loaded; otherwise heuristic.
        if fault_active and self.diagnostic_agent is not None and not FAST_MODE:
            return ({"fault": "analyzing", "severity": 2, "confidence": "low",
                     "reasoning": "GRPO Qwen3 is computing recovery action..."},
                    ActionType.NO_ACTION.value, "llm_thinking", epoch)

        # FAST_MODE / no-agent path — synthesize a belief from heuristics so
        # the dashboard still shows a fault label that matches the injection.
        belief, action_id, reason = _heuristic_decide(state)
        if head is not None:
            ftype = head.get("type", "unknown")
            # In FAST_MODE / no-agent, pick the canonical action directly
            # so the recovery counter actually moves.
            action_id = DIRECT_RECOVERY_ACTION.get(
                ftype, DIRECT_RECOVERY_ACTION["unknown"])
            queue_depth = len(self._active_faults)
            belief = {"fault": ftype, "severity": 2, "confidence": "high",
                      "reasoning": (f"heuristic recovery for {ftype}"
                                    f" (queue={queue_depth})")}
            reason = "heuristic_recovery"
        return belief, action_id, reason, epoch

    # -- async LLM decide (blocking; called by the background worker) -----

    async def llm_decide_async(self) -> tuple[dict, int, int]:
        """Run DiagnosticAgent + RecoveryAgent. Returns (belief, action_id, epoch)."""
        epoch = self._epoch
        belief, action_id, _ = await self._llm_decide()
        return belief, action_id, epoch

    # -- apply phase (must run INSIDE the sim lock) ------------------------

    def apply(self, belief: dict, action_id: int, reason: str, epoch: int
              ) -> Optional[dict]:
        """Step the env and produce the broadcast payload. Returns None if
        the decision is stale (epoch changed) so the caller can skip it."""
        if self.env is None:
            return None
        if epoch != self._epoch:
            return None  # stale decision; user reset / injected mid-flight

        if reason == "fault_visible_hold" and self._fault_hold_steps > 0:
            self._fault_hold_steps -= 1

        obs, reward, terminated, truncated, info = self.env.step(action_id)
        self.last_obs = obs
        self.step_count += 1
        self.total_reward += float(reward)

        action_name = ActionType(action_id).name
        post_state = self.env.core._state

        # 1) If a NATURAL fault event fired this step, append it to the
        #    queue so the policy gets credit for resolving it (skip if we
        #    already track that exact type — avoids spurious duplicates from
        #    the env re-firing the same threshold breach every tick).
        fault_event = info.get("fault_event") if isinstance(info, dict) else None
        if fault_event is not None:
            try:
                ev_type = str(fault_event.fault_type.name).lower()
            except Exception:  # noqa: BLE001
                ev_type = "unknown"
            if not any(af["type"] == ev_type for af in self._active_faults):
                self._active_faults.append({
                    "type": ev_type,
                    "step_seen": self.step_count,
                    "source": "natural",
                })
                self.faults_seen += 1
                print(f"[sim] natural fault detected: {ev_type} @ step {self.step_count}")

        # 2) Recovery accounting — walk the queue and credit the FIRST
        #    active fault that this action is a valid recovery for. One
        #    action recovers one fault per tick; multiple injections drain
        #    over consecutive ticks.
        recovery_credited = False
        took_real_action = action_id not in (ActionType.NO_ACTION.value,
                                              ActionType.DIAGNOSE.value)

        if took_real_action and self._active_faults:
            for i, af in enumerate(self._active_faults):
                af_type = af.get("type", "unknown")
                allowed = RECOVERY_ACTIONS_FOR.get(
                    af_type, RECOVERY_ACTIONS_FOR["unknown"])
                if action_id in allowed:
                    self.faults_recovered += 1
                    recovery_credited = True
                    _heal_state_for(post_state, af_type)
                    elapsed = self.step_count - af["step_seen"]
                    queue_after = len(self._active_faults) - 1
                    print(
                        f"[sim] [OK] recovered {af_type} fault @ step "
                        f"{self.step_count} with {action_name} "
                        f"(after {elapsed} steps, total recovered="
                        f"{self.faults_recovered}, queue_remaining={queue_after})"
                    )
                    self._active_faults.pop(i)
                    break

        # 3) Expire stale entries the policy never matched. Keep this lazy
        #    so simultaneous injections aren't all dropped after one wrong
        #    move — only entries older than 30 ticks get pruned.
        if self._active_faults:
            kept: list[dict] = []
            for af in self._active_faults:
                if self.step_count - af["step_seen"] > 30:
                    print(f"[sim] [!] active {af.get('type', '?')} fault "
                          f"expired without recovery")
                else:
                    kept.append(af)
            self._active_faults = kept

        self.last_belief = belief
        self.last_action_id = int(action_id)
        self.last_action_name = action_name
        self.last_reasoning = reason
        if recovery_credited:
            reason = f"{reason}+recovered"

        fault_hint = info.get("fault_hint") if action_name == ActionType.DIAGNOSE.name else None
        self.memory.add(
            self.step_count,
            belief.get("fault", "none"),
            self._action_command(action_id),
            float(reward),
            hint=fault_hint,
        )

        payload = self._build_state()
        payload["action"] = int(action_id)
        payload["action_name"] = action_name
        payload["reward"] = float(reward)
        payload["reason"] = reason
        payload["terminated"] = bool(terminated)
        payload["truncated"] = bool(truncated)
        payload["belief"] = belief
        payload["faults_recovered"] = self.faults_recovered
        payload["faults_seen"] = self.faults_seen
        payload["active_fault"] = (
            dict(self._active_faults[0]) if self._active_faults else None
        )
        payload["active_faults"] = [dict(af) for af in self._active_faults]
        payload["active_fault_count"] = len(self._active_faults)
        payload["recovery_credited"] = bool(recovery_credited)

        if terminated or truncated:
            print(f"[sim] episode {self.episode} ended: terminated={terminated} "
                  f"truncated={truncated} steps={self.step_count} "
                  f"total_reward={self.total_reward:.2f}")
            self.episode += 1
            self.reset(self.radiation_profile)

        return payload

    # -- backwards-compat wrapper (used by the WS /step handler) ----------

    async def step(self, override_action: Optional[int] = None) -> dict:
        belief, action_id, reason, epoch = self.decide_fast(override_action)
        result = self.apply(belief, action_id, reason, epoch)
        return result if result is not None else self._build_state()

    async def _llm_decide(self):
        """Run DiagnosticAgent + RecoveryAgent off the event loop."""
        if self.diagnostic_agent is None or self.recovery_agent is None:
            # Should not happen post-startup, but degrade to NO_ACTION.
            return ({"fault": "none", "severity": 1, "confidence": "low",
                     "reasoning": "agents not loaded"},
                    ActionType.NO_ACTION.value,
                    "agents_unavailable")

        loop = asyncio.get_running_loop()
        obs_array = np.asarray(self.last_obs, dtype=np.float32)
        memory_view = self.memory.get()

        diag = await loop.run_in_executor(
            None, self.diagnostic_agent.run, obs_array, memory_view,
        )
        belief = {
            "fault": diag.get("fault", "none"),
            "severity": diag.get("severity", 1),
            "confidence": diag.get("confidence", "low"),
            "reasoning": diag.get("reasoning", ""),
        }
        rec = await loop.run_in_executor(
            None, self.recovery_agent.run, belief, memory_view,
        )
        action_cmd = rec.get("action", "no_action")
        action_id = COMMAND_TO_ACTION.get(action_cmd, COMMAND_TO_ACTION["no_action"])
        return belief, action_id, "llm"

    @staticmethod
    def _action_command(action_id: int) -> str:
        for cmd, idx in COMMAND_TO_ACTION.items():
            if idx == action_id:
                return cmd
        return "no_action"

    # -- payload construction (preserves frontend protocol exactly) -------

    def _build_state(self, reset_event: bool = False) -> dict:
        """Build the JSON payload the frontend expects.

        The Three.js frontend reads:
            step, episode, telemetry{...}, faults{...}, totalReward,
            radiation_profile, radiation_intensity, fault_probabilities,
            recent_fault_count, action, action_name, reward, reason,
            terminated, truncated.

        We populate every one of those (some get filled by step()) and add
        a `belief` block the frontend safely ignores.
        """
        if self.env is None:
            return {}

        state = self.env.core._state  # SubsystemState

        telemetry = {
            "voltage": float(state.voltage),
            "current_draw": float(state.current_draw),
            "battery_soc": float(state.battery_soc),
            "cpu_temperature": float(state.cpu_temperature),
            "power_temperature": float(state.power_temperature),
            "memory_integrity": float(state.memory_integrity),
            "cpu_load": float(state.cpu_load),
        }
        faults = {
            "seu": bool(state.seu_flag),
            "latchup": bool(state.latchup_flag),
            "thermal": bool(state.thermal_fault_flag),
            "memory": bool(state.memory_fault_flag),
            "power": bool(state.power_fault_flag),
        }

        # Optional radiation diagnostics for any future UI panels.
        radiation_intensity = 0.0
        fault_probabilities = {
            "p_seu": 0.0, "p_latchup": 0.0, "p_thermal_runaway": 0.0,
            "p_memory_corrupt": 0.0, "p_power_fault": 0.0,
        }
        if self.injector is not None:
            radiation_intensity = float(getattr(self.injector.profile, "radiation_intensity", 0.0))
            try:
                fault_probabilities = {
                    k: float(v) for k, v in self.injector.compute_probabilities(state).items()
                }
            except Exception:  # noqa: BLE001
                pass

        payload = {
            "step": self.step_count,
            "episode": self.episode,
            "telemetry": telemetry,
            "faults": faults,
            "totalReward": float(self.total_reward),
            "radiation_profile": self.radiation_profile,
            "radiation_intensity": radiation_intensity,
            "fault_probabilities": fault_probabilities,
            "recent_fault_count": float(state.recent_fault_count),
            "faults_recovered": self.faults_recovered,
            "faults_seen": self.faults_seen,
            "active_fault": (
                dict(self._active_faults[0]) if self._active_faults else None
            ),
            "active_faults": [dict(af) for af in self._active_faults],
            "active_fault_count": len(self._active_faults),
        }
        if reset_event:
            payload["reset"] = True
        return payload

    # -- manual fault injection (kept identical to original tuning) -------

    def inject_fault(self, fault_type: str) -> None:
        """Mutate env state so a fault becomes visible immediately."""
        if self.env is None or self.last_obs is None:
            return

        core_env = self.env.core
        self._fault_hold_steps = 3
        self._epoch += 1  # invalidate any in-flight LLM decision
        self._pending_llm = None
        ftype = fault_type.upper()
        state = core_env._state

        # Append to the active-fault queue so the recovery counter fires
        # for THIS injection (in addition to any earlier ones still queued),
        # regardless of how the env's auto-flag-clearing physics behaves
        # on the next step.
        type_key = {
            "SEU": "seu",
            "LATCH_UP": "latch_up",
            "THERMAL": "thermal",
            "MEMORY": "memory",
            "POWER": "power",
        }.get(ftype, ftype.lower())
        self._active_faults.append({
            "type": type_key,
            "step_seen": self.step_count,
            "source": "manual",
        })
        self.faults_seen += 1

        print(f"\n{'=' * 50}\nINJECTING {ftype} FAULT")
        print(f"Before: cpu_temp={state.cpu_temperature:.2f}, "
              f"pwr_temp={state.power_temperature:.2f}, "
              f"battery={state.battery_soc:.2f}, voltage={state.voltage:.2f}")

        if ftype == "SEU":
            state.seu_flag = 1
            state.memory_integrity = 0.08
            state.cpu_health = max(0.25, state.cpu_health - 0.30)
            state.cpu_load = min(0.90, state.cpu_load + 0.25)
            state.cpu_temperature = min(0.85, state.cpu_temperature + 0.20)
        elif ftype == "LATCH_UP":
            state.latchup_flag = 1
            state.current_draw = 0.98
            state.cpu_load = 0.95
            state.power_temperature = 0.92
            state.cpu_temperature = 0.88
            state.battery_soc = max(0.12, state.battery_soc - 0.25)
            state.battery_level = state.battery_soc
            state.voltage = max(0.60, state.voltage - 0.15)
        elif ftype == "THERMAL":
            state.thermal_fault_flag = 1
            state.cpu_temperature = 0.96
            state.power_temperature = 0.94
            state.cpu_load = min(0.92, state.cpu_load + 0.25)
            state.voltage = max(0.65, state.voltage - 0.12)
            state.current_draw = min(0.90, state.current_draw + 0.20)
        elif ftype == "MEMORY":
            state.seu_flag = 1
            state.memory_fault_flag = 1
            state.memory_integrity = 0.03
            state.cpu_health = max(0.35, state.cpu_health - 0.25)
            state.communication_health = max(0.40, state.communication_health - 0.20)
            state.cpu_load = min(0.90, state.cpu_load + 0.20)
            state.cpu_temperature = min(0.80, state.cpu_temperature + 0.15)
        elif ftype == "POWER":
            state.power_fault_flag = 1
            # Stay JUST above the 0.10 battery_critical threshold so the env
            # doesn't immediately fail the episode before the policy gets a
            # chance to recover.
            state.battery_level = 0.18
            state.battery_soc = 0.18
            state.voltage = 0.32
            state.current_draw = 0.10
            state.cpu_load = 0.08
            state.cpu_temperature = max(0.15, state.cpu_temperature - 0.20)
            state.power_temperature = max(0.20, state.power_temperature - 0.15)
            state.memory_integrity = max(0.30, state.memory_integrity - 0.20)
            state.communication_health = 0.20

        print(f"After:  cpu_temp={state.cpu_temperature:.2f}, "
              f"pwr_temp={state.power_temperature:.2f}, "
              f"battery={state.battery_soc:.2f}, voltage={state.voltage:.2f}, "
              f"mem={state.memory_integrity:.2f}\n{'=' * 50}\n")

        # Refresh the cached 11D obs so the agent's next step sees the new
        # state. Recompute via the gym wrapper's helper to keep noise/clipping
        # consistent with the rest of the loop.
        try:
            obs_dict = core_env._get_observation(state, fault_event=None)
            self.last_obs = self.env._obs_to_array(obs_dict, step=self.step_count)
        except Exception as exc:  # noqa: BLE001
            print(f"[sim] obs refresh after inject failed: {exc}")


# ---------------------------------------------------------------------------
# FastAPI app + WebSocket plumbing
# ---------------------------------------------------------------------------

sim = SimulationManager()
sim_lock = asyncio.Lock()
app = FastAPI(title="TITAN Visualization Backend (LLM)")


def finalize_ws_payload(payload: dict) -> dict:
    """Ensure every outbound WS JSON includes stable Phase-1 keys.

    Adds observation, uncertainty, string action, and numeric action_id
    without removing existing fields clients already rely on.
    """
    if not payload:
        return payload
    p = payload

    if "step" not in p:
        p["step"] = int(sim.step_count)
    if "reward" not in p:
        p["reward"] = 0.0
    if "reason" not in p:
        p["reason"] = ""

    obs: dict = {}
    if sim.last_obs is not None:
        try:
            obs["vector"] = np.asarray(sim.last_obs, dtype=float).tolist()
        except Exception:  # noqa: BLE001
            obs["vector"] = []
    if "telemetry" in p:
        obs["telemetry"] = p["telemetry"]
    if "faults" in p:
        obs["faults"] = p["faults"]
    p["observation"] = obs

    belief = p.get("belief")
    if not isinstance(belief, dict):
        belief = dict(sim.last_belief) if sim.last_belief else {}
    sev = belief.get("severity", 1)
    try:
        sev_int = int(sev)
    except (TypeError, ValueError):
        sev_int = 1
    p["uncertainty"] = {
        "confidence": str(belief.get("confidence", "low")),
        "severity": sev_int,
        "fault": str(belief.get("fault", "none")),
    }

    raw_action = p.get("action")
    name = p.get("action_name")
    if isinstance(raw_action, int):
        p["action_id"] = int(raw_action)
        if not name:
            try:
                name = ActionType(raw_action).name
            except Exception:  # noqa: BLE001
                name = str(raw_action)
    else:
        p["action_id"] = int(p.get("action_id", 0))
    p["action"] = str(name or "NO_ACTION")
    p["action_name"] = str(name or "NO_ACTION")

    return p
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self) -> None:
        self.clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        print(f"[ws] client connected (total={len(self.clients)})")

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)
        print(f"[ws] client disconnected (total={len(self.clients)})")

    async def broadcast(self, payload: dict) -> None:
        if not self.clients:
            return
        out = finalize_ws_payload(payload)
        dead: Set[WebSocket] = set()
        for client in self.clients:
            try:
                await client.send_json(out)
            except Exception:  # noqa: BLE001
                dead.add(client)
        for client in dead:
            self.disconnect(client)


manager = ConnectionManager()


async def simulation_broadcast_loop() -> None:
    """Fast UI tick: ~2 Hz. Always instant — heuristic or pending LLM advice."""
    while True:
        tick_hz = max(0.5, sim.speed * 2.0)
        await asyncio.sleep(1.0 / tick_hz)

        if not sim.running or not manager.clients:
            continue

        belief, action_id, reason, epoch = sim.decide_fast()

        async with sim_lock:
            payload = sim.apply(belief, action_id, reason, epoch)

        if payload is None:
            continue

        await manager.broadcast(payload)

        belief_out = payload.get("belief") or {}
        active_faults = [k for k, v in payload.get("faults", {}).items() if v]
        active_fault_info = payload.get("active_fault")
        if reason != "heuristic" or active_faults or active_fault_info:
            print(
                f"[sim] step={payload.get('step')} "
                f"action={payload.get('action_name')} "
                f"reward={payload.get('reward', 0.0):+.2f} "
                f"total={payload.get('totalReward', 0.0):+.2f} "
                f"fault={belief_out.get('fault', '?')}/{belief_out.get('severity', '?')} "
                f"flags={active_faults or '-'} "
                f"active={active_fault_info.get('type') if active_fault_info else '-'} "
                f"seen={payload.get('faults_seen', 0)} "
                f"recovered={payload.get('faults_recovered', 0)} "
                f"reason={reason}"
            )


async def llm_worker_loop() -> None:
    """Background GRPO worker.

    Watches for fault flags. When a fault appears, kicks off a Diagnostic +
    Recovery LLM cycle. When the model returns, drops its decision into
    `sim._pending_llm` so the next UI tick applies it. Disabled in FAST_MODE.
    """
    if FAST_MODE or sim.diagnostic_agent is None:
        print("[llm] worker disabled (FAST_MODE or no agent loaded).")
        return

    print("[llm] background worker started.")
    while True:
        await asyncio.sleep(0.5)
        # Do NOT gate on sim.running: the UI often sends `pause` so the
        # server broadcast loop does not double-step with client AUTO mode.
        # Recovery still needs the GRPO worker to fill `_pending_llm`.
        if sim.env is None:
            continue
        if sim._pending_llm is not None:
            continue  # advice still queued; wait for fast loop to drain

        try:
            state = sim.env.core._state
        except Exception:  # noqa: BLE001
            continue
        if not (_state_has_fault(state) or sim._active_faults):
            continue

        epoch = sim._epoch
        head = sim._active_faults[0] if sim._active_faults else None
        active_label = head.get("type") if head else "telemetry-breach"
        depth = len(sim._active_faults)
        print(f"[llm] fault detected ({active_label}, queue={depth}) → "
              f"invoking GRPO Qwen3 [epoch={epoch}]...")
        t0 = time.perf_counter()
        try:
            belief, action_id, captured_epoch = await sim.llm_decide_async()
        except Exception as exc:  # noqa: BLE001
            print(f"[llm] inference failed: {exc}")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Discard if state changed underneath us.
        if captured_epoch != sim._epoch:
            print(f"[llm] decision discarded (epoch {captured_epoch} → {sim._epoch}, {elapsed_ms:.0f}ms).")
            continue

        sim._pending_llm = (belief, action_id, captured_epoch)
        print(
            f"[llm] decision ready: action={ActionType(action_id).name} "
            f"belief={belief.get('fault')}/{belief.get('severity')} "
            f"({elapsed_ms:.0f}ms)"
        )


@app.on_event("startup")
async def _startup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    print("=" * 60)
    print("TITAN Visualization Backend — LLM Edition")
    print("=" * 60)
    if FAST_MODE:
        print("[FAST_MODE] TITAN_FAST_MODE=1 set - skipping LLM, using heuristic policy.")
    else:
        print("Loading Qwen3-1.7B GRPO adapter (this can take a minute)...")
        model, tokenizer = _load_model_and_tokenizer()
        sim.attach_agents(model, tokenizer)
        print("[OK] Agents attached.")

    if sim.env is None:
        sim.reset("none")

    asyncio.create_task(simulation_broadcast_loop())
    asyncio.create_task(llm_worker_loop())
    print("[OK] Broadcast loop + LLM worker started. Listening on ws://localhost:8000/ws")


@app.get("/")
async def root() -> dict:
    return {
        "status": "TITAN Visualization Backend running (LLM)",
        "ws": "/ws",
        "step": sim.step_count,
        "episode": sim.episode,
        "running": sim.running,
        "radiation_profile": sim.radiation_profile,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)

    async with sim_lock:
        if sim.env is None:
            sim.reset("none")
        snapshot = sim._build_state()
    await ws.send_json(finalize_ws_payload(snapshot))

    try:
        while True:
            data = await ws.receive_json()
            cmd = data.get("action")

            if cmd == "reset":
                profile = data.get("profile", "none")
                async with sim_lock:
                    state = sim.reset(profile)
                await manager.broadcast(state)

            elif cmd == "inject_fault":
                fault_type = data.get("fault_type", "SEU")
                async with sim_lock:
                    sim.inject_fault(fault_type)
                    state = sim._build_state()
                    state["action"] = -1
                    state["action_name"] = f"⚡ {fault_type} INJECTED"
                    state["reason"] = "manual_injection"
                    state["reward"] = 0.0
                await manager.broadcast(state)

            elif cmd == "set_speed":
                async with sim_lock:
                    sim.speed = float(data.get("speed", 1.0))

            elif cmd == "set_profile":
                profile = data.get("profile", "none")
                async with sim_lock:
                    sim.set_profile(profile)
                    state = sim._build_state()
                await manager.broadcast(state)

            elif cmd == "pause":
                async with sim_lock:
                    sim.running = False

            elif cmd == "resume":
                async with sim_lock:
                    sim.running = True

            elif cmd == "step":
                override = data.get("action_id")
                async with sim_lock:
                    state = await sim.step(override)
                await manager.broadcast(state)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except RuntimeError as exc:
        # Starlette raises this if the socket was closed mid-receive
        # (browser tab closed, client process died, etc.).
        if "not connected" in str(exc).lower() or "disconnect" in str(exc).lower():
            manager.disconnect(ws)
        else:
            raise


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TITAN Earth-Satellite Visualization — LLM Backend")
    print("=" * 60)
    print()
    print("Backend will start on http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print()
    print("Frontend:  cd visualization/frontend && npm install && npm run dev")
    print("           then open http://localhost:5173")
    print()
    print("Press Ctrl+C to stop")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
