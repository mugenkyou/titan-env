"""
TITAN API — FastAPI backend that wires the GRPO-trained Qwen3-1.7B agents into
TITANGymEnv and exposes the live episode loop over HTTP + WebSocket.

Endpoints
---------
POST /reset          start a new episode at low/medium/high radiation
POST /step           run one diagnose -> decide -> act cycle
POST /inject_fault   manually inject thermal/memory/power/latchup fault
GET  /status         current episode snapshot
WS   /ws/stream      live broadcast of every step / reset / inject event
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# Make the project root importable when run as `python server/titan_api.py`
# or via `uvicorn server.titan_api:app`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Unsloth is the preferred loader (4-bit, fast). It may not be installed on
# every host (e.g. Windows / Python 3.14), so import is guarded and we fall
# back to transformers + peft inside _load_model_and_agents.
try:
    from unsloth import FastLanguageModel
except Exception:  # noqa: BLE001 - unsloth has many possible import-time errors
    FastLanguageModel = None

from TITAN_env.core.environment.fault_injection import (
    INTENSITY_PROFILES,
    FaultInjector,
    FaultType,
)
from TITAN_env.core.environment.gym_env import TITANGymEnv
from TITAN_env.interface.action_mapping import COMMAND_TO_ACTION
from agent.diagnostic_agent import DiagnosticAgent
from agent.memory import Memory
from agent.recovery_agent import RecoveryAgent


# ---------------------------------------------------------------------------
# Constants & config
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "grpo_qwen3_final")
MAX_STEPS = 100

VALID_RADIATION_LEVELS = ("low", "medium", "high")

# Frontend uses simple labels; map them to the canonical FaultType enum here.
FAULT_LABEL_TO_TYPE: Dict[str, FaultType] = {
    "thermal": FaultType.THERMAL_RUNAWAY,
    "memory": FaultType.MEMORY_CORRUPTION,
    "power": FaultType.POWER_FAULT,
    "latchup": FaultType.LATCH_UP,
}

LOG = logging.getLogger("titan_api")


# ---------------------------------------------------------------------------
# Global state (single-episode demo server)
# ---------------------------------------------------------------------------

@dataclass
class TitanState:
    model: Any = None
    tokenizer: Any = None
    diagnostic_agent: Optional[DiagnosticAgent] = None
    recovery_agent: Optional[RecoveryAgent] = None
    env: Optional[TITANGymEnv] = None
    injector: Optional[FaultInjector] = None
    memory: Optional[Memory] = None
    radiation_level: Optional[str] = None
    last_obs: Optional[List[float]] = None
    step_count: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    last_belief: Optional[Dict[str, Any]] = None
    last_action: Optional[str] = None
    last_reward: Optional[float] = None
    ws_clients: Set[WebSocket] = field(default_factory=set)
    step_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


STATE = TitanState()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_and_agents():
    """Load the GRPO Qwen3-1.7B adapter, preferring Unsloth and falling back
    to transformers + peft if Unsloth is unavailable."""
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

    LOG.info("Loading base Qwen/Qwen3-1.7B with transformers + peft adapter ...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    LOG.info("Starting TITAN API; loading model...")
    model, tokenizer = _load_model_and_agents()
    STATE.model = model
    STATE.tokenizer = tokenizer
    STATE.diagnostic_agent = DiagnosticAgent(model, tokenizer)
    STATE.recovery_agent = RecoveryAgent(model, tokenizer)
    LOG.info("TITAN agents ready. Listening for /reset.")
    yield
    LOG.info("Shutting down TITAN API.")


app = FastAPI(title="TITAN API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    radiation_level: str = Field(default="medium", description="low | medium | high")


class InjectRequest(BaseModel):
    fault_type: str = Field(description="thermal | memory | power | latchup")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_list(obs) -> List[float]:
    return [float(x) for x in np.asarray(obs).reshape(-1)]


def _state_snapshot() -> Dict[str, Any]:
    return {
        "loaded": STATE.diagnostic_agent is not None,
        "running": STATE.env is not None,
        "radiation_level": STATE.radiation_level,
        "step": STATE.step_count,
        "max_steps": MAX_STEPS,
        "obs": STATE.last_obs,
        "total_reward": STATE.total_reward,
        "terminated": STATE.terminated,
        "truncated": STATE.truncated,
        "belief": STATE.last_belief,
        "action": STATE.last_action,
        "reward": STATE.last_reward,
        "memory": STATE.memory.get() if STATE.memory else [],
    }


async def _broadcast(payload: Dict[str, Any]) -> None:
    if not STATE.ws_clients:
        return
    dead = []
    for ws in list(STATE.ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:  # noqa: BLE001
            dead.append(ws)
    for ws in dead:
        STATE.ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset(req: ResetRequest):
    level = (req.radiation_level or "medium").lower()
    if level not in VALID_RADIATION_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"radiation_level must be one of {VALID_RADIATION_LEVELS}",
        )

    async with STATE.step_lock:
        if STATE.env is not None:
            try:
                STATE.env.close()
            except Exception:  # noqa: BLE001
                pass

        injector = FaultInjector(INTENSITY_PROFILES[level])
        env = TITANGymEnv(
            fault_injector=injector,
            reward_version="v3",
            training_mode=False,
            max_steps=MAX_STEPS,
        )
        obs, _info = env.reset()

        STATE.env = env
        STATE.injector = injector
        STATE.memory = Memory(max_size=5)
        STATE.radiation_level = level
        STATE.last_obs = _obs_to_list(obs)
        STATE.step_count = 0
        STATE.total_reward = 0.0
        STATE.terminated = False
        STATE.truncated = False
        STATE.last_belief = None
        STATE.last_action = None
        STATE.last_reward = None

    await _broadcast({
        "type": "reset",
        "radiation_level": level,
        "step": 0,
        "obs": STATE.last_obs,
        "max_steps": MAX_STEPS,
    })

    return {
        "obs": STATE.last_obs,
        "step": 0,
        "radiation_level": level,
    }


@app.post("/step")
async def step():
    if STATE.env is None or STATE.diagnostic_agent is None or STATE.memory is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    if STATE.terminated or STATE.truncated:
        raise HTTPException(status_code=409, detail="Episode is over. Call /reset to start a new one.")

    async with STATE.step_lock:
        loop = asyncio.get_running_loop()

        obs_array = np.asarray(STATE.last_obs, dtype=np.float32)
        memory_view = STATE.memory.get()

        # Both agent calls do blocking model.generate() — push them off the event loop
        # so the WebSocket can keep heart-beating other clients.
        diagnostic_result = await loop.run_in_executor(
            None, STATE.diagnostic_agent.run, obs_array, memory_view
        )
        belief = {
            "fault": diagnostic_result.get("fault", "none"),
            "severity": diagnostic_result.get("severity", 1),
            "confidence": diagnostic_result.get("confidence", "low"),
            "reasoning": diagnostic_result.get("reasoning", ""),
        }

        recovery_result = await loop.run_in_executor(
            None, STATE.recovery_agent.run, belief, memory_view
        )
        action = recovery_result.get("action", "no_action")
        action_id = COMMAND_TO_ACTION.get(action, COMMAND_TO_ACTION["no_action"])

        new_obs, reward, terminated, truncated, info = STATE.env.step(action_id)

        STATE.step_count += 1
        STATE.total_reward += float(reward)
        STATE.terminated = bool(terminated)
        STATE.truncated = bool(truncated)
        STATE.last_obs = _obs_to_list(new_obs)
        STATE.last_belief = belief
        STATE.last_action = action
        STATE.last_reward = float(reward)

        fault_hint = info.get("fault_hint") if action == "diagnose" else None
        STATE.memory.add(STATE.step_count, belief["fault"], action, float(reward), hint=fault_hint)

    payload = {
        "type": "step",
        "step": STATE.step_count,
        "obs": STATE.last_obs,
        "belief": belief,
        "action": action,
        "reward": float(reward),
        "total_reward": STATE.total_reward,
        "terminated": STATE.terminated,
        "truncated": STATE.truncated,
        "fault_hint": fault_hint,
    }
    await _broadcast(payload)
    return payload


@app.post("/inject_fault")
async def inject_fault(req: InjectRequest):
    if STATE.env is None or STATE.injector is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

    label = (req.fault_type or "").lower()
    if label not in FAULT_LABEL_TO_TYPE:
        raise HTTPException(
            status_code=400,
            detail=f"fault_type must be one of {sorted(FAULT_LABEL_TO_TYPE.keys())}",
        )

    fault_type = FAULT_LABEL_TO_TYPE[label]
    core = STATE.env.core
    new_state, event = STATE.injector.inject_manual(
        fault_type, core._state, step=STATE.step_count
    )
    core._state = new_state

    payload = {
        "type": "inject",
        "fault_type": label,
        "fault_enum": event.fault_type.value,
        "step": STATE.step_count,
        "magnitude": float(event.magnitude),
    }
    await _broadcast(payload)
    return payload


@app.get("/status")
async def status():
    return _state_snapshot()


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    STATE.ws_clients.add(ws)
    try:
        await ws.send_json({"type": "hello", **_state_snapshot()})
        while True:
            # We don't expect commands over the socket, but we drain incoming
            # messages so the connection isn't closed by the browser side.
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        LOG.warning("WebSocket error: %s", exc)
    finally:
        STATE.ws_clients.discard(ws)
