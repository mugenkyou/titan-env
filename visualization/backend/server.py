"""
TITAN Earth-Satellite Visualization - Backend Server
FastAPI WebSocket server that runs the TITAN environment with Causal RL.

Run: python server.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Set

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

# Import TITAN components
from titan_env.core.environment import FaultInjector, INTENSITY_PROFILES, RadiationProfile
from titan_env.core.environment.actions import ActionType
from titan_env.core.environment.state_model import OBS_KEYS
from titan_env.core.environment.titan_env import TITANEnv
from titan_env.core.rewards.reward_v2 import compute_reward as compute_reward_v2

# Create a "NO RANDOM FAULTS" profile - faults ONLY through manual injection
NO_RANDOM_FAULTS_PROFILE = RadiationProfile(
    radiation_intensity=0.0,  # No radiation
    p_seu=0.0,                # Zero SEU probability
    p_latchup=0.0,            # Zero latchup probability
    p_telemetry=0.0,          # Zero telemetry noise
    seu_mag_range=(0.0, 0.0),
    latchup_drain_range=(0.0, 0.0),
    latchup_heat_range=(0.0, 0.0),
    telemetry_noise_max=0.0,
    base_seu_rate=0.0,        # Zero base rates
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

# No bundled causal policy in titan_env by default; fallback heuristic is used.
causal_policy = None


class TITANVisualizationEnv:
    """Thin wrapper exposing gym-like reset/step over TITANEnv for visualization."""

    def __init__(
        self,
        fault_injector: Optional[FaultInjector] = None,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        self._injector = fault_injector
        self._core = TITANEnv(
            fault_injector=fault_injector,
            max_steps=max_steps,
            seed=seed,
        )
        self._radiation_intensity = self._compute_radiation_intensity()

    @property
    def core(self) -> TITANEnv:
        return self._core

    def _compute_radiation_intensity(self) -> float:
        if self._injector is None:
            return 0.0
        return float(getattr(self._injector.profile, "radiation_intensity", 0.0))

    def _obs_to_array(self, obs_dict: dict) -> np.ndarray:
        return np.array([float(obs_dict.get(k, 0.0)) for k in OBS_KEYS], dtype=np.float32)

    def reset(self):
        obs_dict = self._core.reset()
        return self._obs_to_array(obs_dict), {"step": 0}

    def step(self, action: int):
        obs_dict, done, core_info = self._core.step(action)
        reason = str(core_info.get("failure_reason", ""))
        terminated = bool(done and not reason.startswith("MAX_STEPS"))
        truncated = bool(done and reason.startswith("MAX_STEPS"))
        reward, reward_components = compute_reward_v2(obs_dict, action, terminated)
        info = {
            **core_info,
            "terminated": terminated,
            "truncated": truncated,
            "reward": float(reward),
            **reward_components,
        }
        return self._obs_to_array(obs_dict), float(reward), terminated, truncated, info

app = FastAPI(title="TITAN Visualization Backend")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationManager:
    """Manages the TITAN simulation state."""
    
    def __init__(self):
        self.env: Optional[TITANVisualizationEnv] = None
        self.obs: Optional[np.ndarray] = None
        self.step_count = 0
        self.episode = 1
        self.total_reward = 0
        self.running = True  # Start running immediately
        self.speed = 4.0     # Default to 4x speed for faster simulation
        self.radiation_profile = "none"  # NO random faults by default
        self._fault_hold_steps = 0  # Steps to hold before allowing recovery
        
    def reset(self, profile: str = "none"):
        """Reset the environment.
        
        profile options:
        - "none": NO random faults (only manual injection) - DEFAULT
        - "low", "medium", "high", "storm": Increasing random fault rates
        """
        self.radiation_profile = profile
        
        # Use NO_RANDOM_FAULTS_PROFILE by default - faults only via manual injection
        if profile == "none":
            fault_injector = FaultInjector(NO_RANDOM_FAULTS_PROFILE)
            print("✓ Using NO RANDOM FAULTS mode - faults only via manual injection")
        else:
            fault_injector = FaultInjector(INTENSITY_PROFILES.get(profile, INTENSITY_PROFILES["medium"]))
            print(f"⚠ Using {profile} radiation profile - random faults enabled")
        
        self.env = TITANVisualizationEnv(
            fault_injector=fault_injector,
        )
        self.obs, info = self.env.reset()
        self.step_count = 0
        self.total_reward = 0
        return self._get_state(info)

    def set_profile(self, profile: str):
        """Update radiation profile on-the-fly without resetting episode state."""
        self.radiation_profile = profile

        if self.env is None:
            self.reset(profile)
            return

        if profile == "none":
            fault_injector = FaultInjector(NO_RANDOM_FAULTS_PROFILE)
        else:
            fault_injector = FaultInjector(INTENSITY_PROFILES.get(profile, INTENSITY_PROFILES["medium"]))

        # Keep current episode/state, only swap injector for subsequent steps.
        self.env._injector = fault_injector
        self.env.core._injector = fault_injector
        self.env._radiation_intensity = self.env._compute_radiation_intensity()
    
    def step(self, action: Optional[int] = None):
        """Execute one step with optional action override."""
        if self.env is None:
            self.reset()
        
        # During fault hold period, force NO_ACTION so fault is visible
        if self._fault_hold_steps > 0:
            self._fault_hold_steps -= 1
            action = 0  # Force NO_ACTION during hold
            decision_info = {"reason": "fault_visible"}
        # Get action from Causal RL policy if no override
        elif action is None:
            if causal_policy is not None:
                core_state = self.env.core.state if hasattr(self.env, 'core') else None
                action, decision_info = causal_policy.select_action(self.obs, core_state)
            else:
                # Heuristic fallback
                action = self._heuristic_action()
                decision_info = {"reason": "heuristic"}
        else:
            decision_info = {"reason": "manual_override"}
        
        # Execute step
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.step_count += 1
        self.total_reward += reward
        
        state = self._get_state(info)
        state["action"] = int(action)
        state["action_name"] = ActionType(action).name
        state["reward"] = float(reward)
        state["reason"] = decision_info.get("reason", "")
        state["terminated"] = terminated
        state["truncated"] = truncated
        
        if terminated or truncated:
            # Episode ended - either satellite failed (terminated) or max steps reached (truncated)
            self.episode += 1
            self.reset(self.radiation_profile)
        
        return state
    
    def _get_state(self, info: dict = None) -> dict:
        """Get current state as JSON-serializable dict."""
        if self.obs is None:
            return {}
        
        # Build telemetry dict from observation - OBS_KEYS order:
        # 0:voltage, 1:current_draw, 2:battery_soc, 3:cpu_temperature,
        # 4:power_temperature, 5:memory_integrity, 6:cpu_load,
        # 7:seu_flag, 8:latchup_flag, 9:thermal_fault_flag,
        # 10:memory_fault_flag, 11:power_fault_flag, 12:recent_fault_count
        
        # Direct index access for clarity
        telemetry = {
            "voltage": float(self.obs[0]),
            "current_draw": float(self.obs[1]),
            "battery_soc": float(self.obs[2]),
            "cpu_temperature": float(self.obs[3]),
            "power_temperature": float(self.obs[4]),
            "memory_integrity": float(self.obs[5]),
            "cpu_load": float(self.obs[6]),
        }
        
        faults = {
            "seu": bool(self.obs[7] > 0.5),
            "latchup": bool(self.obs[8] > 0.5),
            "thermal": bool(self.obs[9] > 0.5),
            "memory": bool(self.obs[10] > 0.5),
            "power": bool(self.obs[11] > 0.5),
        }
        
        # Debug: print when faults are active
        active_faults = [k for k,v in faults.items() if v]
        if active_faults or telemetry['memory_integrity'] < 0.5:
            print(f"[_get_state] mem_integrity={telemetry['memory_integrity']:.3f}, faults={active_faults}")
        
        # Debug: print actual values every 100 steps
        if self.step_count % 100 == 0:
            print(f"[Step {self.step_count}] battery={telemetry['battery_soc']:.3f} "
                  f"cpu_temp={telemetry['cpu_temperature']:.3f} "
                  f"mem={telemetry['memory_integrity']:.3f} "
                  f"cpu_load={telemetry['cpu_load']:.3f} "
                  f"voltage={telemetry['voltage']:.3f} "
                  f"faults={sum(faults.values())}")
        
        # Print every step briefly for debugging
        if self.step_count < 10 or self.step_count % 50 == 0:
            active_faults = [k for k,v in faults.items() if v]
            if active_faults:
                print(f"  -> Active faults: {active_faults}")
        
        radiation_intensity = 0.0
        fault_probabilities = {
            "p_seu": 0.0,
            "p_latchup": 0.0,
            "p_thermal_runaway": 0.0,
            "p_memory_corrupt": 0.0,
            "p_power_fault": 0.0,
        }

        if self.env is not None and getattr(self.env, "core", None) is not None:
            injector = getattr(self.env.core, "_injector", None)
            core_state = getattr(self.env.core, "_state", None)
            if injector is not None:
                profile_obj = injector.profile
                radiation_intensity = float(getattr(profile_obj, "radiation_intensity", 0.0))
                if core_state is not None:
                    fault_probabilities = {
                        k: float(v)
                        for k, v in injector.compute_probabilities(core_state).items()
                    }

        return {
            "step": self.step_count,
            "episode": self.episode,
            "telemetry": telemetry,
            "faults": faults,
            "totalReward": self.total_reward,
            "radiation_profile": self.radiation_profile,
            "radiation_intensity": radiation_intensity,
            "fault_probabilities": fault_probabilities,
            "recent_fault_count": float(self.obs[12]) if len(self.obs) > 12 else 0.0,
        }
    
    def _heuristic_action(self) -> int:
        """Simple heuristic fallback if no causal policy."""
        if self.obs is None:
            return 0
        
        # Check fault flags
        seu = self.obs[7] > 0.5 if len(self.obs) > 7 else False
        latchup = self.obs[8] > 0.5 if len(self.obs) > 8 else False
        thermal = self.obs[9] > 0.5 if len(self.obs) > 9 else False
        memory = self.obs[10] > 0.5 if len(self.obs) > 10 else False
        power = self.obs[11] > 0.5 if len(self.obs) > 11 else False
        
        if thermal:
            return 5  # THERMAL_THROTTLING
        if latchup or power:
            return 4  # POWER_CYCLE
        if seu or memory:
            return 2  # MEMORY_SCRUB
        
        # Check temperature
        cpu_temp = self.obs[3] if len(self.obs) > 3 else 0.3
        if cpu_temp > 0.7:
            return 5  # THERMAL_THROTTLING
        if cpu_temp > 0.5:
            return 3  # LOAD_SHEDDING
        
        return 0  # NO_ACTION
    
    def inject_fault(self, fault_type: str):
        """Manually inject a fault by modifying the environment's internal state.
        
        Based on DAG causal relationships:
        - SEU: memory_integrity ↓, cpu_health ↓ (radiation bit-flip)
        - LATCH_UP: current_draw ↑↑, power_temperature ↑, cpu_load ↑, battery drains
        - THERMAL: cpu_temperature ↑↑, power_temperature ↑, cpu_load stressed
        - MEMORY: memory_integrity ↓↓, signal_stability ↓
        - POWER: voltage ↓↓, battery_soc ↓↓, current_draw unstable (SHUTDOWN)
        """
        if self.env is None or self.obs is None:
            return
        
        # Access the internal TITAN environment
        core_env = self.env.core if hasattr(self.env, 'core') else None
        if core_env is None:
            print(f"Warning: Cannot access core env for fault injection")
            return
        
        # Pause briefly so the fault is visible before recovery
        self._fault_hold_steps = 3  # Hold fault for a few steps before allowing recovery
        
        # Modify state to EXCEED thresholds so flags persist after step()
        fault_type_upper = fault_type.upper()
        state = core_env._state
        
        print(f"\n{'='*50}")
        print(f"INJECTING {fault_type_upper} FAULT")
        print(f"Before: cpu_temp={state.cpu_temperature:.2f}, pwr_temp={state.power_temperature:.2f}, "
              f"battery={state.battery_soc:.2f}, voltage={state.voltage:.2f}")
        
        if fault_type_upper == "SEU":
            # SEU (Single Event Upset) - radiation causes bit flips
            state.seu_flag = 1
            state.memory_integrity = 0.08  # Well below 0.15 threshold
            state.cpu_health = max(0.25, state.cpu_health - 0.30)
            state.cpu_load = min(0.90, state.cpu_load + 0.25)
            state.cpu_temperature = min(0.85, state.cpu_temperature + 0.20)
            
        elif fault_type_upper == "LATCH_UP":
            # Latch-up causes parasitic current path
            state.latchup_flag = 1
            state.current_draw = 0.98  # Massive current spike
            state.cpu_load = 0.95
            state.power_temperature = 0.92  # Above 0.90 threshold
            state.cpu_temperature = 0.88
            state.battery_soc = max(0.12, state.battery_soc - 0.25)
            state.battery_level = state.battery_soc
            state.voltage = max(0.60, state.voltage - 0.15)
            
        elif fault_type_upper == "THERMAL":
            # Thermal runaway - heat cascades through system
            state.thermal_fault_flag = 1
            state.cpu_temperature = 0.96  # Well above 0.90 threshold
            state.power_temperature = 0.94
            state.cpu_load = min(0.92, state.cpu_load + 0.25)
            state.voltage = max(0.65, state.voltage - 0.12)
            state.current_draw = min(0.90, state.current_draw + 0.20)
            
        elif fault_type_upper == "MEMORY":
            # Memory corruption - severe data loss
            # Set SEU flag since physics engine damages memory based on seu_flag
            state.seu_flag = 1
            state.memory_fault_flag = 1
            state.memory_integrity = 0.03  # Severe corruption
            state.cpu_health = max(0.35, state.cpu_health - 0.25)
            state.communication_health = max(0.40, state.communication_health - 0.20)
            state.cpu_load = min(0.90, state.cpu_load + 0.20)
            state.cpu_temperature = min(0.80, state.cpu_temperature + 0.15)
            
        elif fault_type_upper == "POWER":
            # POWER FAULT - Complete system failure cascade
            state.power_fault_flag = 1
            state.battery_level = 0.03
            state.battery_soc = 0.03
            state.voltage = 0.25  # Severe voltage collapse
            state.current_draw = 0.08  # System shutting down
            state.cpu_load = 0.05
            state.cpu_temperature = max(0.15, state.cpu_temperature - 0.20)
            state.power_temperature = max(0.20, state.power_temperature - 0.15)
            state.memory_integrity = max(0.25, state.memory_integrity - 0.25)
            state.communication_health = 0.15
        
        print(f"After:  cpu_temp={state.cpu_temperature:.2f}, pwr_temp={state.power_temperature:.2f}, "
              f"battery={state.battery_soc:.2f}, voltage={state.voltage:.2f}, mem={state.memory_integrity:.2f}")
        
        # Update observation array from new state
        obs_dict = core_env._get_observation(state)
        self.obs = np.array([obs_dict[k] for k in OBS_KEYS], dtype=np.float32)
        
        print(f"✓ Injected {fault_type} - Obs: mem_integrity={self.obs[5]:.3f}")
        print(f"  Flags: seu={self.obs[7]:.0f} latch={self.obs[8]:.0f} "
              f"therm={self.obs[9]:.0f} mem={self.obs[10]:.0f} pwr={self.obs[11]:.0f}")
        print(f"{'='*50}\n")


# Global simulation manager
sim = SimulationManager()
sim_lock = asyncio.Lock()


class ConnectionManager:
    """Tracks active websocket clients and broadcasts shared simulation state."""

    def __init__(self):
        self.clients: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)
        print(f"Client connected (total={len(self.clients)})")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.clients:
            self.clients.remove(websocket)
        print(f"Client disconnected (total={len(self.clients)})")

    async def broadcast(self, payload: dict):
        if not self.clients:
            return

        stale_clients: Set[WebSocket] = set()
        for client in self.clients:
            try:
                await client.send_json(payload)
            except Exception:
                stale_clients.add(client)

        for client in stale_clients:
            self.disconnect(client)


manager = ConnectionManager()


async def simulation_broadcast_loop():
    """Single shared simulation tick loop for all connected clients."""
    while True:
        tick_hz = max(1.0, sim.speed * 20.0)
        await asyncio.sleep(1.0 / tick_hz)

        if not sim.running:
            continue

        if not manager.clients:
            continue

        async with sim_lock:
            state = sim.step()

        await manager.broadcast(state)

        has_fault = any(state.get("faults", {}).values())
        action_name = state.get("action_name", "")
        if has_fault and action_name != "NO_ACTION":
            await asyncio.sleep(0.2)


@app.on_event("startup")
async def _startup_loop():
    if sim.env is None:
        sim.reset("none")
    asyncio.create_task(simulation_broadcast_loop())


@app.get("/")
async def root():
    return {"status": "TITAN Visualization Backend running"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    async with sim_lock:
        if sim.env is None:
            sim.reset("none")
        state = sim._get_state()
    await websocket.send_json(state)
    
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "reset":
                profile = data.get("profile", "none")
                async with sim_lock:
                    state = sim.reset(profile)
                await manager.broadcast(state)

            elif data.get("action") == "inject_fault":
                fault_type = data.get("fault_type", "SEU")
                async with sim_lock:
                    sim.inject_fault(fault_type)
                    state = sim._get_state()
                    state["action"] = -1
                    state["action_name"] = f"⚡ {fault_type} INJECTED"
                    state["reason"] = "manual_injection"
                    state["reward"] = 0
                await manager.broadcast(state)

            elif data.get("action") == "set_speed":
                async with sim_lock:
                    sim.speed = float(data.get("speed", 1.0))

            elif data.get("action") == "set_profile":
                profile = data.get("profile", "none")
                async with sim_lock:
                    sim.set_profile(profile)
                    state = sim._get_state()
                await manager.broadcast(state)

            elif data.get("action") == "pause":
                async with sim_lock:
                    sim.running = False

            elif data.get("action") == "resume":
                async with sim_lock:
                    sim.running = True

            elif data.get("action") == "step":
                action = data.get("action_id")
                async with sim_lock:
                    state = sim.step(action)
                await manager.broadcast(state)
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    print("=" * 60)
    print("TITAN Earth-Satellite Visualization Backend")
    print("=" * 60)
    print()
    print("Starting server on http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
