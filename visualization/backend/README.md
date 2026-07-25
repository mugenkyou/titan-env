# TITAN Earth-Satellite Visualization - Backend

FastAPI WebSocket server running TITAN environment with Causal RL.

## Setup

```bash
pip install fastapi uvicorn websockets
python server.py
```

## Endpoints

- `GET /` - Health check
- `WebSocket /ws` - Real-time simulation stream

## Protocol

### Server -> Client (JSON)
```json
{
    "step": 42,
    "episode": 1,
    "telemetry": {
        "battery_soc": 0.85,
        "cpu_temperature": 0.32,
        ...
    },
    "faults": {
        "seu": false,
        "latchup": false,
        ...
    },
    "action": 3,
    "action_name": "LOAD_SHEDDING",
    "reason": "thermal_prevention",
    "reward": 0.87,
    "terminated": false
}
```

### Client -> Server (JSON)
```json
{"action": "reset", "profile": "medium"}
{"action": "inject_fault", "fault_type": "SEU"}
{"action": "set_speed", "speed": 2.0}
{"action": "step", "action_id": 5}
```
