# TITAN Earth-Satellite 3D Visualization

Photorealistic 3D simulation of a satellite orbiting Earth with real-time Causal RL fault recovery.

## Features

- 🌍 **Photorealistic Earth** - Atmosphere, clouds, day/night cycle
- 🛰️ **Detailed Satellite** - Solar panels, antenna, thermal radiators
- ⚡ **Fault Visualization** - SEU sparkles, thermal glow, power surges
- 🤖 **Causal RL Agent** - Pure do-calculus decisions displayed live
- 📊 **Telemetry Dashboard** - All 13 observation variables

## Quick Start

```bash
# 1. Create directories and files
python setup_visualization.py

# 2. Install frontend dependencies
cd visualization/frontend
npm install

# 3. Start backend (in one terminal)
cd visualization/backend
python server.py

# 4. Start frontend (in another terminal)
cd visualization/frontend
npm run dev

# 5. Open http://localhost:5173
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Browser (Three.js)                     │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐   │
│  │  Earth   │  │ Satellite │  │ Telemetry Dashboard│   │
│  └──────────┘  └──────────┘  └─────────────────────┘   │
└──────────────────────WebSocket───────────────────────────┘
                           │
               ┌───────────▼───────────┐
               │   Python Backend      │
               │  - TITAN Env          │
               │  - Causal RL Policy   │
               └───────────────────────┘
```

## Controls

- **⏸ Pause/Resume** - Stop/start simulation
- **1x Speed** - Cycle through speeds (0.5x, 1x, 2x, 4x)
- **↺ Reset** - Start new episode
- **⚡ Inject** - Manually trigger fault

## Satellite Subsystems

| Component | Visual | TITAN Variables |
|-----------|--------|-------------------|
| Solar Panels | Blue wings | battery_soc, current_draw |
| Main Body | Gold foil | cpu_temperature |
| CPU Module | Green glow | cpu_load, seu_flag |
| Power Module | Blue glow | voltage, battery_soc |
| Antenna | Silver dish | signal_stability |

## Fault Effects

| Fault | Visual Effect |
|-------|---------------|
| SEU | Blue sparkle particles |
| Latch-up | Yellow power surge |
| Thermal Runaway | Orange/red glow |
| Memory Fault | Green flicker |
| Power Fault | Yellow flashes |
