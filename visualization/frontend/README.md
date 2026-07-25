# TITAN Earth-Satellite Visualization - Frontend

3D visualization of satellite orbiting Earth with real-time Causal RL fault recovery.

## Setup

```bash
npm install
npm run dev
```

## Textures

For best visuals, download Earth textures from NASA Visible Earth:
- https://visibleearth.nasa.gov/collection/1484/blue-marble

Place in `public/textures/`:
- `earth_day.jpg` - Daytime Earth texture
- `earth_night.jpg` - City lights at night  
- `earth_clouds.jpg` - Cloud layer
- `earth_bump.jpg` - Surface bump map
- `earth_specular.jpg` - Ocean specular map

The app will work without these (using procedural textures).

## Architecture

```
src/
├── main.ts              # Entry point, scene setup
├── earth/
│   └── Earth.ts         # Earth with atmosphere, clouds
├── satellite/
│   └── Satellite.ts     # Satellite model and fault effects
├── ui/
│   └── TelemetryPanel.ts # HUD updates
└── connection/
    └── WebSocketClient.ts # Backend communication
```
