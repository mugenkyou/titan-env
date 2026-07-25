/**
 * TITAN Earth-Satellite 3D Visualization
 * Main entry point with Causal RL Fault Recovery Chart
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { createEarth, updateEarth } from "./earth/Earth";
import {
  createSatellite,
  updateSatellite,
  SatelliteState,
} from "./satellite/Satellite";
import { WebSocketClient } from "./connection/WebSocketClient";
import {
  updateTelemetryUI,
  updateFaultUI,
  addActionLog,
  updateStatsUI,
} from "./ui/TelemetryPanel";
import { initChart, addChartData, resetChart } from "./ui/FaultRecoveryChart";

// Scene setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  10000,
);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.getElementById("app")!.appendChild(renderer.domElement);

// Camera position - moved back for bigger satellite
camera.position.set(0, 2.5, 6);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 2.5;
controls.maxDistance = 25;

// Lighting - enhanced for better satellite visibility
const ambientLight = new THREE.AmbientLight(0x8090a0, 1.2); // Brighter ambient
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffffff, 3.0);
sunLight.position.set(5, 3, 5);
scene.add(sunLight);

// Strong fill light from opposite side for shadow areas
const fillLight = new THREE.DirectionalLight(0x6688cc, 1.0);
fillLight.position.set(-5, -2, -3);
scene.add(fillLight);

// Rim light for satellite visibility
const rimLight = new THREE.DirectionalLight(0xaaccff, 1.2);
rimLight.position.set(0, 5, -5);
scene.add(rimLight);

// Additional front light to illuminate satellite
const frontLight = new THREE.DirectionalLight(0xffffff, 0.8);
frontLight.position.set(0, 0, 10);
scene.add(frontLight);

// Hemisphere light for natural sky/ground lighting
const hemiLight = new THREE.HemisphereLight(0xaabbff, 0x444466, 0.6);
scene.add(hemiLight);

// ============ SUN ============
// Create a glowing Sun far away
const sunGroup = new THREE.Group();

// Sun sphere (emissive)
const sunGeometry = new THREE.SphereGeometry(15, 64, 64);
const sunMaterial = new THREE.MeshBasicMaterial({
  color: 0xffdd44,
});
const sunMesh = new THREE.Mesh(sunGeometry, sunMaterial);
sunGroup.add(sunMesh);

// Sun glow (inner)
const sunGlowGeometry = new THREE.SphereGeometry(18, 32, 32);
const sunGlowMaterial = new THREE.ShaderMaterial({
  uniforms: {
    glowColor: { value: new THREE.Color(0xffaa00) },
    intensity: { value: 1.5 },
  },
  vertexShader: `
        varying vec3 vNormal;
        void main() {
            vNormal = normalize(normalMatrix * normal);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
  fragmentShader: `
        uniform vec3 glowColor;
        uniform float intensity;
        varying vec3 vNormal;
        void main() {
            float glow = pow(0.6 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
            gl_FragColor = vec4(glowColor * intensity, glow * 0.8);
        }
    `,
  transparent: true,
  blending: THREE.AdditiveBlending,
  side: THREE.BackSide,
});
const sunGlow = new THREE.Mesh(sunGlowGeometry, sunGlowMaterial);
sunGroup.add(sunGlow);

// Sun corona (outer glow)
const coronaGeometry = new THREE.SphereGeometry(25, 32, 32);
const coronaMaterial = new THREE.ShaderMaterial({
  uniforms: {
    glowColor: { value: new THREE.Color(0xff6600) },
  },
  vertexShader: `
        varying vec3 vNormal;
        void main() {
            vNormal = normalize(normalMatrix * normal);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
  fragmentShader: `
        uniform vec3 glowColor;
        varying vec3 vNormal;
        void main() {
            float intensity = pow(0.4 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.0);
            gl_FragColor = vec4(glowColor, intensity * 0.4);
        }
    `,
  transparent: true,
  blending: THREE.AdditiveBlending,
  side: THREE.BackSide,
});
const corona = new THREE.Mesh(coronaGeometry, coronaMaterial);
sunGroup.add(corona);

// Position Sun far away
sunGroup.position.set(150, 50, 150);
scene.add(sunGroup);

// ============ RADIATION PARTICLES ============
// Create radiation particle system for solar radiation visualization
const RADIATION_COUNT = 2000;
const radiationGeometry = new THREE.BufferGeometry();
const radiationPositions = new Float32Array(RADIATION_COUNT * 3);
const radiationVelocities = new Float32Array(RADIATION_COUNT * 3);
const radiationColors = new Float32Array(RADIATION_COUNT * 3);
const radiationSizes = new Float32Array(RADIATION_COUNT);

// Initialize radiation particles
for (let i = 0; i < RADIATION_COUNT; i++) {
  // Start from Sun direction
  const t = Math.random();
  radiationPositions[i * 3] = 100 - t * 150; // x: from Sun toward Earth
  radiationPositions[i * 3 + 1] = 30 + (Math.random() - 0.5) * 60; // y: spread
  radiationPositions[i * 3 + 2] = 100 - t * 150; // z: from Sun toward Earth

  // Velocity toward Earth
  radiationVelocities[i * 3] = -0.8 - Math.random() * 0.4;
  radiationVelocities[i * 3 + 1] = (Math.random() - 0.5) * 0.2;
  radiationVelocities[i * 3 + 2] = -0.8 - Math.random() * 0.4;

  // Color: yellow/orange/white mix
  const colorChoice = Math.random();
  if (colorChoice < 0.4) {
    radiationColors[i * 3] = 1.0; // R
    radiationColors[i * 3 + 1] = 0.8; // G
    radiationColors[i * 3 + 2] = 0.2; // B - yellow
  } else if (colorChoice < 0.7) {
    radiationColors[i * 3] = 1.0;
    radiationColors[i * 3 + 1] = 0.5;
    radiationColors[i * 3 + 2] = 0.1; // orange
  } else {
    radiationColors[i * 3] = 1.0;
    radiationColors[i * 3 + 1] = 1.0;
    radiationColors[i * 3 + 2] = 0.8; // white-ish
  }

  radiationSizes[i] = 0.3 + Math.random() * 0.5;
}

radiationGeometry.setAttribute(
  "position",
  new THREE.BufferAttribute(radiationPositions, 3),
);
radiationGeometry.setAttribute(
  "color",
  new THREE.BufferAttribute(radiationColors, 3),
);
radiationGeometry.setAttribute(
  "size",
  new THREE.BufferAttribute(radiationSizes, 1),
);

const radiationMaterial = new THREE.PointsMaterial({
  size: 0.4,
  vertexColors: true,
  transparent: true,
  opacity: 0.8,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

const radiationParticles = new THREE.Points(
  radiationGeometry,
  radiationMaterial,
);
radiationParticles.visible = false; // Hidden by default (profile = "none")
scene.add(radiationParticles);

// Radiation intensity settings per profile
const RADIATION_SETTINGS: Record<
  string,
  { visible: boolean; speed: number; density: number; color: THREE.Color }
> = {
  none: {
    visible: false,
    speed: 0,
    density: 0,
    color: new THREE.Color(0xffaa00),
  },
  low: {
    visible: true,
    speed: 0.3,
    density: 0.3,
    color: new THREE.Color(0xffdd66),
  },
  medium: {
    visible: true,
    speed: 0.6,
    density: 0.6,
    color: new THREE.Color(0xffaa33),
  },
  high: {
    visible: true,
    speed: 1.0,
    density: 0.85,
    color: new THREE.Color(0xff6600),
  },
  storm: {
    visible: true,
    speed: 1.8,
    density: 1.0,
    color: new THREE.Color(0xff3300),
  },
};

let currentRadiationSettings = RADIATION_SETTINGS["none"];

// Function to update radiation based on profile
function updateRadiationProfile(profile: string) {
  currentRadiationSettings =
    RADIATION_SETTINGS[profile] || RADIATION_SETTINGS["none"];
  radiationParticles.visible = currentRadiationSettings.visible;

  // Update corona color based on activity
  if (coronaMaterial.uniforms) {
    coronaMaterial.uniforms.glowColor.value = currentRadiationSettings.color;
  }

  console.log(
    `☀️ Radiation profile: ${profile} - visible=${currentRadiationSettings.visible}`,
  );
}

// Stars background
const starsGeometry = new THREE.BufferGeometry();
const starPositions = new Float32Array(10000 * 3);
for (let i = 0; i < starPositions.length; i += 3) {
  const r = 500 + Math.random() * 500;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  starPositions[i] = r * Math.sin(phi) * Math.cos(theta);
  starPositions[i + 1] = r * Math.sin(phi) * Math.sin(theta);
  starPositions[i + 2] = r * Math.cos(phi);
}
starsGeometry.setAttribute(
  "position",
  new THREE.BufferAttribute(starPositions, 3),
);
const starsMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.5 });
const stars = new THREE.Points(starsGeometry, starsMaterial);
scene.add(stars);

// Create Earth and Satellite
const earth = createEarth();
scene.add(earth);

const satellite = createSatellite();
scene.add(satellite.group);

// Satellite state
let satelliteState: SatelliteState = {
  telemetry: {
    battery_soc: 0.8,
    cpu_temperature: 0.3,
    power_temperature: 0.25,
    cpu_load: 0.4,
    current_draw: 0.35,
    memory_integrity: 1.0,
    signal_stability: 0.9,
    voltage: 0.85,
  },
  faults: {
    seu: false,
    latchup: false,
    thermal: false,
    memory: false,
    power: false,
  },
  action: null,
  step: 0,
  episode: 1,
  totalReward: 0,
  faultsRecovered: 0,
};

// Simulation state
let isPaused = false;
let simSpeed = 1;
let orbitAngle = 0;
let currentProfile = "none";

const BGM_URL = "/audio/space.bg.mp3";
let bgmAudio: HTMLAudioElement | null = null;

function getOrCreateBackgroundAudio(): HTMLAudioElement {
  if (bgmAudio) {
    return bgmAudio;
  }

  const audio = new Audio(BGM_URL);
  audio.loop = true;
  audio.preload = "auto";
  audio.volume = 0.45;

  audio.addEventListener("error", () => {
    console.warn("Failed to load background music file:", BGM_URL);
  });

  bgmAudio = audio;
  return audio;
}

async function toggleBackgroundMusic() {
  const btn = document.getElementById("btn-music");
  if (!btn) return;

  try {
    const audio = getOrCreateBackgroundAudio();

    if (audio.paused) {
      await audio.play();
      btn.textContent = "🎵";
      btn.setAttribute("title", "Pause Music");
    } else {
      audio.pause();
      btn.textContent = "🔇";
      btn.setAttribute("title", "Play Music");
    }
  } catch (error) {
    console.warn("Could not toggle background music.", error);
    btn.textContent = "🔇";
    btn.setAttribute("title", "Music Unavailable");
  }
}

// Initialize the fault recovery chart
initChart();

// WebSocket connection
const wsClient = new WebSocketClient("ws://localhost:8000/ws");

wsClient.onMessage((data) => {
  // Debug: log all incoming data
  if (data.faults) {
    const activeFaults = Object.entries(data.faults)
      .filter(([k, v]) => v)
      .map(([k]) => k);
    if (activeFaults.length > 0) {
      console.log("🔴 Active faults:", activeFaults, data.faults);
    }
  }

  // Update satellite state from backend
  if (data.telemetry) {
    // Merge telemetry, keeping defaults for missing fields
    satelliteState.telemetry = {
      ...satelliteState.telemetry,
      ...data.telemetry,
      // Add signal_stability as derived from memory_integrity if not provided
      signal_stability:
        data.telemetry.signal_stability ??
        data.telemetry.memory_integrity ??
        0.9,
    };
  }
  if (data.faults) {
    satelliteState.faults = data.faults;
  }

  // Track fault injection for chart
  let injectedFaultType: string | undefined = undefined;

  if (data.action !== undefined) {
    satelliteState.action = data.action;
    const actionName = data.action_name || "";

    // Check if this is a fault injection (special action -1 or name contains INJECTED)
    if (actionName.includes("INJECTED") || data.action === -1) {
      injectedFaultType = actionName
        .replace("⚡ ", "")
        .replace(" INJECTED", "");
      addActionLog(
        data.step,
        data.action_name || data.action,
        data.reason || "",
      );
      console.log(`🔴 FAULT INJECTED: ${injectedFaultType}`);
    }
    // Check if this is a recovery action (actions 1-6) — log it but DO NOT
    // increment the recovered counter here. The backend is the source of
    // truth and now broadcasts `faults_recovered` (only credited when the
    // action actually matches the active fault type).
    else if (data.action > 0 && data.action <= 6) {
      addActionLog(
        data.step,
        data.action_name || data.action,
        data.reason || "causal_rl",
      );
      console.log(`🟢 RECOVERY ACTION: ${actionName} (action=${data.action})`);
    }
    // Skip logging NO_ACTION to reduce spam
  }
  if (data.step !== undefined) {
    satelliteState.step = data.step;
  }
  if (data.reward !== undefined) {
    satelliteState.totalReward += data.reward;
  }
  if (data.episode !== undefined) {
    satelliteState.episode = data.episode;
  }
  if (typeof data.faults_recovered === "number") {
    satelliteState.faultsRecovered = data.faults_recovered;
  }
  if (data.terminated && !data.truncated) {
    // Satellite failed - reset
    satelliteState.totalReward = 0;
    satelliteState.faultsRecovered = 0;
    resetChart();
  }

  // Add data to the Causal RL chart on EVERY step
  if (data.step !== undefined) {
    const faultCount = Object.values(data.faults || {}).filter((v) => v).length;
    const isRecoveryAction =
      data.action !== undefined && data.action > 0 && data.action <= 6;
    const reward = data.reward || 0;

    // Debug logging for chart data
    if (faultCount > 0 || isRecoveryAction) {
      console.log(
        `📊 Chart data: step=${data.step} faults=${faultCount} recovery=${isRecoveryAction} reward=${reward.toFixed(2)}`,
      );
    }

    addChartData(
      data.step,
      faultCount,
      isRecoveryAction,
      reward,
      injectedFaultType,
    );
  }

  // Debug: log telemetry changes
  if (data.step && data.step % 50 === 0) {
    console.log(
      `[Step ${data.step}] Telemetry:`,
      data.telemetry,
      "Faults:",
      data.faults,
    );
  }
});

wsClient.onConnect(() => {
  console.log("Connected to TITAN backend");
  hideLoading();
});

wsClient.onDisconnect(() => {
  console.log("Disconnected from backend - running in demo mode");
  hideLoading();
});

// Demo mode - simulate data if no backend (NO RANDOM FAULTS - only manual)
let demoInterval: number;
function startDemoMode() {
  console.log("⚠️ Running in DEMO MODE - backend not connected");
  demoInterval = setInterval(() => {
    if (isPaused) return;

    // Only simulate telemetry changes, NO random faults
    satelliteState.telemetry.battery_soc = Math.max(
      0.1,
      satelliteState.telemetry.battery_soc - 0.0005 * simSpeed,
    );
    satelliteState.telemetry.cpu_temperature =
      0.3 + Math.sin(Date.now() / 5000) * 0.05;
    satelliteState.telemetry.cpu_load =
      0.4 + Math.sin(Date.now() / 3000) * 0.05;

    // NO RANDOM FAULTS - faults only through manual injection
    // Faults are injected via the inject buttons which send to backend

    satelliteState.step++;
  }, 100) as unknown as number;
}

// Hide loading screen
function hideLoading() {
  const loading = document.getElementById("loading");
  if (loading) {
    loading.classList.add("hidden");
  }
  // Start demo mode if no backend
  setTimeout(() => {
    if (!wsClient.isConnected()) {
      startDemoMode();
    }
  }, 2000);
}

// UI Controls
document.getElementById("btn-pause")?.addEventListener("click", () => {
  isPaused = !isPaused;
  const btn = document.getElementById("btn-pause");
  if (btn) btn.textContent = isPaused ? "▶" : "⏸";
  wsClient.send({ action: isPaused ? "pause" : "resume" });
});

document.getElementById("btn-speed")?.addEventListener("click", () => {
  const speeds = [0.5, 1, 2, 4];
  const idx = speeds.indexOf(simSpeed);
  simSpeed = speeds[(idx + 1) % speeds.length];
  const btn = document.getElementById("btn-speed");
  if (btn) btn.textContent = simSpeed + "x";
  wsClient.send({ action: "set_speed", speed: simSpeed });
});

document.getElementById("btn-reset")?.addEventListener("click", () => {
  wsClient.send({ action: "reset", profile: currentProfile });
  satelliteState.step = 0;
  satelliteState.totalReward = 0;
  satelliteState.faultsRecovered = 0;
  satelliteState.episode++;
  resetChart();
});

document.getElementById("btn-music")?.addEventListener("click", () => {
  void toggleBackgroundMusic();
});

// Profile selector
document.getElementById("profile-select")?.addEventListener("change", (e) => {
  const select = e.target as HTMLSelectElement;
  currentProfile = select.value;
  console.log(`🌌 Radiation profile changed to: ${currentProfile}`);

  // Update radiation visual effects
  updateRadiationProfile(currentProfile);

  // Reset with new profile
  wsClient.send({ action: "reset", profile: currentProfile });
  satelliteState.step = 0;
  satelliteState.totalReward = 0;
  satelliteState.faultsRecovered = 0;
  resetChart();
});

// Individual fault injection buttons
document.getElementById("btn-inject-seu")?.addEventListener("click", () => {
  wsClient.send({ action: "inject_fault", fault_type: "SEU" });
  console.log("🔵 Injecting SEU fault");
});

document.getElementById("btn-inject-latchup")?.addEventListener("click", () => {
  wsClient.send({ action: "inject_fault", fault_type: "LATCH_UP" });
  console.log("🔴 Injecting LATCH_UP fault");
});

document.getElementById("btn-inject-thermal")?.addEventListener("click", () => {
  wsClient.send({ action: "inject_fault", fault_type: "THERMAL" });
  console.log("🟠 Injecting THERMAL fault");
});

document.getElementById("btn-inject-memory")?.addEventListener("click", () => {
  wsClient.send({ action: "inject_fault", fault_type: "MEMORY" });
  console.log("🟢 Injecting MEMORY fault");
});

document.getElementById("btn-inject-power")?.addEventListener("click", () => {
  wsClient.send({ action: "inject_fault", fault_type: "POWER" });
  console.log("🟡 Injecting POWER fault");
});

// Animation loop
function animate() {
  requestAnimationFrame(animate);

  const delta = 0.016 * simSpeed;
  const time = Date.now() * 0.001;

  if (!isPaused) {
    // Update orbit
    orbitAngle += delta * 0.1;
    const orbitRadius = 2.5;
    satellite.group.position.x = Math.cos(orbitAngle) * orbitRadius;
    satellite.group.position.z = Math.sin(orbitAngle) * orbitRadius;
    satellite.group.position.y = Math.sin(orbitAngle * 0.5) * 0.3;

    // Face direction of travel
    satellite.group.lookAt(
      satellite.group.position.x - Math.sin(orbitAngle),
      0,
      satellite.group.position.z + Math.cos(orbitAngle),
    );

    // Update visuals
    updateEarth(earth, delta);
    updateSatellite(satellite, satelliteState, delta);

    // ============ ANIMATE RADIATION PARTICLES ============
    if (currentRadiationSettings.visible && radiationParticles.visible) {
      const positions = radiationGeometry.attributes.position
        .array as Float32Array;
      const speed = currentRadiationSettings.speed;
      const density = currentRadiationSettings.density;

      for (let i = 0; i < RADIATION_COUNT; i++) {
        // Only animate a portion based on density
        if (Math.random() > density) continue;

        // Move particles toward Earth
        positions[i * 3] += radiationVelocities[i * 3] * speed;
        positions[i * 3 + 1] += radiationVelocities[i * 3 + 1] * speed;
        positions[i * 3 + 2] += radiationVelocities[i * 3 + 2] * speed;

        // Reset particles that reach Earth area
        const dist = Math.sqrt(
          positions[i * 3] ** 2 +
            positions[i * 3 + 1] ** 2 +
            positions[i * 3 + 2] ** 2,
        );

        if (dist < 5 || positions[i * 3] < -50) {
          // Respawn from Sun direction
          positions[i * 3] = 80 + Math.random() * 40;
          positions[i * 3 + 1] = (Math.random() - 0.5) * 60;
          positions[i * 3 + 2] = 80 + Math.random() * 40;
        }
      }

      radiationGeometry.attributes.position.needsUpdate = true;
    }

    // Animate Sun glow pulsing
    sunMesh.scale.setScalar(1 + Math.sin(time * 2) * 0.02);
  }

  // Update UI
  updateTelemetryUI(satelliteState.telemetry);
  updateFaultUI(satelliteState.faults);
  updateStatsUI(satelliteState);

  controls.update();
  renderer.render(scene, camera);
}

// Handle resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Start
animate();
hideLoading();

console.log("🛰️ TITAN Earth-Satellite Simulation initialized");
