import { WebSocketClient } from "./connection/WebSocketClient";

type FaultKey = "seu" | "latchup" | "thermal" | "memory" | "power";

type Telemetry = {
  voltage: number;
  current_draw: number;
  battery_soc: number;
  cpu_temperature: number;
  power_temperature: number;
  memory_integrity: number;
  cpu_load: number;
};

type Faults = Record<FaultKey, boolean>;

type BackendMessage = {
  step: number;
  episode: number;
  telemetry: Telemetry;
  faults: Faults;
  action?: number;
  action_name?: string;
  reason?: string;
  reward?: number;
  totalReward?: number;
  total_reward?: number;
  radiation_profile?: string;
  radiation_intensity?: number;
  fault_probabilities?: {
    p_seu: number;
    p_latchup: number;
    p_thermal_runaway: number;
    p_memory_corrupt: number;
    p_power_fault: number;
  };
  recent_fault_count?: number;
};

type RecoveryWindow = {
  fault: FaultKey;
  startStep: number;
};

type LineSeries = {
  values: number[];
  color: string;
  width?: number;
};

type Threshold = {
  value: number;
  color: string;
};

type FaultProbabilities = {
  p_seu: number;
  p_latchup: number;
  p_thermal_runaway: number;
  p_memory_corrupt: number;
  p_power_fault: number;
};

const MAX_POINTS = 120;
const SERIES_SAMPLE_STEP = 2;
const CHART_RENDER_INTERVAL_MS = 120;
const FAULT_ORDER: FaultKey[] = [
  "seu",
  "latchup",
  "thermal",
  "memory",
  "power",
];
const FAULT_LABEL: Record<FaultKey, string> = {
  seu: "SEU",
  latchup: "Latch-up",
  thermal: "Thermal",
  memory: "Memory",
  power: "Power",
};

const history = {
  battery: [] as number[],
  voltage: [] as number[],
  current: [] as number[],
  cpuTemp: [] as number[],
  powerTemp: [] as number[],
  cpuLoad: [] as number[],
  memory: [] as number[],
  reward: [] as number[],
  totalReward: [] as number[],
  radiation: [] as number[],
  faultCount: [] as number[],
  degradation: [] as number[],
  episode: [] as number[],
  successCumulative: [] as number[],
  failureCumulative: [] as number[],
};

const stats = {
  injections: 0,
  recovered: 0,
  failed: 0,
  recoveryDurations: [] as number[],
};

const recoveryWindows = new Map<FaultKey, RecoveryWindow>();
let lastTelemetry: Telemetry | null = null;
let lastStep = 0;
let paused = false;
let spikeUntil = 0;
let activeProfile = "none";
let lastSeriesStep = -1;
let pendingChartRender = false;
let chartsDirty = false;
let lastChartRenderTs = 0;
let lastFaultProbabilities: FaultProbabilities = {
  p_seu: 0,
  p_latchup: 0,
  p_thermal_runaway: 0,
  p_memory_corrupt: 0,
  p_power_fault: 0,
};

const wsStatus = must("ws-status");
const statusStep = must("status-step");
const statusEpisode = must("status-episode");
const statusReward = must("status-reward");

const flowState = must("flow-state");
const flowAction = must("flow-action");
const flowReward = must("flow-reward");

const impactImmediate = must("impact-immediate");
const impactPropagation = must("impact-propagation");
const impactRecovery = must("impact-recovery");
const actionExplain = must("action-explain");

const faultEvents = must("fault-events");

const envRadiation = must("env-radiation");
const envCorrelation = must("env-correlation");
const envDegradation = must("env-degradation");
const envProfile = must("env-profile");

const txtRecoveryRate = must("txt-recovery-rate");
const txtRecoveryTime = must("txt-recovery-time");
const txtRatio = must("txt-ratio");
const txtStability = must("txt-stability");

const barRecoveryRate = must("bar-recovery-rate");
const barRecoveryTime = must("bar-recovery-time");
const barRatio = must("bar-ratio");
const barStability = must("bar-stability");

const nodePower = must("node-power");
const nodeThermal = must("node-thermal");
const nodeCompute = must("node-compute");
const nodePowerText = must("node-power-text");
const nodeThermalText = must("node-thermal-text");
const nodeComputeText = must("node-compute-text");

const profileSelect = mustSel("profile-select") as HTMLSelectElement;
const btnReset = mustSel("btn-reset") as HTMLButtonElement;
const btnPause = mustSel("btn-pause") as HTMLButtonElement;

const faultTypeSelect = mustSel("fault-type") as HTMLSelectElement;
const faultIntensitySelect = mustSel("fault-intensity") as HTMLSelectElement;
const faultFrequencySelect = mustSel("fault-frequency") as HTMLSelectElement;
const btnInject = mustSel("btn-inject") as HTMLButtonElement;
const btnRadiationSpike = mustSel("btn-radiation-spike") as HTMLButtonElement;

const chartPower = mustCanvas("chart-power");
const chartThermal = mustCanvas("chart-thermal");
const chartCompute = mustCanvas("chart-compute");
const chartRecovery = mustCanvas("chart-recovery");
const chartReward = mustCanvas("chart-reward");
const chartEpisode = mustCanvas("chart-episode");
const chartSuccess = mustCanvas("chart-success");
const chartOutcomes = mustCanvas("chart-outcomes");
const chartRadiation = mustCanvas("chart-radiation");
const chartDegradation = mustCanvas("chart-degradation");
const chartCausal = mustCanvas("chart-causal");

const wsClient = new WebSocketClient("ws://localhost:8000/ws");

wsClient.onConnect(() => {
  wsStatus.textContent = "ONLINE";
  wsStatus.style.color = "#23c5a8";
});

wsClient.onDisconnect(() => {
  wsStatus.textContent = "RECONNECTING";
  wsStatus.style.color = "#f6b64a";
});

wsClient.onMessage((msg: BackendMessage) => {
  handleMessage(msg);
});

profileSelect.addEventListener("change", () => {
  activeProfile = profileSelect.value;
  wsClient.send({ action: "set_profile", profile: profileSelect.value });
});

btnReset.addEventListener("click", () => {
  wsClient.send({ action: "reset", profile: profileSelect.value });
});

btnPause.addEventListener("click", () => {
  paused = !paused;
  wsClient.send({ action: paused ? "pause" : "resume" });
  btnPause.textContent = paused ? "Resume" : "Pause";
});

btnInject.addEventListener("click", () => {
  const bursts = Number(faultIntensitySelect.value);
  const spacingMs = Number(faultFrequencySelect.value);
  const faultType = faultTypeSelect.value;
  stats.injections += bursts;

  for (let idx = 0; idx < bursts; idx += 1) {
    window.setTimeout(() => {
      wsClient.send({ action: "inject_fault", fault_type: faultType });
    }, idx * spacingMs);
  }
});

btnRadiationSpike.addEventListener("click", () => {
  if (spikeUntil > Date.now()) {
    return;
  }

  const previousProfile = activeProfile;
  spikeUntil = Date.now() + 9000;
  wsClient.send({ action: "set_profile", profile: "storm" });

  window.setTimeout(() => {
    wsClient.send({ action: "set_profile", profile: previousProfile });
  }, 9000);
});

function handleMessage(msg: BackendMessage): void {
  const telemetry = msg.telemetry;
  const faults = msg.faults;
  if (!telemetry || !faults) {
    return;
  }

  lastStep = msg.step ?? lastStep;
  const totalReward = msg.totalReward ?? msg.total_reward ?? 0;
  activeProfile = msg.radiation_profile ?? activeProfile;
  if (profileSelect.value !== activeProfile) {
    profileSelect.value = activeProfile;
  }

  statusStep.textContent = String(msg.step ?? 0);
  statusEpisode.textContent = String(msg.episode ?? 1);
  statusReward.textContent = totalReward.toFixed(2);

  const faultCount = FAULT_ORDER.reduce(
    (acc, key) => acc + (faults[key] ? 1 : 0),
    0,
  );
  const recentFaultCount = msg.recent_fault_count ?? faultCount / 5;
  const radiationValue = Math.max(
    0,
    Math.min(100, (msg.radiation_intensity ?? 0) * 100),
  );

  envProfile.textContent = `${activeProfile} (backend)`;
  envRadiation.textContent = radiationValue.toFixed(1);

  const probs = msg.fault_probabilities;
  if (probs) {
    lastFaultProbabilities = probs;
    const pTotal =
      probs.p_seu +
      probs.p_latchup +
      probs.p_thermal_runaway +
      probs.p_memory_corrupt +
      probs.p_power_fault;
    impactPropagation.textContent = `p(sum)=${pTotal.toFixed(3)} from backend model`;
  }

  const degradation = clamp01(
    0.35 * (1 - telemetry.memory_integrity) +
      0.25 * telemetry.cpu_temperature +
      0.25 * telemetry.power_temperature +
      0.15 * telemetry.current_draw,
  );

  const currentStep = msg.step ?? 0;
  if (currentStep !== lastSeriesStep && currentStep % SERIES_SAMPLE_STEP === 0) {
    push(history.battery, telemetry.battery_soc);
    push(history.voltage, telemetry.voltage);
    push(history.current, telemetry.current_draw);
    push(history.cpuTemp, telemetry.cpu_temperature);
    push(history.powerTemp, telemetry.power_temperature);
    push(history.cpuLoad, telemetry.cpu_load);
    push(history.memory, telemetry.memory_integrity);
    push(history.reward, msg.reward ?? 0);
    push(history.totalReward, totalReward);
    push(history.radiation, radiationValue / 100);
    push(history.faultCount, Math.max(0, Math.min(1, recentFaultCount)));
    push(history.degradation, degradation);
    push(history.episode, (msg.episode ?? 1) / 20);
    lastSeriesStep = currentStep;
    chartsDirty = true;
  }

  trackRecoveryWindows(faults, msg.step ?? 0);
  updateFlow(
    telemetry,
    msg.action_name ?? "NO_ACTION",
    msg.reward ?? 0,
    msg.reason ?? "",
  );
  updateImpact(telemetry, faults);
  updateSubsystemMap(telemetry, faults, msg.action_name ?? "NO_ACTION");
  updateAnalytics(degradation);
  updateEnvironmentMetrics();
  scheduleChartRender();

  if (msg.action_name && msg.action_name.includes("INJECTED")) {
    addFaultEvent(`Injected ${msg.action_name}`, "critical-text");
  } else if ((msg.action_name ?? "") !== "NO_ACTION" && faultCount > 0) {
    addFaultEvent(`${msg.action_name} on active fault`, "");
  }

  lastTelemetry = telemetry;
}

function scheduleChartRender(): void {
  if (!chartsDirty || pendingChartRender) {
    return;
  }

  pendingChartRender = true;
  window.requestAnimationFrame((ts) => {
    pendingChartRender = false;
    if (!chartsDirty) {
      return;
    }

    if (ts - lastChartRenderTs < CHART_RENDER_INTERVAL_MS) {
      scheduleChartRender();
      return;
    }

    lastChartRenderTs = ts;
    chartsDirty = false;
    renderAllCharts();
  });
}

function trackRecoveryWindows(faults: Faults, step: number): void {
  for (const fault of FAULT_ORDER) {
    const isActive = faults[fault];
    const existing = recoveryWindows.get(fault);

    if (isActive && !existing) {
      recoveryWindows.set(fault, { fault, startStep: step });
    }

    if (!isActive && existing) {
      const duration = Math.max(1, step - existing.startStep);
      stats.recovered += 1;
      stats.recoveryDurations.push(duration / 20);
      recoveryWindows.delete(fault);
    }
  }

  for (const active of recoveryWindows.values()) {
    if (step - active.startStep > 120) {
      stats.failed += 1;
      recoveryWindows.delete(active.fault);
      addFaultEvent(
        `${FAULT_LABEL[active.fault]} exceeded recovery window`,
        "critical-text",
      );
    }
  }

  push(history.successCumulative, stats.recovered);
  push(history.failureCumulative, stats.failed);
}

function updateFlow(
  telemetry: Telemetry,
  actionName: string,
  reward: number,
  reason: string,
): void {
  const dominantIssue = resolveDominantIssue(telemetry);
  flowState.textContent = dominantIssue;
  flowAction.textContent = actionName;
  flowReward.textContent = reward.toFixed(3);

  const strategy = explainAction(reason, actionName, telemetry);
  actionExplain.textContent = strategy;
}

function updateImpact(telemetry: Telemetry, faults: Faults): void {
  if (!lastTelemetry) {
    impactImmediate.textContent = "Awaiting baseline";
    impactPropagation.textContent = "-";
    impactRecovery.textContent = "-";
    return;
  }

  const tempDelta = telemetry.cpu_temperature - lastTelemetry.cpu_temperature;
  const powerDelta = telemetry.voltage - lastTelemetry.voltage;
  const memDelta = telemetry.memory_integrity - lastTelemetry.memory_integrity;

  impactImmediate.textContent = `dT ${signed(tempDelta)} | dV ${signed(powerDelta)}`;

  const affected = inferSubsystems(faults, telemetry);
  if (
    impactPropagation.textContent === "-" ||
    impactPropagation.textContent === "Contained"
  ) {
    impactPropagation.textContent = affected.join(" → ") || "Contained";
  }

  const activeAges = Array.from(recoveryWindows.values()).map(
    (x) => lastStep - x.startStep,
  );
  if (activeAges.length === 0) {
    impactRecovery.textContent = "Stable";
  } else {
    const eta = Math.max(1, 30 - Math.floor(avg(activeAges)));
    impactRecovery.textContent = `${eta / 10}s`;
  }

  if (Math.abs(memDelta) > 0.08) {
    addFaultEvent(
      `Memory integrity changed ${signed(memDelta)}`,
      Math.abs(memDelta) > 0.15 ? "critical-text" : "",
    );
  }
}

function updateSubsystemMap(
  telemetry: Telemetry,
  faults: Faults,
  actionName: string,
): void {
  const powerSeverity = Math.max(
    1 - telemetry.voltage,
    telemetry.current_draw,
    faults.power ? 1 : 0,
  );
  const thermalSeverity = Math.max(
    telemetry.cpu_temperature,
    telemetry.power_temperature,
    faults.thermal ? 1 : 0,
  );
  const computeSeverity = Math.max(
    telemetry.cpu_load,
    1 - telemetry.memory_integrity,
    faults.memory || faults.seu ? 1 : 0,
  );

  applyNode(nodePower, severityClass(powerSeverity));
  applyNode(nodeThermal, severityClass(thermalSeverity));
  applyNode(nodeCompute, severityClass(computeSeverity));

  nodePowerText.textContent = `V ${telemetry.voltage.toFixed(2)} | I ${telemetry.current_draw.toFixed(2)} | ${faults.power ? "POWER FAULT" : "managed"}`;
  nodeThermalText.textContent = `CPU ${telemetry.cpu_temperature.toFixed(2)} | PWR ${telemetry.power_temperature.toFixed(2)} | ${faults.thermal ? "THERMAL FAULT" : "cooling"}`;
  nodeComputeText.textContent = `Load ${telemetry.cpu_load.toFixed(2)} | Mem ${telemetry.memory_integrity.toFixed(2)} | ${faults.memory || faults.seu ? "DATA AT RISK" : "coherent"}`;

  if (actionName !== "NO_ACTION") {
    addFaultEvent(`Action: ${actionName}`, "");
  }
}

function updateAnalytics(degradation: number): void {
  const attempts = Math.max(1, stats.recovered + stats.failed);
  const recoveryRate = (stats.recovered / attempts) * 100;
  const avgRecoveryTime =
    stats.recoveryDurations.length > 0 ? avg(stats.recoveryDurations) : 0;
  const ratioLabel = `${stats.recovered}:${stats.failed}`;
  const stability = Math.max(
    0,
    Math.min(
      100,
      Math.round(100 - degradation * 65 - (stats.failed / attempts) * 35),
    ),
  );

  txtRecoveryRate.textContent = `${recoveryRate.toFixed(1)}%`;
  txtRecoveryTime.textContent = `${avgRecoveryTime.toFixed(1)}s`;
  txtRatio.textContent = ratioLabel;
  txtStability.textContent = `${stability}`;

  setBar(barRecoveryRate, recoveryRate);
  setBar(barRecoveryTime, Math.min(100, (avgRecoveryTime / 10) * 100), true);
  setBar(
    barRatio,
    Math.min(
      100,
      (stats.recovered / Math.max(1, stats.recovered + stats.failed)) * 100,
    ),
  );
  setBar(barStability, stability);
}

function updateEnvironmentMetrics(): void {
  const faults = denormalize(history.faultCount);
  const radiation = denormalize(history.radiation);
  const degradation = denormalize(history.degradation);

  const corr = correlation(radiation, faults);
  const currentDegradation =
    degradation.length > 0 ? degradation[degradation.length - 1] : 0;

  envCorrelation.textContent = Number.isFinite(corr) ? corr.toFixed(2) : "0.00";
  envDegradation.textContent = currentDegradation.toFixed(2);
}

function renderAllCharts(): void {
  drawLineChart(
    chartPower,
    [
      { values: history.battery, color: "#4de1b2" },
      { values: history.voltage, color: "#58b8ff" },
      { values: history.current, color: "#ffd267" },
    ],
    [
      { value: 0.3, color: "rgba(255,100,110,0.45)" },
      { value: 0.6, color: "rgba(246,182,74,0.35)" },
    ],
  );

  drawLineChart(
    chartThermal,
    [
      { values: history.cpuTemp, color: "#ff8b74" },
      { values: history.powerTemp, color: "#f7c062" },
    ],
    [
      { value: 0.7, color: "rgba(246,182,74,0.35)" },
      { value: 0.9, color: "rgba(255,100,110,0.5)" },
    ],
  );

  drawLineChart(
    chartCompute,
    [
      { values: history.cpuLoad, color: "#8ec1ff" },
      { values: history.memory, color: "#3fe7c5" },
    ],
    [
      { value: 0.25, color: "rgba(255,100,110,0.45)" },
      { value: 0.5, color: "rgba(246,182,74,0.3)" },
    ],
  );

  drawLineChart(chartRecovery, [
    { values: history.faultCount, color: "#ff646e" },
    { values: history.successCumulative.map(scaleToWindow), color: "#4de1b2" },
  ]);
  drawLineChart(chartReward, [
    {
      values: history.totalReward.map(scaleReward),
      color: "#f6b64a",
      width: 2.2,
    },
  ]);
  drawLineChart(chartEpisode, [{ values: history.episode, color: "#6fb8ff" }]);
  drawLineChart(chartSuccess, [
    { values: history.successCumulative.map(scaleToWindow), color: "#4de1b2" },
    { values: history.failureCumulative.map(scaleToWindow), color: "#ff646e" },
  ]);
  drawLineChart(chartOutcomes, [
    { values: history.successCumulative.map(scaleToWindow), color: "#51b7ff" },
    { values: history.failureCumulative.map(scaleToWindow), color: "#ff8b74" },
  ]);
  drawLineChart(chartRadiation, [
    { values: history.radiation, color: "#ffd267" },
    { values: history.faultCount, color: "#ff646e" },
  ]);
  drawLineChart(chartDegradation, [
    { values: history.degradation, color: "#ff9e71" },
  ]);
  drawCausalGraph(
    chartCausal,
    lastFaultProbabilities,
    flowAction.textContent || "NO_ACTION",
  );
}

function drawLineChart(
  canvas: HTMLCanvasElement,
  series: LineSeries[],
  thresholds: Threshold[] = [],
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (width <= 0 || height <= 0) {
    return;
  }

  const targetWidth = Math.floor(width * dpr);
  const targetHeight = Math.floor(height * dpr);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.clearRect(0, 0, width, height);

  for (const thr of thresholds) {
    const y = (1 - thr.value) * height;
    ctx.strokeStyle = thr.color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  for (const line of series) {
    if (line.values.length < 2) {
      continue;
    }

    ctx.strokeStyle = line.color;
    ctx.lineWidth = line.width ?? 1.6;
    ctx.beginPath();

    const points = line.values;
    const len = points.length;
    for (let idx = 0; idx < len; idx += 1) {
      const x = (idx / Math.max(1, len - 1)) * width;
      const y = (1 - clamp01(points[idx])) * height;
      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();
  }
}

function drawCausalGraph(
  canvas: HTMLCanvasElement,
  probs: FaultProbabilities,
  actionName: string,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const targetWidth = Math.floor(width * dpr);
  const targetHeight = Math.floor(height * dpr);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pFaults = clamp01(
    probs.p_seu +
      probs.p_latchup +
      probs.p_thermal_runaway +
      probs.p_memory_corrupt +
      probs.p_power_fault,
  );
  const pThermal = clamp01(probs.p_thermal_runaway);

  const nodes = [
    { x: width * 0.12, y: height * 0.32, label: `Radiation ${activeProfile}` },
    {
      x: width * 0.38,
      y: height * 0.25,
      label: `Faults p=${pFaults.toFixed(2)}`,
    },
    {
      x: width * 0.38,
      y: height * 0.68,
      label: `Thermal p=${pThermal.toFixed(2)}`,
    },
    { x: width * 0.64, y: height * 0.42, label: "State" },
    {
      x: width * 0.86,
      y: height * 0.42,
      label: actionName.replaceAll("_", " "),
    },
  ];

  const links: [number, number][] = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [3, 4],
  ];

  ctx.strokeStyle = "rgba(124, 186, 255, 0.5)";
  ctx.lineWidth = 1.2;
  for (const [from, to] of links) {
    ctx.beginPath();
    ctx.moveTo(nodes[from].x, nodes[from].y);
    ctx.lineTo(nodes[to].x, nodes[to].y);
    ctx.stroke();
  }

  nodes.forEach((node, idx) => {
    const intensity =
      idx === 1
        ? pFaults
        : idx === 2
          ? pThermal
          : idx === 4 && actionName !== "NO_ACTION"
            ? 1
            : 0.15;
    const fillAlpha = 0.15 + 0.45 * intensity;
    ctx.fillStyle = `rgba(13, 34, 56, ${fillAlpha.toFixed(3)})`;
    ctx.strokeStyle = "rgba(104, 166, 230, 0.8)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(node.x, node.y, 19, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#d4e8ff";
    ctx.font = "11px JetBrains Mono";
    ctx.textAlign = "center";
    ctx.fillText(node.label, node.x, node.y + 34);
  });
}

function explainAction(
  reason: string,
  actionName: string,
  telemetry: Telemetry,
): string {
  const reasonMap: Record<string, string> = {
    manual_injection: "Manual fault injection issued from console.",
    manual_override: "Manual action override executed from operator control.",
    fault_visible:
      "Fault-hold mode active; policy temporarily forced to NO_ACTION.",
    heuristic:
      "Heuristic fallback policy active because causal policy is unavailable.",
  };

  if (reason && reasonMap[reason]) {
    return `${reasonMap[reason]} Current action: ${actionName.replaceAll("_", " ")}.`;
  }

  if (reason) {
    return `Policy reason: ${reason}. Current action: ${actionName.replaceAll("_", " ")}.`;
  }

  const lowMem = telemetry.memory_integrity < 0.35;
  const highTemp =
    telemetry.cpu_temperature > 0.75 || telemetry.power_temperature > 0.75;
  const powerRisk = telemetry.voltage < 0.35 || telemetry.current_draw > 0.8;

  if (actionName.includes("MEMORY")) {
    return "Memory path prioritized: corruption signal dominated expected reward penalty.";
  }
  if (actionName.includes("THERMAL")) {
    return "Thermal throttling selected: projected overheating risk exceeded compute throughput benefit.";
  }
  if (actionName.includes("POWER")) {
    return "Power-cycle selected: bus instability and current surge crossed protection threshold.";
  }
  if (lowMem || highTemp || powerRisk) {
    return "Agent in watch mode: elevated risk detected but waiting for stronger causal confirmation.";
  }
  return "Nominal policy branch active: preserving efficiency and maintaining mission stability.";
}

function resolveDominantIssue(telemetry: Telemetry): string {
  if (telemetry.memory_integrity < 0.3) {
    return "Memory Integrity Collapse";
  }
  if (telemetry.cpu_temperature > 0.85 || telemetry.power_temperature > 0.85) {
    return "Thermal Escalation";
  }
  if (telemetry.voltage < 0.32) {
    return "Power Instability";
  }
  if (telemetry.cpu_load > 0.8) {
    return "Compute Stress";
  }
  return "Nominal Trajectory";
}

function inferSubsystems(faults: Faults, telemetry: Telemetry): string[] {
  const links: string[] = [];
  if (faults.power || telemetry.voltage < 0.35) {
    links.push("Power");
  }
  if (
    faults.thermal ||
    telemetry.cpu_temperature > 0.75 ||
    telemetry.power_temperature > 0.75
  ) {
    links.push("Thermal");
  }
  if (faults.memory || faults.seu || telemetry.memory_integrity < 0.5) {
    links.push("Compute");
  }
  return links;
}

function applyNode(
  el: HTMLElement,
  state: "normal" | "warn" | "critical",
): void {
  el.classList.remove("normal", "warn", "critical");
  el.classList.add(state);
}

function severityClass(value: number): "normal" | "warn" | "critical" {
  if (value > 0.85) {
    return "critical";
  }
  if (value > 0.6) {
    return "warn";
  }
  return "normal";
}

function setBar(el: HTMLElement, percent: number, inverse = false): void {
  const p = Math.max(0, Math.min(100, percent));
  el.style.width = `${p.toFixed(1)}%`;
  el.classList.remove("warn", "critical");

  const score = inverse ? 100 - p : p;
  if (score < 45) {
    el.classList.add("critical");
  } else if (score < 70) {
    el.classList.add("warn");
  }
}

function addFaultEvent(message: string, cls: string): void {
  const item = document.createElement("div");
  item.className = `fault-item ${cls}`.trim();
  item.innerHTML = `<strong>T+${lastStep}</strong> ${message}`;
  faultEvents.prepend(item);

  while (faultEvents.children.length > 18) {
    faultEvents.removeChild(faultEvents.lastElementChild as Node);
  }
}

function signed(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}

function scaleToWindow(value: number): number {
  const max = Math.max(
    1,
    ...history.successCumulative,
    ...history.failureCumulative,
  );
  return value / max;
}

function scaleReward(value: number): number {
  const rewards = history.totalReward;
  const min = Math.min(0, ...rewards);
  const max = Math.max(1, ...rewards);
  return (value - min) / Math.max(1e-6, max - min);
}

function denormalize(values: number[]): number[] {
  return values.map((value) => clamp01(value));
}

function correlation(xs: number[], ys: number[]): number {
  if (xs.length < 2 || ys.length < 2 || xs.length !== ys.length) {
    return 0;
  }
  const avgX = avg(xs);
  const avgY = avg(ys);
  let top = 0;
  let bottomX = 0;
  let bottomY = 0;
  for (let idx = 0; idx < xs.length; idx += 1) {
    const dx = xs[idx] - avgX;
    const dy = ys[idx] - avgY;
    top += dx * dy;
    bottomX += dx * dx;
    bottomY += dy * dy;
  }
  const denom = Math.sqrt(bottomX * bottomY);
  if (denom < 1e-8) {
    return 0;
  }
  return top / denom;
}

function avg(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

function push(buffer: number[], value: number): void {
  buffer.push(clamp01(value));
  if (buffer.length > MAX_POINTS) {
    buffer.shift();
  }
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function must(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Missing element: ${id}`);
  }
  return el;
}

function mustSel(selector: string): Element {
  const el = document.getElementById(selector);
  if (!el) {
    throw new Error(`Missing element: ${selector}`);
  }
  return el;
}

function mustCanvas(id: string): HTMLCanvasElement {
  const el = document.getElementById(id);
  if (!el || !(el instanceof HTMLCanvasElement)) {
    throw new Error(`Missing canvas: ${id}`);
  }
  return el;
}

window.addEventListener("resize", () => {
  chartsDirty = true;
  scheduleChartRender();
});
