import { WebSocketClient } from "./connection/WebSocketClient";
import { trendArrow } from "./utils/trends";
import { wsBackendUrl } from "./utils/wsBackendUrl";

const fmt = (n: number) => n.toFixed(2);

function toneClass(msg: Record<string, unknown>): string {
  const name = String(msg.action_name || msg.action || "").toUpperCase();
  if (msg.terminated && !msg.truncated) return "tone-red";
  if (name.includes("DIAGNOSE")) return "tone-blue";
  if (name === "NO_ACTION" || name === "") return "tone-gray";
  if (name.includes("INJECT")) return "tone-red";
  const id = typeof msg.action_id === "number" ? msg.action_id : null;
  if (id !== null && id > 0 && id <= 6) return "tone-green";
  if (id === -1) return "tone-red";
  return "tone-gray";
}

function uncText(u: unknown): string {
  if (!u || typeof u !== "object") return "—";
  const o = u as Record<string, unknown>;
  const parts = [o.fault, o.confidence].filter(Boolean).map(String);
  return parts.length ? parts.join(", ") : "—";
}

function telemetryFrom(msg: Record<string, unknown>) {
  const obs = msg.observation;
  if (obs && typeof obs === "object" && "telemetry" in obs) {
    const t = (obs as { telemetry?: Record<string, number> }).telemetry;
    if (t) return t;
  }
  const top = msg.telemetry;
  if (top && typeof top === "object") return top as Record<string, number>;
  return null;
}

function render(msg: Record<string, unknown>): void {
  const root = document.getElementById("decision-root");
  const elH = document.getElementById("el-header");
  const elO = document.getElementById("el-obs");
  const elU = document.getElementById("el-unc");
  const elA = document.getElementById("el-action");
  const elR = document.getElementById("el-reason");
  if (!root || !elH || !elO || !elU || !elA || !elR) return;

  root.className = toneClass(msg);
  const step = typeof msg.step === "number" ? msg.step : "—";
  const act = String(msg.action_name || msg.action || "—");
  elH.textContent = `STEP ${step}  [${act}]`;

  const tel = telemetryFrom(msg);
  if (tel) {
    const c = tel.cpu_temperature ?? 0;
    const p = tel.power_temperature ?? 0;
    const b = tel.battery_soc ?? 0;
    elO.textContent =
      `cpu_temp ${fmt(c)} ${trendArrow("cpu_temp", c)}\n` +
      `power_temp ${fmt(p)} ${trendArrow("power_temp", p)}\n` +
      `battery ${fmt(b)} ${trendArrow("battery", b)}`;
  } else {
    elO.textContent = "—";
  }

  elU.textContent = uncText(msg.uncertainty);
  elA.textContent = act;
  elR.textContent = String(msg.reason ?? "—");
}

const ws = new WebSocketClient(wsBackendUrl());
const conn = document.getElementById("conn");

ws.onConnect(() => {
  if (conn) {
    conn.textContent = "Live";
    conn.classList.add("live");
  }
});

ws.onDisconnect(() => {
  if (conn) {
    conn.textContent = "Disconnected";
    conn.classList.remove("live");
  }
});

ws.onMessage((data) => {
  if (data && typeof data === "object") render(data as Record<string, unknown>);
});
