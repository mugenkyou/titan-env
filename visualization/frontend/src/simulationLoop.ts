/**
 * Client-side AUTO stepping. Uses the existing WebSocket command { action: "step" }.
 * At most one interval is active; clear on stop or dependency change.
 */

const BASE_INTERVAL_MS = 500;

export type SimMode = "manual" | "auto";

export type WsLike = {
  send(data: object): void;
  isConnected(): boolean;
};

let intervalId: ReturnType<typeof setInterval> | null = null;
let wsRef: WsLike | null = null;
let getSpeed: () => number = () => 1;
let getMode: () => SimMode = () => "manual";

function intervalMsForSpeed(speed: number): number {
  const s = Math.max(0.25, speed);
  return Math.max(50, BASE_INTERVAL_MS / s);
}

function tick(): void {
  if (getMode() !== "auto") return;
  if (!wsRef?.isConnected()) return;
  wsRef.send({ action: "step" });
}

export function configureSimulationLoop(opts: {
  ws: WsLike;
  getSpeed: () => number;
  getMode: () => SimMode;
}): void {
  wsRef = opts.ws;
  getSpeed = opts.getSpeed;
  getMode = opts.getMode;
}

export function stopLoop(): void {
  if (intervalId !== null) {
    clearInterval(intervalId);
    intervalId = null;
  }
}

export function startLoop(): void {
  stopLoop();
  if (getMode() !== "auto") return;
  intervalId = setInterval(tick, intervalMsForSpeed(getSpeed()));
}

export function updateSpeed(): void {
  if (intervalId !== null && getMode() === "auto") {
    stopLoop();
    startLoop();
  }
}
