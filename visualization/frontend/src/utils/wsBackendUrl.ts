/** WebSocket URL for the TITAN viz backend (same host as the page, port 8000). */
export function wsBackendUrl(port = 8000): string {
  const host = window.location.hostname || "localhost";
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${host}:${port}/ws`;
}
