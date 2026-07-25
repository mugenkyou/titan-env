"""
TITAN startup script — visualization edition.

Boots two processes:

  1. The LLM-powered FastAPI backend at  ws://localhost:8000/ws
     (visualization/backend/server.py — loads the GRPO Qwen3-1.7B adapter
      via Unsloth, falls back to transformers + peft if unavailable.)

  2. The Three.js Earth/satellite frontend dev server (Vite) at
     http://localhost:5173 (visualization/frontend).

When both are reachable the browser is opened to the frontend URL.

Usage:
    python server/run.py

Env overrides:
    TITAN_HOST          (default 127.0.0.1)
    TITAN_PORT          (default 8000)
    TITAN_FRONTEND_PORT (default 5173)
    TITAN_NO_BROWSER=1  skip auto-opening the browser
    TITAN_NO_FRONTEND=1 skip launching the Vite dev server
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402

FRONTEND_DIR = PROJECT_ROOT / "visualization" / "frontend"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_port(host: str, port: int, timeout: float = 180.0) -> bool:
    """Poll a TCP port until it accepts a connection or timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _find_npm() -> str | None:
    """Return a launchable npm executable name, accounting for Windows .cmd."""
    for candidate in ("npm.cmd", "npm"):
        if shutil.which(candidate):
            return candidate
    return None


def _start_frontend(frontend_port: int) -> subprocess.Popen | None:
    """Spawn `npm run dev` for the Vite frontend. Returns the process handle."""
    if not FRONTEND_DIR.is_dir():
        print(f"[run] visualization/frontend not found at {FRONTEND_DIR}; skipping.")
        return None

    npm = _find_npm()
    if npm is None:
        print("[run] npm not found on PATH; cannot start the Vite frontend.")
        print("      Install Node.js or set TITAN_NO_FRONTEND=1 to skip.")
        return None

    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.is_dir():
        print(f"[run] Installing frontend deps with `{npm} install` (one-time) ...")
        try:
            subprocess.run(
                [npm, "install"],
                cwd=str(FRONTEND_DIR),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[run] npm install failed: {exc}")
            return None

    print(f"[run] Starting Vite dev server on port {frontend_port} ...")
    env = os.environ.copy()
    env.setdefault("VITE_PORT", str(frontend_port))
    try:
        proc = subprocess.Popen(
            [npm, "run", "dev", "--", "--port", str(frontend_port), "--host"],
            cwd=str(FRONTEND_DIR),
            env=env,
        )
        return proc
    except Exception as exc:  # noqa: BLE001
        print(f"[run] Failed to launch Vite: {exc}")
        return None


def _open_browser(host: str, backend_port: int, frontend_port: int) -> None:
    print(f"[run] Waiting for backend on {host}:{backend_port} ...")
    if not _wait_for_port(host, backend_port, timeout=600.0):
        print(f"[run] Backend never became reachable at {host}:{backend_port}.")
        return

    if os.environ.get("TITAN_NO_FRONTEND") == "1":
        print(f"[run] Backend ready at http://{host}:{backend_port}")
        return

    print(f"[run] Waiting for frontend on {host}:{frontend_port} ...")
    if not _wait_for_port(host, frontend_port, timeout=180.0):
        print(f"[run] Frontend never became reachable at {host}:{frontend_port}.")
        return

    url = f"http://localhost:{frontend_port}/"
    print(f"[run] Opening {url}")
    try:
        webbrowser.open(url)
    except Exception as exc:  # noqa: BLE001
        print(f"[run] Could not open browser: {exc}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    host = os.environ.get("TITAN_HOST", "127.0.0.1")
    backend_port = int(os.environ.get("TITAN_PORT", "8000"))
    frontend_port = int(os.environ.get("TITAN_FRONTEND_PORT", "5173"))

    frontend_proc: subprocess.Popen | None = None
    if os.environ.get("TITAN_NO_FRONTEND") != "1":
        frontend_proc = _start_frontend(frontend_port)

    if os.environ.get("TITAN_NO_BROWSER") != "1":
        threading.Thread(
            target=_open_browser,
            args=(host, backend_port, frontend_port),
            daemon=True,
        ).start()

    print()
    print("=" * 60)
    print(f"[run] TITAN backend  -> http://{host}:{backend_port}  (ws=/ws)")
    if frontend_proc is not None:
        print(f"[run] TITAN frontend -> http://{host}:{frontend_port}/")
    print("=" * 60)
    print("[run] First-time model load can take 30s–2min depending on cache.")
    print("[run] Press Ctrl+C to stop.")
    print()

    try:
        uvicorn.run(
            "visualization.backend.server:app",
            host=host,
            port=backend_port,
            log_level="info",
            reload=False,
        )
    finally:
        if frontend_proc is not None and frontend_proc.poll() is None:
            print("[run] Shutting down Vite dev server ...")
            try:
                frontend_proc.terminate()
                frontend_proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                frontend_proc.kill()


if __name__ == "__main__":
    main()
