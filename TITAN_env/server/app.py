from __future__ import annotations

import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from TITAN_env.interface.models import Action
from TITAN_env.interface.openenv_wrapper import TITANEnv


_APP_ENV: TITANEnv | None = None
_APP_LOCK = threading.RLock()


def create_env() -> TITANEnv:
    """Create the OpenEnv-compatible TITAN wrapper instance."""
    return TITANEnv()


def _get_env() -> TITANEnv:
    global _APP_ENV
    if _APP_ENV is None:
        _APP_ENV = create_env()
    return _APP_ENV


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/ping", "/health"}:
            _send_json(self, HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/state":
            with _APP_LOCK:
                env = _get_env()
                payload = _model_dump(env.state())
            _send_json(self, HTTPStatus.OK, payload)
            return
        _send_json(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            _send_json(self, HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return

        if self.path == "/reset":
            with _APP_LOCK:
                env = _get_env()
                observation = env.reset()
                response = _model_dump(observation)
            _send_json(self, HTTPStatus.OK, response)
            return

        if self.path == "/step":
            command = str(payload.get("command", "no_action"))
            with _APP_LOCK:
                env = _get_env()
                observation, reward, done, info = env.step(Action(command=command))
                response = {
                    "observation": _model_dump(observation),
                    "reward": _model_dump(reward),
                    "done": bool(done),
                    "info": info,
                }
            _send_json(self, HTTPStatus.OK, response)
            return

        _send_json(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})


def serve(host: str = "0.0.0.0", port: int | None = None) -> None:
    """Run a lightweight HTTP server for container deployments."""
    active_port = int(port or os.environ.get("PORT", "7860"))
    server = ThreadingHTTPServer((host, active_port), _RequestHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    """Minimal script entry point required by OpenEnv packaging checks."""
    _ = create_env()
    return 0


if __name__ == "__main__":
    serve()
