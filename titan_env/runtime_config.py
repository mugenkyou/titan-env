from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

AGENT_DIR = REPO_ROOT / "agent"
VISUALIZATION_DIR = REPO_ROOT / "visualization"
VISUALIZATION_BACKEND_DIR = VISUALIZATION_DIR / "backend"
VISUALIZATION_FRONTEND_DIR = VISUALIZATION_DIR / "frontend"
MODELS_DIR = REPO_ROOT / "models"
ADAPTER_DIR = REPO_ROOT / "grpo_qwen3_final"
LEGACY_LOCAL_MODEL_PATH = MODELS_DIR / "grpo_titan_final"


def is_hf_spaces() -> bool:
    return bool(os.environ.get("HUGGINGFACE_SPACES") or os.environ.get("SPACE_ID"))


def get_runtime_host(default: str = "127.0.0.1") -> str:
    if is_hf_spaces():
        default = "0.0.0.0"
    return os.environ.get("TITAN_HOST", default)


def get_backend_port(default: int = 7860) -> int:
    return int(os.environ.get("PORT", os.environ.get("TITAN_PORT", str(default))))


def get_frontend_port(default: int = 5173) -> int:
    return int(os.environ.get("TITAN_FRONTEND_PORT", str(default)))


def backend_ws_url(host: str | None = None, port: int | None = None) -> str:
    active_host = host or get_runtime_host()
    active_port = port if port is not None else get_backend_port()
    return f"ws://{active_host}:{active_port}/ws"