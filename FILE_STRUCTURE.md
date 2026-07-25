# TITAN File Structure (Submission-Ready)

Last updated: 2026-04-01

This document describes the repository layout and key components for OpenEnv Round 1.

## Root Directory

### Configuration & Metadata

- **`openenv.yaml`** ⭐ CRITICAL
  - OpenEnv framework manifest
  - Defines environment class, action/observation spaces, entrypoint
  - Entrypoint: `TITAN_env.interface.openenv_wrapper:TITANEnv`

- **`pyproject.toml`**
  - Package metadata and dependencies
  - Includes: openai, gymnasium, pydantic, pytest
  - Entry point: `TITAN-env = TITAN_env.server.app:main`

- **`Dockerfile`** ⭐ CRITICAL
  - Multi-stage Python 3.11 slim container
  - Installs dependencies via editable install (`pip install -e .`)
  - Exposes port 7860
  - CMD: `python -m TITAN_env.server.app`

- **`LICENSE`**
  - Proprietary license terms

- **`.gitignore`**
  - Ignore rules for Python artifacts, venv, egg-info, etc.

### Entrypoints & Setup

- **`inference.py`** ⭐ CRITICAL
  - Root-level inference entrypoint for HF Spaces
  - Runs all three tasks sequentially with LLM agent (OpenAI client)
  - Falls back to no-op model if API credentials missing
  - Logs output in required `[START]`, `[STEP]`, `[END]` format
  - Only file executed when deployed to HuggingFace

### Documentation

- **`README.md`**
  - Overview, setup instructions, inference usage, deployment guide
  - Includes validator instructions and HTTP endpoint examples

- **`FILE_STRUCTURE.md`** (this file)
  - Complete project layout and component descriptions

- **`docs/`**
  - Project documentation and diagrams:
    - `system-architecture.md`: Layered architecture and system overview
    - `repository-capabilities-and-limitations.md`: Capabilities, limitations, and simulation scope
    - `component-diagram.mmd`: Mermaid diagram of system components and interactions

- **`ARCHITECTURE.md`** (optional)
  - Detailed system architecture and design decisions

- **`QUICKSTART.md`** (optional)
  - Quick reference for common tasks

### Lock Files

- **`uv.lock`** / **`requirements.txt`** (if present)
  - Dependency lock for reproducible builds

---

## `TITAN_env/` — Main Package

### `TITAN_env/__init__.py`

- Package root; exports key classes and utilities

---

## `TITAN_env/interface/` — OpenEnv Wrapper

Exposes the OpenEnv-compatible interface to the core simulator.

- **`openenv_wrapper.py`** ⭐ CRITICAL
  - `TITANEnv` class: OpenEnv-compatible wrapper
  - Implements `reset()`, `step(action)`, returns `(observation, reward, done, info)`
  - Uses deterministic seeding for reproducibility

- **`models.py`**
  - Pydantic models: `Observation`, `Action`, `Reward`
  - Validates observation/action payloads

- **`action_mapping.py`**
  - Maps string commands → action IDs
  - Commands: `no_action`, `reset`, `memory_scrub`, `load_shedding`, `power_cycle`, `thermal_throttle`, `isolate`

- **`llm_interface.py`**
  - `render_observation()`: Converts Observation → text for LLM prompts
  - `parse_action()`: Parses LLM output → valid action string
  - Bridges LLM reasoning and environment actions

- **`__init__.py`**
  - Exports: `TITANEnv`, `Observation`, `Action`, `render_observation()`, `parse_action()`

---

## `TITAN_env/server/` — HTTP Server

Deployment entrypoint for containerized execution.

- **`app.py`** ⭐ CRITICAL
  - HTTP server using Python `http.server.ThreadingHTTPServer`
  - Singleton environment instance with thread-safe RLock
  - Endpoints:
    - `GET /ping` → `{"status": "ok"}`
    - `POST /reset` → Observation JSON
    - `POST /step` → Accept `{"command": ...}`, return Observation JSON
  - Functions:
    - `serve(host, port)`: Start server (default 0.0.0.0:7860)
    - `main()`: Factory entrypoint for validator compatibility

- **`__init__.py`**
  - Package marker

---

## `TITAN_env/core/` — Simulator Core

The underlying satellite fault recovery simulator (not modified for hackathon).

### `TITAN_env/core/environment/`

- **`TITAN_env.py`**
  - Main simulation runtime: `TITANEnvCore` class
  - Step logic, observation generation, reward calculation

- **`state_model.py`**
  - State representation: battery, temperature, CPU load, memory, faults, etc.
  - Physics transitions and constraints

- **`actions.py`**
  - Action definitions and effects on state
  - Recovery actions (memory_scrub, load_shedding, etc.)

- **`fault_injection.py`**
  - Radiation fault model: SEU, thermal stress, multi-fault scenarios
  - Probabilistic fault generation during episodes

- **`gym_env.py`**
  - Gymnasium-compatible wrapper around core

### `TITAN_env/core/rewards/`

- **`reward_v1.py`**
  - Baseline reward function (energy + recovery focus)

- **`reward_v2.py`**
  - Multi-objective reward (efficiency + thermal + survivability)

---

## `TITAN_env/evaluation/` — Task & Grading System

Evaluation framework for the three hackathon tasks.

### Tasks

- **`task_easy.py`**
  - `EASY_TASK`: Single SEU fault recovery
  - 30-step max, memory integrity success condition

- **`task_medium.py`**
  - `MEDIUM_TASK`: Thermal stabilization under load
  - 50-step max, multi-objective grading

- **`task_hard.py`**
  - `HARD_TASK`: Multi-fault survival
  - 100-120 step scenario, high-radiation environment

- **`base_task.py`**
  - `BaseTask` dataclass: Common interface for all tasks
  - Methods: `run_actions()`, `evaluate()`, `run_policy()`

- **`registry.py`**
  - `resolve_task_bundle(name)`: Maps short names → task/grader pairs
  - Canonical names: `easy`, `medium`, `hard`
  - `available_task_names()`: List all tasks

### Graders

- **`grader_easy.py`**
  - Scores easy task: memory_integrity (75%) + efficiency (25%)

- **`grader_medium.py`**
  - Scores medium task: thermal reduction (55%) + energy efficiency (25%) + settle time (20%)

- **`grader_hard.py`**
  - Scores hard task: survival steps (60%) + min battery (25%) + thermal stability (15%)

### Pipeline

- **`runner.py`**
  - Core `run_task()`, `run_all_tasks()` orchestration
  - `EvaluationTrajectory` collection and conversion
  - Observation translation (core dict → OpenEnv model)

- **`trajectory.py`**
  - `EvaluationTrajectory`: Stores sequences of obs/actions/rewards/dones

- **`scoring.py`**
  - `score_trajectory()`: Applies registered grader to trajectory

---

## Ignored & Auto-Generated Files

- **`__pycache__/`, `.pyc` files** — Python bytecode, ignored by version control
- **`.pytest_cache/`** — Pytest cache, ignored by version control
- **`TITAN_env.egg-info/`** — Auto-generated by `pip install -e .`; safe to ignore

## `scripts/` — Utilities

- **`hardening_check.py`**
  - Main validation and hardening script (run before submission)
  - Checks reproducibility, Docker build, API fallback, log format, OpenEnv compliance

- **`validate-submission.sh`**
  - OpenEnv submission validator
  - Checks HF Space reachability, Docker build, openenv compliance
  - Run: `bash scripts/validate-submission.sh https://your-space.hf.space`

---

## `server/` — Validator Compatibility Shim

OpenEnv validator expects `server.app:main` as a fallback entrypoint.

- **`server/app.py`**
  - Thin forward to `TITAN_env.server.app:main`

- **`server/__init__.py`**
  - Package marker

---

## `.venv/` — Virtual Environment (Local Only)

Local Python virtual environment with installed dependencies. Not committed to git.

---

## Summary Table

| Component                                  | Purpose              | Critical? | Notes                                         |
| ------------------------------------------ | -------------------- | --------- | --------------------------------------------- |
| `openenv.yaml`                             | Manifest             | ✅        | Defines entrypoint, action/observation spaces |
| `inference.py`                             | Inference entrypoint | ✅        | LLM agent + fallback, runs all tasks          |
| `TITAN_env/interface/openenv_wrapper.py` | OpenEnv wrapper      | ✅        | `TITANEnv` class                            |
| `TITAN_env/server/app.py`                | HTTP server          | ✅        | Deployment container entrypoint               |
| `Dockerfile`                               | Container config     | ✅        | HF Spaces deployment                          |
| `pyproject.toml`                           | Dependencies         | ✅        | Must include openai, gymnasium, pydantic      |
| `README.md`                                | Documentation        | ✅        | Setup, usage, validation instructions         |

---

## Deployment Checklist

✅ `openenv.yaml` created and valid  
✅ `inference.py` written and tested  
✅ `TITAN_env/server/app.py` HTTP endpoints implemented  
✅ `Dockerfile` built and tested  
✅ `pyproject.toml` includes all required dependencies  
✅ `README.md` updated with setup and validator instructions  
✅ Validation script passes: `python scripts/hardening_check.py`  
✅ `openenv validate` passes  
✅ Repository ready for push and HF Space deployment
