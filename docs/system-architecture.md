# STRATOS System Architecture

## Purpose

STRATOS is a deterministic benchmark environment for autonomous satellite fault recovery.
It is designed to support two execution modes:

1. Batch evaluation mode via `inference.py`
2. Service mode via `stratos_env/server/app.py` HTTP endpoints

The architecture prioritizes reproducibility, clear contracts, and OpenEnv compatibility.

## System Layers

### 1) Simulation Core

This layer owns state transitions and mission dynamics.

- Satellite-like state model (power, thermal, memory, CPU, fault flags)
- Fault generation/injection behavior
- Action effects on the underlying state
- Reward computation primitives

Primary modules:

- `stratos_env/core/environment/stratos_env.py`
- `stratos_env/core/environment/state_model.py`
- `stratos_env/core/environment/fault_injection.py`
- `stratos_env/core/environment/actions.py`
- `stratos_env/core/rewards/reward_v2.py`

### 2) Interface and Contract Layer

This layer converts raw simulator data into OpenEnv-safe contracts.

- Typed `Action`, `Observation`, and `Reward` models
- Command-to-discrete action mapping
- Wrapper methods: `reset()`, `step()`, `state()`
- LLM-facing helpers for observation rendering and command parsing

Primary modules:

- `stratos_env/interface/openenv_wrapper.py`
- `stratos_env/interface/models.py`
- `stratos_env/interface/action_mapping.py`
- `stratos_env/interface/llm_interface.py`

### 3) Task and Scoring Layer

This layer defines benchmark scenarios and scoring rules.

- Task registry and task bundles
- Deterministic task initialization by seed
- Trajectory collection and grader record conversion
- Bounded scoring in `[0.0, 1.0]`

Primary modules:

- `stratos_env/tasks/base_task.py`
- `stratos_env/tasks/task_easy.py`
- `stratos_env/tasks/task_medium.py`
- `stratos_env/tasks/task_hard.py`
- `stratos_env/tasks/registry.py`
- `stratos_env/evaluation/trajectory.py`
- `stratos_env/evaluation/scoring.py`
- `stratos_env/evaluation/runner.py`

### 4) Orchestration and Agent Layer

This layer runs task loops and drives policy decisions.

- Prompt construction per task and step
- Model invocation through OpenAI-compatible Chat Completions
- Action parsing and guarded execution
- Structured logs: `[START]`, `[STEP]`, `[END]`

Primary module:

- `inference.py`

### 5) Serving and Deployment Layer

This layer exposes the environment over HTTP and packages runtime artifacts.

- Thread-safe singleton environment in the server process
- Endpoints: `GET /ping`, `GET /health`, `GET /state`, `POST /reset`, `POST /step`
- Container packaging for reproducible deployment
- Submission readiness scripts

Primary modules:

- `stratos_env/server/app.py`
- `Dockerfile`
- `scripts/validate-submission.sh`
- `scripts/hardening_check.py`

## Runtime Flows

### A) Batch Evaluation Flow (`inference.py`)

1. Load registered tasks from the task registry.
2. Build a task-specific seeded core environment.
3. Wrap the core with `StratosEnv` OpenEnv wrapper.
4. Convert observation to LLM prompt context.
5. Parse model output into a validated action command.
6. Execute `step()` and append trajectory records.
7. Score trajectory with task grader and emit summary.

### B) HTTP Service Flow (`stratos_env/server/app.py`)

1. Request enters `ThreadingHTTPServer` request handler.
2. Handler acquires lock and resolves singleton environment.
3. `/reset` returns a normalized observation.
4. `/step` accepts a command and returns observation, reward, done, and info.
5. `/state` exposes the latest structured observation snapshot.

## Data Contracts

- Commands are limited to the canonical action set.
- Observations are normalized to bounded values for stability.
- Rewards are wrapped as typed values and serialized for transport.
- Fault flags are surfaced as a compact `faults` list for downstream policies.

## Determinism and Reproducibility

- Determinism is task-seed driven in evaluation mode.
- Trajectory and scoring are deterministic for fixed seeds and fixed model outputs.
- Logging format is stable and machine-parseable for benchmark pipelines.

## Design Principles

- Clear boundaries: simulation, interface, evaluation, and serving are separated.
- Explicit contracts: typed models at interface boundaries reduce ambiguity.
- Operational resilience: request-level guards and exception handling prevent hard crashes.
- Deployment portability: local and hosted execution use the same package/runtime model.
