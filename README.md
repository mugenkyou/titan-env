# TITAN Env

**Autonomous satellite fault recovery under radiation stress.**

> Proprietary software. See [LICENSE](./LICENSE).

## What It Does

- Simulates satellite subsystem faults (SEU, thermal, latch-up, memory corruption)
- Provides an LLM-friendly interface for autonomous recovery decisions
- Scores agent performance on realistic fault scenarios (0.0–1.0)

## Why It Matters

Real satellites face radiation-induced faults that require autonomous recovery when ground control is unavailable. TITAN enables training and evaluation of AI agents for these critical safety scenarios.

## How It Works

```
Observation → LLM → Action → Environment → Score
```

## Quick Start

```bash
pip install -e .
openenv validate
python inference.py
```

## Tasks

- **easy_single_fault_recovery** — Recover onboard memory after radiation-induced corruption before critical data loss. Goal: memory_integrity > 0.9 within 30 steps.
- **medium_thermal_stabilization** — Stabilize CPU and power subsystem temperatures following a thermal fault cascade. Goal: cpu_temp and power_temp within safe thresholds within 40 steps.
- **hard_multi_fault_survival** — Survive a compounding multi-fault scenario spanning memory, thermal, and latch-up anomalies. Goal: keep the system operational for 50 steps without full failure.

Each task has a corresponding grader that returns a score in `[0.0, 1.0]`.

---

## Technical Details

### Environment

The simulator models satellite subsystem health, radiation faults, and
recovery actions. The OpenEnv wrapper (openenv_wrapper) exposes normalized observations.
All continuous telemetry fields are normalized to `[0, 1]`.

#### Observation Space

Power and electrical:

- `voltage`: normalized bus voltage level
- `current_draw`: normalized power demand
- `battery`: normalized battery state of charge

Thermal:

- `cpu_temp`: normalized CPU thermal load
- `power_temp`: normalized power subsystem thermal load

Compute and memory:

- `cpu_load`: normalized processing utilization
- `memory`: normalized memory integrity health

Fault awareness:

- `signal`: derived system health proxy
- `recent_fault_count`: normalized recent fault pressure
- `faults`: active fault label list (for example: `seu`, `thermal`, `memory`)

The underlying core simulator also tracks fault flags (fault_injection) and additional internal
state used for evaluation and grading (reward_v2).

### Action Space

Valid commands accepted by the environment:

| Action             | Description                                             |
| ------------------ | ------------------------------------------------------- |
| `no_action`        | Hold system state and observe progression.              |
| `reset`            | Perform broad recovery reset for unstable states.       |
| `memory_scrub`     | Repair memory integrity after radiation-induced faults. |
| `thermal_throttle` | Reduce heat generation by throttling load.              |
| `load_shedding`    | Drop non-critical load to preserve stability margins.   |
| `power_cycle`      | Reinitialize affected subsystem power state.            |
| `isolate`          | Contain fault spread by isolating a subsystem.          |

## Setup

### Prerequisites

- Python 3.11+ (verify: `python --version`)
- pip or uv package manager
- Docker (optional, for container deployment)
- openenv-core (for validation: `pip install openenv-core`)

### Installation

```bash
# Clone the repository
git clone https://github.com/mugenkyou/TITAN-environment.git
cd TITAN-environment

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install package with dependencies
python -m pip install -e .
```

This installs:

- Core TITAN environment simulator
- OpenEnv wrapper and validation framework
- OpenAI client (for LLM-based inference)
- Testing dependencies (pytest)

### Verify Installation

```bash
# Check OpenEnv compliance
openenv validate
```

## Run Inference

### With LLM Agent (OpenAI)

Set environment variables:

```bash
# Optional: override API endpoint (defaults to OpenAI)
export API_BASE_URL="https://api.openai.com/v1"
# Optional: override model (defaults to gpt-4o-mini)
export MODEL_NAME="gpt-4o-mini"
# Required: your HuggingFace or OpenAI API token
export HF_TOKEN="your-hf-or-api-token"
# Optional: used only if your env loader needs a docker image name
export LOCAL_IMAGE_NAME="your-local-image"
```

Then run:

```bash
python inference.py
```

### Fallback Mode (No API Key)

If API credentials are missing, the script gracefully falls back to a no-op agent:

```bash
python inference.py  # Uses fallback behavior if OPENAI_API_KEY not set
```

### Output Format

The script logs all task executions in the standardized format:

```text
[START] task=easy env=TITAN_env.interface.openenv_wrapper:TITANEnv model=fallback-noop
[STEP] step=1 action=no_action reward=0.75 done=false error=null
[STEP] step=2 action=reset reward=0.85 done=false error=null
[END] success=true steps=12 rewards=0.75,0.85
[START] task=medium env=TITAN_env.interface.openenv_wrapper:TITANEnv model=fallback-noop
...
```

Each task produces three sections:

- `[START]`: Task name, environment, and model name
- `[STEP]`: Per-step results (action, reward, completion status, errors)
- `[END]`: Summary with success flag, total steps, and comma-separated rewards

## Deployment

### Local Docker

Build and run the container locally:

```bash
docker build -t TITAN-env .
docker run -p 7860:7860 -e PORT=7860 TITAN-env
```

### HuggingFace Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Select "Docker" runtime
3. Push this repository (includes `Dockerfile` and `openenv.yaml`)
4. HuggingFace will auto-build and deploy

### HTTP Endpoints

The server exposes:

- `GET /ping` -> HTTP 200, returns `{"status": "ok"}`
- `POST /reset` -> Resets environment, returns initial observation JSON
- `POST /step` -> Advances environment with payload `{"command": "<action>"}`

Example:

```bash
# Ping
curl -X GET http://localhost:7860/ping

# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"command": "no_action"}'
```

## Validation

### Submission Validator

Use the provided validation script to check your submission before upload:

```bash
# On Linux/macOS
bash scripts/validate-submission.sh https://your-space.hf.space

# On Windows (PowerShell)
cd scripts
# Run validator manually or use WSL
```

The validator checks:

1. **HF Space Reachability**: Confirms `/reset` endpoint responds with HTTP 200
2. **Docker Build**: Verifies `Dockerfile` compiles without errors
3. **OpenEnv Compliance**: Runs `openenv validate` to ensure manifest is correct

### Manual Validation

```bash
# Check OpenEnv manifest
openenv validate

# Build Docker image
docker build -t TITAN-env .

# Test HTTP server
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

## Validation

### Hardening & Submission Validator

Use the provided hardening check script to validate your submission before upload:

```bash
python scripts/hardening_check.py
```

This script checks:

1. Reproducibility: Ensures inference results are consistent
2. Docker Build: Verifies Dockerfile builds successfully
3. API Key Fallback: Confirms fallback agent activates without API keys
4. Log Format: Validates all log lines match [START], [STEP], or [END]
5. OpenEnv Compliance: Runs `openenv validate` and checks manifest
6. (If present) Pytest: Runs tests if a tests/ folder exists

```

## Notes

- The package entrypoint is `TITAN_env.interface.openenv_wrapper:TITANEnv`.
- The deployment server is `TITAN_env.server.app`.
- For a full folder breakdown, see [FILE_STRUCTURE.md](./FILE_STRUCTURE.md).
- Additional architecture and system documentation is available in the `docs/` folder.
- System architecture: [docs/system-architecture.md](./docs/system-architecture.md)
- Component diagram: [docs/component-diagram.mmd](./docs/component-diagram.mmd)
```
