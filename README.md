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
- **medium_thermal_stabilization** — Stabilize CPU and power subsystem temperatures following a thermal fault cascade. Goal: reduce cpu_temperature below 0.70 within 50 steps while keeping average current draw under 0.55.
- **hard_multi_fault_survival** — Survive a compounding multi-fault scenario spanning memory, thermal, and power anomalies. Goal: survive 100 steps without full failure while keeping battery_soc at or above 0.20.

Each task has a corresponding grader that returns a score in `[0.0, 1.0]`.

### Baseline Scores

Scores below were generated with the built-in no-op fallback policy and seed `42`:

| Task     |   Score | Steps |
| -------- | ------: | ----: |
| `easy`   | `1.000` |  `30` |
| `medium` | `0.996` |  `50` |
| `hard`   | `0.660` | `120` |

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
git clone https://github.com/mugenkyou/titan-environment.git
cd titan-environment

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

### API Key Configuration

The inference engine supports multiple ways to provide API credentials (tried in order):

1. **local.env file** (recommended for development):

   ```bash
   # Create local.env in the project root
   echo "HF_TOKEN=your-token-here" >> local.env
   # or
   echo "OPENAI_API_KEY=sk-..." >> local.env
   ```

   The file is automatically loaded at runtime and is git-ignored.

2. **Environment variables** (recommended for CI/production):

   ```bash
   export HF_TOKEN="your-token"
   export OPENAI_API_KEY="sk-..."
   ```

3. **Override API endpoint** (optional):
   ```bash
   export API_BASE_URL="https://api.openai.com/v1"
   export MODEL_NAME="gpt-4o-mini"
   ```

The inference script reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` first, then falls back to `OPENAI_API_KEY` if `HF_TOKEN` is not set.

### With LLM Agent (OpenAI)

Once credentials are set via any method above, run:

```bash
python inference.py
```

The agent will use your configured API key to call the LLM for policy decisions.

### Fallback Mode (No API Key)

If no API credentials are found, inference gracefully activates a built-in fallback agent:

```bash
python inference.py  # Falls back to no-op policy if credentials are missing
```

This allows reproducible offline testing and validation without external dependencies.

### Output Format

The script logs all task executions in the standardized format:

```text
[START] task=easy env=titan_env.interface.openenv_wrapper:TITANEnv model=fallback-noop
[STEP] step=1 action=no_action reward=0.75 done=false error=null
[STEP] step=2 action=reset reward=0.85 done=false error=null
[END] success=true steps=12 rewards=0.75,0.85
[START] task=medium env=titan_env.interface.openenv_wrapper:TITANEnv model=fallback-noop
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
docker build -t titan-env .
docker run -p 7860:7860 -e PORT=7860 titan-env
```

### HuggingFace Spaces

1. Live Space: https://huggingface.co/spaces/mugenkyou/titan-env
2. Create a new Space at https://huggingface.co/spaces if you need to redeploy or fork it
3. Select "Docker" runtime
4. Push this repository (includes `Dockerfile` and `openenv.yaml`)
5. HuggingFace will auto-build and deploy

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
bash scripts/validate-submission.sh https://huggingface.co/spaces/mugenkyou/titan-env

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
docker build -t titan-env .

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

- The package entrypoint is `titan_env.interface.openenv_wrapper:TITANEnv`.
- The deployment server is `titan_env.server.app`.
- For a full folder breakdown, see [FILE_STRUCTURE.md](./FILE_STRUCTURE.md).
- Additional architecture and system documentation is available in the `docs/` folder.
- System architecture: [docs/system-architecture.md](./docs/system-architecture.md)
- Component diagram: [docs/component-diagram.mmd](./docs/component-diagram.mmd)
```
