# TITAN Env (worksonmymachine)

Autonomous satellite fault recovery under radiation stress, with an OpenEnv-compatible environment, GRPO-trained Qwen3 adapter support, and a live 3D dashboard.

## Project Links

- Hugging Face Space: https://huggingface.co/spaces/mugenkyou/worksonmymachine/blob/main/BLOG.md
- YouTube Demo: https://youtu.be/PIixwA7NSxk
- Blog: https://huggingface.co/spaces/mugenkyou/worksonmymachine/blob/main/BLOG.md

## What This Repository Includes

- TITAN simulation environment with injected satellite fault scenarios
- Agent stack for diagnosis and recovery decisions
- OpenEnv wrapper and validation manifest (`openenv.yaml`)
- Inference runner with API-key and fallback behavior
- 3D visualization frontend + backend streaming loop
- GRPO adapter artifacts under `grpo_qwen3_final/`

## Quick Start (Unified)

This section replaces the old QUICKSTART.md. Everything you need is here.

### 1. Prerequisites

- Python 3.11+ (3.12 recommended)
- pip
- Node.js 18+ and npm (for visualization frontend)
- Optional: NVIDIA GPU + recent drivers for faster model inference

### 2. Install Dependencies

From repository root:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Frontend dependencies:

```bash
cd visualization/frontend
npm install
cd ../..
```

### 3. Validate OpenEnv Integration

```bash
openenv validate
```

### 4. Run Inference

```bash
python inference.py
```

If no API credentials are found, inference falls back to the built-in policy.

### 5. Run Full Dashboard (Backend + Frontend)

Recommended single command from repo root:

```bash
python server/run.py
```

This starts:

- Backend API/WebSocket on `http://127.0.0.1:8000` (`/ws`)
- Frontend Vite app on `http://localhost:5173`

### 6. Fast UI-Only Mode (No LLM Load)

For quick frontend and simulation iteration:

Windows PowerShell:

```powershell
$env:TITAN_FAST_MODE = "1"
python server/run.py
```

macOS/Linux:

```bash
export TITAN_FAST_MODE=1
python server/run.py
```

To return to GRPO/LLM-backed mode, unset the variable.

## API Keys and Model Configuration

The inference/runtime code checks credentials in this order:

1. `HF_TOKEN`
2. `OPENAI_API_KEY`

Optional overrides:

- `API_BASE_URL`
- `MODEL_NAME`

You can store local development secrets in a `local.env` file at repository root.

## Core Commands

```bash
# OpenEnv validation
openenv validate

# Run inference
python inference.py

# Run dashboard stack
python server/run.py

# Hardening checks
python scripts/hardening_check.py
```

## Main Tasks

- `easy_single_fault_recovery`
- `medium_thermal_stabilization`
- `hard_multi_fault_survival`

Scores are constrained to `(0, 1)` by the task graders.

## Deployment

### Docker

```bash
docker build -t titan-env .
docker run -p 7860:7860 -e PORT=7860 titan-env
```

### Expected HTTP Endpoints

- `GET /ping`
- `POST /reset`
- `POST /step` with payload `{ "command": "<action>" }`

## Notes

- OpenEnv wrapper entrypoint: `titan_env.interface.openenv_wrapper:TITANEnv`
- Primary server module: `titan_env.server.app`
- Training notebook: `notebooks/grpo_qwen3_training.ipynb`
- Validation report: `VALIDATION_REPORT.md`
- Diagnostic summary: `DIAGNOSTIC_AND_RUNNER_SUMMARY.md`

## License

Proprietary software. See `LICENSE`.
