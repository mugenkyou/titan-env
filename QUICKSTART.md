# TITAN — Full setup protocol (new machine)

This document is the **complete** path from an empty machine to a running 3D Earth + satellite dashboard with the GRPO Qwen3 recovery agents.

**What this project is not:** it does **not** use **Ollama**, LM Studio, or vLLM out of the box. Inference is **in-process** via **PyTorch** + **Hugging Face** (`transformers`, optional **Unsloth**, **PEFT**). The trained policy is a **LoRA adapter** in `grpo_qwen3_final/` on top of the base **`Qwen/Qwen3-1.7B`** weights.

---

## 0. Run GRPO + web UI (first success)

Goal: **backend loads Diagnostic + Recovery agents** (GRPO adapter) and the **Vite frontend** talks to **`ws://<host>:8000/ws`** — same stack for the main dashboard, **Decision**, and **Analysis** pages.

### Checklist before `python server/run.py`

| Requirement | Why |
|---------------|-----|
| **`grpo_qwen3_final/`** at repo root with `adapter_config.json` + weights | LoRA GRPO policy; without it, startup may error or degrade depending on code path. |
| **`TITAN_FAST_MODE` unset** | If set to `1`, the server **never loads** the LLM — heuristics only. For GRPO, remove it from the environment. |
| **CUDA `torch` (GPU)** or patience (CPU) | GPU: seconds per LLM step typical. CPU: often **30–60+ s** per LLM step; use `TITAN_FAST_MODE=1` for UI dev without waiting. |
| **`pip install -r requirements.txt`** + **`npm install`** in `visualization/frontend` | Python deps + frontend dev server. |

### One command (repo root, recommended)

**Windows PowerShell** (venv activated; use `py -3.12` if needed):

```powershell
Remove-Item Env:TITAN_FAST_MODE -ErrorAction SilentlyContinue
python server/run.py
```

**macOS / Linux:**

```bash
unset TITAN_FAST_MODE
python server/run.py
```

This runs **Uvicorn** on **`http://127.0.0.1:8000`** (WebSocket **`/ws`**) and spawns **`npm run dev`** for **`http://localhost:5173`**. First LLM load can take **30s–2min**; first-time **Qwen3-1.7B** download can take longer (see §9).

Wait for console lines like **`[OK] Agents attached.`** and **`[OK] Broadcast loop + LLM worker started`**. Then use the browser UI; switch simulation to **Auto** in the top bar if you want continuous client-driven steps while the agents run on faults (hybrid policy — §13).

### Heuristic-only (no GRPO load)

Useful when you only need the 3D UI and env stepping without GPU memory or downloads:

```powershell
$env:TITAN_FAST_MODE = "1"
python server/run.py
```

### Two terminals (manual), same model

```powershell
# Terminal 1 — backend only (from repo root)
Remove-Item Env:TITAN_FAST_MODE -ErrorAction SilentlyContinue
python visualization/backend/server.py

# Terminal 2 — frontend
cd visualization/frontend
npm run dev
```

Open **http://localhost:5173**. The frontend discovers the WebSocket host from the browser URL and uses **port 8000** for `/ws` (see `visualization/frontend/src/utils/wsBackendUrl.ts`).

---

## Contents

0. [Run GRPO + web UI (first success)](#0-run-grpo--web-ui-first-success)  
1. [Hardware and OS assumptions](#1-hardware-and-os-assumptions)  
2. [Install system prerequisites](#2-install-system-prerequisites)  
3. [Clone the repository](#3-clone-the-repository)  
4. [Adapter weights (`grpo_qwen3_final/`)](#4-adapter-weights-grpo_qwen3_final)  
5. [Python virtual environment](#5-python-virtual-environment)  
6. [GPU mode — full PyTorch + CUDA protocol](#6-gpu-mode--full-pytorch--cuda-protocol)  
7. [CPU-only mode (no GPU)](#7-cpu-only-mode-no-gpu)  
8. [Python dependencies (`requirements.txt`)](#8-python-dependencies-requirementstxt)  
9. [Downloading Qwen (base model) and cache layout](#9-downloading-qwen-base-model-and-cache-layout)  
10. [Optional Unsloth (4-bit, faster on GPU)](#10-optional-unsloth-4-bit-faster-on-gpu)  
11. [Frontend (Node.js / npm)](#11-frontend-nodejs--npm)  
12. [Run the application](#12-run-the-application)  
13. [Environment variables reference](#13-environment-variables-reference)  
14. [Verify GPU and model loading](#14-verify-gpu-and-model-loading)  
15. [Troubleshooting](#15-troubleshooting)  
16. [Command cheat sheet](#16-command-cheat-sheet)

---

## 1. Hardware and OS assumptions

| Mode | Expectation |
|------|-------------|
| **GPU (recommended)** | NVIDIA GPU with a recent driver (CUDA 12.x runtime is typical for current drivers). RTX 30xx / 40xx / 50xx laptop or desktop works well. |
| **CPU** | Works for the **simulation** and **FAST_MODE** heuristics. Full **Qwen3-1.7B** inference is **very slow** (often tens of seconds per step without a GPU). |
| **RAM** | Plan for **8 GB+** free for the Python process when the base model is loaded in FP16; **4-bit** (Unsloth) uses less VRAM/RAM pressure. |
| **Disk** | **~5 GB** for Hugging Face cache if you use the transformers fallback (base model). Adapter in `grpo_qwen3_final/` is on the order of **tens to low hundreds of MB** depending on packaging. |

---

## 2. Install system prerequisites

### 2.1 Git

- **Windows:** [Git for Windows](https://git-scm.com/download/win)  
- **macOS:** `xcode-select --install` (includes git) or install from git-scm.com  
- **Linux:** `sudo apt install git` (Debian/Ubuntu) or your distro equivalent  

Optional but useful for large files in the repo:

```bash
git lfs install
```

### 2.2 Python

Use **Python 3.12** if you can (matches `requirements.txt` notes). **3.14** on Windows can work if a **CUDA-enabled** `torch` wheel exists for that Python version; otherwise use 3.12.

- **Windows:** install from [python.org](https://www.python.org/downloads/) and tick **“Add python.exe to PATH”**.  
- Verify:

```powershell
py --list
python --version
```

### 2.3 Node.js (for the Vite frontend)

Install **Node.js 18 LTS or newer** (includes `npm`).

- **Windows:** [nodejs.org](https://nodejs.org/) MSI installer.  
- Verify:

```bash
node --version
npm --version
```

### 2.4 NVIDIA driver (GPU mode only)

Install the latest **Game Ready** or **Studio** driver from [NVIDIA](https://www.nvidia.com/Download/index.aspx).  
Then confirm the GPU is visible:

```powershell
nvidia-smi
```

You should see GPU name, driver version, and CUDA version in the header.

---

## 3. Clone the repository

```bash
git clone https://github.com/mugenkyou/worksonmymachine.git
cd worksonmymachine
```

If the project uses Git LFS for large assets:

```bash
git lfs pull
```

---

## 4. Adapter weights (`grpo_qwen3_final/`)

The **GRPO-trained LoRA adapter** must live at the repo root in:

```text
grpo_qwen3_final/
  adapter_config.json
  adapter_model.safetensors   (or split shards — follow your checkpoint layout)
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  …
```

If this folder is **missing** after `git clone`:

1. Run `git lfs pull` if weights are stored with LFS.  
2. Or obtain a zip/tar from the maintainer and extract into `grpo_qwen3_final/`.

**Without** this folder, you can still run **`TITAN_FAST_MODE=1`** (heuristics only). Full LLM mode expects the adapter directory to exist.

---

## 5. Python virtual environment

Always use a venv so `pip` does not break your system Python.

**Windows (PowerShell), from repo root:**

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser   # if Activate.ps1 is blocked
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 6. GPU mode — full PyTorch + CUDA protocol

Goal: `import torch` → `torch.cuda.is_available()` is **`True`** and the backend logs your GPU name.

### 6.1 Why order matters

Install **`torch` with CUDA first**, then `pip install -r requirements.txt`. If you install a generic `torch` from PyPI first, you may get **`+cpu`** and the LLM will run on CPU only.

### 6.2 Install CUDA-enabled PyTorch (cu128 index)

This matches the PyTorch **CUDA 12.8** wheel line used in this project’s docs:

```bash
pip uninstall -y torch torchvision torchaudio 2>nul; pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Windows PowerShell** (same idea; `2>$null` suppresses errors if nothing to uninstall):

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If `pip` hangs on huge wheels, download the `.whl` from the same index in a browser or with `curl.exe -L -C -`, then:

```powershell
pip install --no-deps --force-reinstall path\to\torch-...+cu128-....whl
```

Then install the rest of PyTorch ecosystem if needed:

```powershell
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

(Only if your code imports them; TITAN core may not need them.)

### 6.3 Verify GPU mode

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

Expected (example):

```text
torch 2.x.x+cu128
cuda True
device NVIDIA GeForce RTX 4060 Laptop GPU
```

If **`cuda False`**, you still have a CPU build: uninstall `torch` and repeat §6.2 from a **clean venv** if needed.

### 6.4 Windows console and Unicode

If the backend prints Unicode and the console errors, set:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

before `python server/run.py`.

---

## 7. CPU-only mode (no GPU)

Install CPU PyTorch (acceptable for **FAST_MODE** or very short tests):

```bash
pip install torch
```

Expect **minutes per LLM step** if you load full Qwen3-1.7B + adapter on CPU. For day-to-day use without a GPU:

```powershell
$env:TITAN_FAST_MODE = "1"
python server/run.py
```

---

## 8. Python dependencies (`requirements.txt`)

With the venv **activated** and **`torch` already installed** the way you want (§6 or §7):

```bash
pip install -r requirements.txt
```

This pulls **FastAPI**, **uvicorn**, **websockets**, **gymnasium**, **numpy**, **transformers**, **peft**, **accelerate**, **safetensors**, **huggingface_hub**, etc. It intentionally does **not** pin `torch` (you chose CUDA vs CPU above).

---

## 9. Downloading Qwen (base model) and cache layout

There are **two** loading paths in `visualization/backend/server.py`:

| Path | When it runs | What gets downloaded |
|------|----------------|------------------------|
| **A — Unsloth** | `import unsloth` succeeds **and** `FastLanguageModel.from_pretrained(grpo_qwen3_final, load_in_4bit=True)` succeeds | Unsloth may still resolve base weights depending on adapter packaging; often **less** host RAM than full FP16 if 4-bit works. |
| **B — transformers + PEFT (fallback)** | Unsloth missing or fails | **`AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", ...)`** downloads the **public base model** from the Hugging Face Hub into your cache on **first run**. Tokenizer is loaded from **`grpo_qwen3_final/`** in code, then PEFT merges the adapter. |

### 9.1 Where files go (Hugging Face cache)

By default:

| OS | Typical cache directory |
|----|-------------------------|
| Windows | `C:\Users\<you>\.cache\huggingface\hub\` |
| Linux / macOS | `~/.cache/huggingface/hub/` |

Override if you want models on a bigger disk:

```powershell
$env:HF_HOME = "D:\hf-cache"
```

```bash
export HF_HOME=/data/hf-cache
```

### 9.2 Size (rough order of magnitude)

- **`Qwen/Qwen3-1.7B`** (FP16 / safetensors via `transformers`): on the order of **several GB** in the hub cache the first time.  
- **`grpo_qwen3_final/`** adapter: much smaller than the base (LoRA).

### 9.3 Prefetch Qwen without starting the server (optional)

If you want to **download ahead of time** using the CLI:

```bash
pip install huggingface_hub[cli]
huggingface-cli download Qwen/Qwen3-1.7B --local-dir ./models/Qwen3-1.7B
```

The app code currently loads **`Qwen/Qwen3-1.7B`** by **repo id** from the hub in fallback mode, not from `./models/...`, so prefetching into the default **HF cache** is what aligns with `from_pretrained("Qwen/Qwen3-1.7B")`. Easiest “prefetch” is:

```bash
python -c "from transformers import AutoModelForCausalLM; import torch; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', torch_dtype=torch.float16)"
```

That populates the cache; interrupt after download if you only wanted the files.

### 9.4 Authentication (`HF_TOKEN`)

`Qwen/Qwen3-1.7B` is a **public** model; a token is usually **not** required. If Hugging Face ever returns **401/403** (rate limits, org policies, mirrors):

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).  
2. **Windows PowerShell:**

   ```powershell
   $env:HF_TOKEN = "hf_xxxxxxxx"
   ```

3. **Linux / macOS:**

   ```bash
   export HF_TOKEN=hf_xxxxxxxx
   ```

Or run `huggingface-cli login` once.

---

## 10. Optional Unsloth (4-bit, faster on GPU)

Install **after** CUDA `torch` is working:

```bash
pip install unsloth
```

Unsloth wheels are sensitive to **torch + CUDA + Python version**. If import fails, stay on the **transformers + PEFT** path (still uses GPU via `device_map="auto"` when CUDA is available).

---

## 11. Frontend (Node.js / npm)

From repo root:

```bash
cd visualization/frontend
npm install
cd ../..
```

Do **not** commit `node_modules/`; it is listed in `.gitignore`.

---

## 12. Run the application

From **repository root**, with venv activated:

```bash
python server/run.py
```

This starts:

1. **FastAPI + WebSocket** backend — `http://127.0.0.1:8000`, WebSocket path **`/ws`**.  
2. **Vite** dev server for the Three.js UI — **`http://localhost:5173`**.  
3. Opens the browser when both ports respond (unless disabled).

### 12.1 Full GRPO / hybrid (default)

- Do **not** set `TITAN_FAST_MODE`.  
- On startup the backend loads **Unsloth** (if available) or **transformers + PEFT** with **`grpo_qwen3_final/`** and attaches **DiagnosticAgent** + **RecoveryAgent**.  
- **Hybrid:** quiet telemetry steps can stay fast; when faults queue, the **LLM worker** runs GRPO Qwen3 recovery (see comments in `visualization/backend/server.py`).  
- Optional: `TITAN_ALWAYS_LLM=1` forces an LLM pass every step (heavy).

### 12.2 Frontend + backend together

| URL | Role |
|-----|------|
| `http://localhost:5173/` | Main Three.js dashboard (`index.html`) |
| `http://localhost:5173/decision.html` | Decision view (`decision.html`) |
| `http://localhost:5173/analysis.html` | Analysis console |

All of the above use the **same** WebSocket backend on **port 8000** once the Python process is up.

**Backend only** (no npm / no browser):

```powershell
$env:TITAN_NO_FRONTEND = "1"
python visualization/backend/server.py
```

You can still run **`npm run dev`** manually in `visualization/frontend` in a second terminal.

---

## 13. Environment variables reference

| Variable | Values | Effect |
|----------|--------|--------|
| `TITAN_FAST_MODE` | `1` | **Skips GRPO/LLM entirely** — no model load; **heuristic** policy only. **Unset** this variable for full GRPO agents. |
| `TITAN_ALWAYS_LLM` | `1` | Run the LLM on **every** step (slow unless GPU + good stack). |
| `TITAN_NO_FRONTEND` | `1` | `server/run.py` does not spawn `npm run dev`. |
| `TITAN_NO_BROWSER` | `1` | Do not auto-open a browser tab. |
| `TITAN_HOST` | default `127.0.0.1` | Uvicorn bind host. |
| `TITAN_PORT` | default `8000` | Backend port. |
| `TITAN_FRONTEND_PORT` | default `5173` | Vite dev server port. |
| `HF_HOME` / `HF_TOKEN` | optional | Hugging Face cache dir / auth (§9). |
| `PYTHONIOENCODING` | `utf-8` | Helps Windows consoles with Unicode logs. |

Default behaviour (no `TITAN_FAST_MODE`): **hybrid** — quiet steps use heuristics; when faults are active the GRPO pipeline is used (see `visualization/backend/server.py` comments).

---

## 14. Verify GPU and model loading

1. **Torch sees CUDA** (§6.3).  
2. Start the backend and read the first log lines: you should see `CUDA available: True` and the device name when not in `TITAN_FAST_MODE`.  
3. Optional one-shot agent demo (short episodes):

   ```bash
   python agent/demo.py
   ```

4. Optional queue test (no network, no LLM file load beyond what the script imports):

   ```bash
   python scripts/test_queue_direct.py
   ```

---

## 15. Troubleshooting

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| Console shows **FAST_MODE** / “skipping LLM” | `TITAN_FAST_MODE=1` in environment | Unset: `Remove-Item Env:TITAN_FAST_MODE` (PowerShell) or `unset TITAN_FAST_MODE` (bash), then restart the backend. |
| `torch.cuda.is_available()` is `False` but you have an NVIDIA GPU | CPU-only `torch` installed | Uninstall `torch`, reinstall from **cu128** index (§6). |
| `OSError` / “Application Control policy” loading `c10.dll` (Windows) | Smart App Control / WDAC blocking PyTorch | Use WSL2, different Python install path, or adjust Windows policy per your org’s rules. |
| First startup **downloads for a long time** | Normal: **`Qwen/Qwen3-1.7B`** cache fill | Wait, or prefetch (§9.3). Ensure disk space. |
| `pip` hangs | Huge wheel + slow network | Download `.whl` manually, `pip install` the file (§6.2). |
| `git push` HTTP 500 with **multi-GB** pack | Accidentally committed **`wheels/*.whl`** or **`node_modules/`** | Never commit those; use `.gitignore` and `git rm --cached` if needed. |
| UI shows “Waiting for simulation…” | Backend not running or wrong port | Confirm `python server/run.py` and firewall allows `8000` / `5173`. |

---

## 16. Command cheat sheet

Copy from top to bottom on a **fresh machine** (GPU path). Adjust Python command (`py -3.12` vs `python3.12`) for your OS.

```bash
git clone https://github.com/mugenkyou/worksonmymachine.git
cd worksonmymachine
git lfs pull    # if your fork uses LFS for adapters

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
# source .venv/bin/activate             # Linux / macOS

python -m pip install --upgrade pip
pip uninstall -y torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

pip install -r requirements.txt
pip install unsloth                       # optional, GPU 4-bit path

cd visualization/frontend && npm install && cd ../..

# Full GRPO + UI (ensure FAST_MODE is off)
Remove-Item Env:TITAN_FAST_MODE -ErrorAction SilentlyContinue   # Windows
# unset TITAN_FAST_MODE                                          # Linux / macOS
python server/run.py
```

**CPU / heuristic-only shortcut:**

```powershell
$env:TITAN_FAST_MODE = "1"
python server/run.py
```

---

**Summary:** Install **Git**, **Python 3.12+**, **Node.js**, and (for GPU) **NVIDIA drivers** + **CUDA-enabled PyTorch**. Place **`grpo_qwen3_final/`** in the repo. Run **`pip install -r requirements.txt`**, **`npm install`** in the frontend folder, then **`python server/run.py`**. The **Qwen3-1.7B base** weights download automatically on first LLM load if you are on the **transformers + PEFT** fallback; **Unsloth** is optional but recommended on GPU when it installs cleanly.
