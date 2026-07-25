# TITAN — Quick start for a new machine

This repo runs a **local** GRPO-tuned Qwen3 model (Hugging Face `transformers` + optional **Unsloth**). It does **not** use Ollama. If you only want a fast UI without loading the LLM, use `TITAN_FAST_MODE=1` (see below).

---

## 1. What you need installed

| Tool | Why |
|------|-----|
| **Git** | Clone the repository |
| **Python 3.12+** (3.14 works on Windows if you have CUDA wheels) | Backend + agents |
| **Node.js 18+** and **npm** | Vite frontend (`visualization/frontend`) |
| **NVIDIA GPU + driver** (optional) | Fast inference; CPU works but is slow |

You do **not** need Ollama, Docker, or a separate “model server” unless you add that yourself.

---

## 2. Clone and enter the repo

```bash
git clone https://github.com/mugenkyou/worksonmymachine.git
cd worksonmymachine
```

---

## 3. Model weights (`grpo_qwen3_final/`)

The trained **LoRA adapter** lives in `grpo_qwen3_final/` (e.g. `adapter_model.safetensors`, `adapter_config.json`, tokenizer files). If that folder is missing after clone, either:

- Pull **Git LFS** objects if the repo stores them with LFS:  
  `git lfs install && git lfs pull`
- Or copy the folder from whoever gave you the project / your backup.

Without this folder, the backend will still start in **FAST_MODE** (heuristic only); full GRPO mode needs the adapter.

---

## 4. Python environment (recommended: venv)

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

---

## 5. PyTorch (CUDA vs CPU)

**GPU (recommended — RTX / CUDA 12.x):** install the CUDA build **before** `requirements.txt` so you do not get the CPU-only wheel.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**CPU only (slow LLM, fine for smoke tests):**

```bash
pip install torch
```

Check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 6. Rest of the Python stack

```bash
pip install -r requirements.txt
```

**Optional (faster 4-bit load on GPU, after CUDA torch):**

```bash
pip install unsloth
```

If Unsloth is missing, the backend falls back to **transformers + peft** and may download the base **`Qwen/Qwen3-1.7B`** from Hugging Face on first run (~4 GB cache). Set `HF_TOKEN` if the hub asks for auth.

---

## 7. Frontend dependencies

```bash
cd visualization/frontend
npm install
cd ../..
```

---

## 8. Run everything (backend + Vite + browser)

From the **repository root**:

```bash
python server/run.py
```

- Backend: `http://localhost:8000` (WebSocket `ws://localhost:8000/ws`)
- Frontend: `http://localhost:5173` (opened automatically when ready)

**Backend only** (no Vite, no browser):

```powershell
$env:TITAN_NO_FRONTEND="1"
python visualization/backend/server.py
```

---

## 9. Optional: skip the LLM (instant policy)

Useful on weak machines or when you only want the 3D sim + heuristics:

**Windows PowerShell:**

```powershell
$env:TITAN_FAST_MODE="1"
python server/run.py
```

**macOS / Linux:**

```bash
export TITAN_FAST_MODE=1
python server/run.py
```

---

## 10. One-line sanity checks

```bash
# Agents demo (short episodes; needs model + deps)
python agent/demo.py

# Recovery queue logic (no server; FAST_MODE inside script)
python scripts/test_queue_direct.py
```

---

## Summary table

| Step | Command |
|------|---------|
| Clone | `git clone … && cd worksonmymachine` |
| venv | `python -m venv .venv` then activate |
| CUDA torch | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| Deps | `pip install -r requirements.txt` |
| Optional Unsloth | `pip install unsloth` |
| Frontend | `cd visualization/frontend && npm install && cd ../..` |
| Run | `python server/run.py` |

**Ollama:** not used by this project. The stack is **Python + torch + (optional) unsloth + npm**.
