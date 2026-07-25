# VISUALIZATION_GUIDE

## System Overview

### Component Diagram (component-diagram.mmd)
Shows the complete TITAN architecture with integrated RL training pipeline:

**Training Flow (Top Section - 🎓)**
- `TITAN Env` collects trajectories from deterministic environment
- `reward_v2.py` computes multi-objective reward signal for each step
- `GRPO Training` (Unsloth + TRL) fine-tunes Qwen2.5-1.5B on satellite control
- `Checkpoint` (models/grpo_titan_final) stores trained model weights
- Dashed line shows feedback: reward signal flows back to training loop
- Dashed line shows deployment: trained model loads into inference.py

**Inference & Deployment Flow (Middle/Bottom Sections)**
- Three client paths: batch inference, validation, HTTP service
- inference.py orchestrates task execution with loaded model
- Serving layer exposes HTTP endpoints for remote deployment
- Interface layer handles command parsing and observation sanitization
- Core simulation runs deterministic physics and fault injection
- Evaluation layer scores trajectories against task-specific graders

**Data Flow Semantics**
- Solid arrows: synchronous forward data flow
- Dashed arrows: training/deployment feedback loops

### Reward Curve (reward_curve_grpo.png)
Training progress visualization showing:
- **X-axis**: Training step (iteration count)
- **Y-axis**: Mean reward per episode
- **Convergence**: ~1.0 mean reward indicates successful GRPO training
- **Implication**: Trained policy learns to balance fault diagnosis, action cost, and survival

## How They Connect

### Training → Deployment Pipeline
1. TITAN environment generates {observation, action, reward, done} trajectories
2. Multi-objective reward function (reward_v2.py) computes 5 weighted components:
   - `uptime` (no active faults)
   - `fault_severity` (normalized fault pressure)
   - `energy_usage` (current draw penalty)
   - `recovery_latency` (distance from nominal state)
   - `action_cost` (per-action penalty from ACTION_COSTS)
3. GRPO training optimizes policy to maximize cumulative reward
4. Reward curve shows convergence: policy learns to handle partial observability
5. Trained checkpoint (models/grpo_titan_final) deployed via inference.py

### Checkpoint Properties
- Model: Qwen2.5-1.5B (1.5 billion parameters)
- LoRA adapter fine-tuned for 1000+ steps
- Tokenizer configured for satellite command vocabulary
- Template: jinja chat template compatible with OpenAI Chat Completions
- Files: adapter_model.safetensors + tokenizer.json (in models/grpo_titan_final/)

### Inference Loop (inference.py)
1. Load trained checkpoint (or fallback to no-op policy if credentials missing)
2. For each task (easy/medium/hard):
   - Reset environment deterministically by seed
   - For each step up to task.max_steps:
     - Render observation to LLM-friendly text
     - Call model: "Given state X, choose action from [no_action, reset, ...]"
     - Parse response into canonical command
     - Execute step() and collect reward
3. Score trajectory with task grader → clamped to (0, 1)
4. Log structured [START]/[STEP]/[END] output

### Serving Layer (stratos_env/server/app.py)
- Same environment core and interface
- Exposes endpoints: /ping, /health, /state, /reset, /step
- Allows remote clients to run inference loops without local RL training
- Thread-safe singleton environment access

## Consistency Checks

| Component | Training Source | Deployment Source | Alignment |
|-----------|-----------------|-------------------|-----------|
| Observation schema | OBS_KEYS (state_model.py) | Observation model (models.py) | ✓ 13D vector + faults list |
| Action space | ACTION_COSTS + ActionType | action_mapping.py | ✓ 8 discrete actions |
| Reward formula | reward_v2.py weights | Trajectory scoring | ✓ Multi-objective decomposition |
| Termination | check_failure() | openenv_wrapper step() | ✓ Failure OR max_steps |
| Score bounds | safe_score() | scoring.py clamping | ✓ (0.01, 0.99) |

## Validation Flow

```
reward_curve_grpo.png (training success)
         ↓
    Checkpoint ready for deployment
         ↓
component-diagram.mmd (RL → Inference → Service)
         ↓
inference.py --seed 42 (reproducible evaluation)
         ↓
hardening_check.py (determinism + fallback validation)
         ↓
validate-submission.sh (docker build + openenv validate + /reset ping)
```

All visualizations align: reward training → checkpoint deployment → deterministic evaluation → service readiness.
