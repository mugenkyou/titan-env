"""
TITAN Multi-Agent Fault Recovery Demo

Loads the GRPO-trained Qwen3-1.7B LoRA adapter, instantiates the diagnostic
and recovery agents, and runs three episodes (low / medium / high radiation)
against TITANGymEnv. Per-step output is mirrored to demo_output.txt and a
summary table is printed at the end.

Note on the model: grpo_qwen3_final/ is a LoRA adapter (PEFT), not a merged
full model, so we load the base model with AutoModelForCausalLM and attach
the adapter with peft.PeftModel. The tokenizer ships with the adapter dir.
"""

import os
import sys
from contextlib import redirect_stdout
from io import StringIO

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    sys.path.insert(0, "/kaggle/working/worksonmymachine")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from titan_env.core.environment.fault_injection import (
    INTENSITY_PROFILES,
    FaultInjector,
)
from titan_env.core.environment.gym_env import TITANGymEnv
from agent.diagnostic_agent import DiagnosticAgent
from agent.memory import Memory
from agent.recovery_agent import RecoveryAgent
from agent.run_episode import run_episode


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "grpo_qwen3_final")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "demo_output.txt")
QUANTIZED_BASE_MODEL = "unsloth/qwen3-1.7b-unsloth-bnb-4bit"
FP_BASE_MODEL = "Qwen/Qwen3-1.7B"
MAX_STEPS = 10


def load_trained_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=ADAPTER_DIR,
            max_seq_length=1024,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer, device
    except Exception as e:
        print(f"Unsloth failed ({e}), falling back to transformers...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer, device


def _format_summary_row(idx: int, name: str, steps: int, reward: float) -> str:
    """Render one row of the summary table with aligned columns."""
    label = f"Episode {idx} ({name}):"
    return f"{label:<20} steps={steps:<3}  reward={reward:.2f}"


def main() -> None:
    """Run the multi-agent demonstration across three radiation profiles."""
    buffer = StringIO()

    def emit(line: str = "") -> None:
        """Print to stdout and capture into the demo_output.txt buffer."""
        print(line)
        print(line, file=buffer)

    emit("=" * 60)
    emit("TITAN Multi-Agent Fault Recovery Demo")
    emit("=" * 60)
    emit("Loading GRPO-trained Qwen3-1.7B (base + LoRA adapter)...")

    model, tokenizer, device = load_trained_model()

    emit(f"Model loaded on device: {device}")
    emit(f"Adapter: {ADAPTER_DIR}")

    diagnostic_agent = DiagnosticAgent(model, tokenizer)
    recovery_agent = RecoveryAgent(model, tokenizer)

    profile_names = ["low", "medium", "high"]
    results = []

    for episode_idx, profile_name in enumerate(profile_names, start=1):
        emit()
        emit(f"=== EPISODE {episode_idx} — {profile_name} radiation ===")

        env = TITANGymEnv(
            fault_injector=FaultInjector(INTENSITY_PROFILES[profile_name]),
            reward_version="v3",
            training_mode=False,
        )
        memory = Memory(max_size=5)

        # run_episode prints per-step output via print(); capture it so we
        # can mirror it to both the live stdout and the output file.
        captured = StringIO()
        with redirect_stdout(captured):
            total_reward, steps_survived, _ = run_episode(
                env,
                diagnostic_agent,
                recovery_agent,
                memory,
                max_steps=MAX_STEPS,
            )

        captured_text = captured.getvalue()
        sys.stdout.write(captured_text)
        sys.stdout.flush()
        buffer.write(captured_text)

        emit(
            f"=== EPISODE END: survived {steps_survived} steps, "
            f"total_reward={total_reward:.2f} ==="
        )

        results.append((profile_name, steps_survived, total_reward))
        env.close()

    emit()
    emit("=== SUMMARY ===")
    for i, (name, steps, reward) in enumerate(results, start=1):
        emit(_format_summary_row(i, name, steps, reward))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())

    print(f"\nOutput saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
