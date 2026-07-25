# GRPO Qwen3 adapter (`grpo_qwen3_final/`)

The fine-tuned **LoRA adapter** for `Qwen3-1.7B` is **not** committed to git (the directory is listed in `.gitignore` to keep the repo and Hub submissions small).

## If you need the weights

1. **From project owners:** obtain the adapter from a **Hugging Face model repo** or another distribution channel, then extract it so you have:
   - `grpo_qwen3_final/adapter_config.json`
   - `grpo_qwen3_final/adapter_model.safetensors` (or sharded safetensors)
   - tokenizer files as needed (`tokenizer.json`, `tokenizer_config.json`, etc.)  
2. **Train your own:** run the training notebook: [`notebooks/grpo_qwen3_training.ipynb`](../notebooks/grpo_qwen3_training.ipynb) (Unsloth + TRL). See [QUICKSTART.md](../QUICKSTART.md) for the full install path.

## Inference / visualization

With `grpo_qwen3_final/` present, local inference and the 3D backend can load the adapter. Without it, set `TITAN_FAST_MODE=1` for heuristics-only, or use the HTTP/OpenEnv path that does not require the LLM.
