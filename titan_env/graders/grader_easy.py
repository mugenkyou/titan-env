from __future__ import annotations

from typing import Any, Dict, List


def safe_score(score: float) -> float:
    return min(max(round(float(score), 4), 0.01), 0.99)


def grade_easy(trajectory: List[Dict[str, Any]], max_steps: int = 30) -> float:
    """Grade easy task without using reward values."""
    if not trajectory:
        return safe_score(0.0)

    final_state = trajectory[-1].get("state", {})
    # Support both 'memory_integrity' (core) and 'memory' (OpenEnv wrapper) field names
    final_memory = safe_score(float(final_state.get("memory_integrity", final_state.get("memory", 0.0))))

    steps_taken = max(0, len(trajectory) - 1)
    efficiency = 1.0 - min(float(steps_taken), float(max_steps)) / float(max_steps)
    efficiency = safe_score(efficiency)

    score = 1.0 * final_memory  # focus on memory recovery
    return safe_score(score)
