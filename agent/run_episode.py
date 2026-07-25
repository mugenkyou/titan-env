"""
TITAN Episode Runner

Orchestrates diagnostic and recovery agents through an episode.
"""

from typing import Any, List, Tuple

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TITAN_env.interface.action_mapping import COMMAND_TO_ACTION


def run_episode(
    env: Any,
    diagnostic_agent: Any,
    recovery_agent: Any,
    memory: Any,
    max_steps: int = 100,
) -> Tuple[float, int, List[str]]:
    """
    Run a single episode with diagnostic and recovery agents.

    Parameters
    ----------
    env : Any
        TITAN gymnasium environment.
    diagnostic_agent : DiagnosticAgent
        Agent for fault diagnosis.
    recovery_agent : RecoveryAgent
        Agent for action selection.
    memory : Memory
        Memory buffer for step history.
    max_steps : int
        Maximum steps per episode.

    Returns
    -------
    Tuple[float, int, List[str]]
        (total_reward, steps_survived, memory_history)
    """
    obs, info = env.reset()
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while step_count < max_steps and not (terminated or truncated):
        step_count += 1

        diagnostic_result = diagnostic_agent.run(obs, memory.get())
        fault = diagnostic_result.get("fault", "none")
        severity = diagnostic_result.get("severity", 1)
        confidence = diagnostic_result.get("confidence", "low")
        reasoning = diagnostic_result.get("reasoning", "")
        diag_think = diagnostic_result.get("think_trace")

        belief = {
            "fault": fault,
            "severity": severity,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        recovery_result = recovery_agent.run(belief, memory.get())
        action = recovery_result.get("action", "no_action")
        recovery_think = recovery_result.get("think_trace")

        action_id = COMMAND_TO_ACTION.get(action, COMMAND_TO_ACTION["no_action"])

        obs, reward, terminated, truncated, info = env.step(action_id)
        total_reward += reward

        fault_hint = None
        if action == "diagnose" and "fault_hint" in info:
            fault_hint = info["fault_hint"]

        summary = f"Step {step_count}: fault={fault}, action={action}, reward={reward:.2f}"
        if fault_hint:
            summary += f", hint={fault_hint}"
        memory.add(step_count, fault, action, reward, hint=fault_hint)

        print(f"STEP {step_count}:")
        if diag_think:
            print(f"  DIAGNOSTIC <think>{diag_think}</think>")
        print(f"  BELIEF: fault={fault} | severity={severity} | confidence={confidence}")
        if recovery_think:
            print(f"  RECOVERY <think>{recovery_think}</think>")
        print(f"  ACTION: {action}")
        print(f"  REWARD: {reward:.3f}")
        print()

    memory_history = memory.get()
    return total_reward, step_count, memory_history
