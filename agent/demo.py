"""
TITAN Multi-Agent Fault Recovery Demo

Demonstrates the diagnostic and recovery agents working together on a satellite
in a simulated fault environment with three difficulty levels.
"""

import sys
import os
from io import StringIO
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastLanguageModel
from TITAN_env.core.environment.gym_env import TITANGymEnv
from TITAN_env.core.environment.fault_injection import FaultInjector, INTENSITY_PROFILES
from agent.diagnostic_agent import DiagnosticAgent
from agent.recovery_agent import RecoveryAgent
from agent.memory import Memory
from agent.run_episode import run_episode


def main() -> None:
    """Run the multi-agent demonstration."""
    
    output_buffer = StringIO()
    
    print("Loading Qwen3-1.7B model...", file=output_buffer)
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen3-1.7B",
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.", file=output_buffer)
    
    diagnostic_agent = DiagnosticAgent(model, tokenizer)
    recovery_agent = RecoveryAgent(model, tokenizer)
    
    profiles = [
        ("low", INTENSITY_PROFILES["low"]),
        ("medium", INTENSITY_PROFILES["medium"]),
        ("high", INTENSITY_PROFILES["high"]),
    ]
    
    for episode_num, (profile_name, profile) in enumerate(profiles, 1):
        print(f"\n{'='*60}", file=output_buffer)
        print(f"=== EPISODE {episode_num} — {profile_name} radiation ===", file=output_buffer)
        print(f"{'='*60}\n", file=output_buffer)
        
        env = TITANGymEnv(
            fault_injector=FaultInjector(profile),
            reward_version="v3",
            training_mode=False,
        )
        
        memory = Memory(max_size=5)
        
        obs, info = env.reset()
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        
        while not (done or truncated):
            step_count += 1
            
            telemetry = _extract_telemetry(obs)
            print(f"STEP {step_count}:", file=output_buffer)
            print(f"  TELEMETRY: {telemetry}", file=output_buffer)
            print(file=output_buffer)
            
            diagnostic_result = diagnostic_agent.run(obs)
            fault = diagnostic_result.get("fault", "unknown")
            severity = diagnostic_result.get("severity", 0)
            confidence = diagnostic_result.get("confidence", 0.0)
            think_trace = diagnostic_result.get("think_trace")
            
            print("  DIAGNOSTIC AGENT:", file=output_buffer)
            if think_trace:
                print(f"  <think>{think_trace}</think>", file=output_buffer)
            print(f"  FAULT: {fault} | SEVERITY: {severity} | CONFIDENCE: {confidence:.2f}", file=output_buffer)
            print(file=output_buffer)
            
            belief = {
                "fault": fault,
                "severity": severity,
                "confidence": confidence,
                "reasoning": diagnostic_result.get("reasoning", ""),
            }
            
            recovery_result = recovery_agent.run(belief, memory.get())
            action = recovery_result.get("action", "no_action")
            recovery_think = recovery_result.get("think_trace")
            
            print("  RECOVERY AGENT:", file=output_buffer)
            if recovery_think:
                print(f"  <think>{recovery_think}</think>", file=output_buffer)
            print(f"  ACTION: {action}", file=output_buffer)
            print(file=output_buffer)
            
            from TITAN_env.interface.action_mapping import discrete_from_command
            action_id = discrete_from_command(action)
            
            obs, reward, done, truncated, info = env.step(action_id)
            total_reward += reward
            
            print(f"  ENV: reward={reward:.3f}", file=output_buffer)
            print(f"  {'─'*50}", file=output_buffer)
            
            memory.add(step_count, fault, action, reward)
        
        print(f"\n=== EPISODE END: survived {step_count} steps, total_reward={total_reward:.2f} ===\n", file=output_buffer)
        
        env.close()
    
    output_text = output_buffer.getvalue()
    print(output_text)
    
    with open("demo_output.txt", "w") as f:
        f.write(output_text)
    
    print("Output saved to demo_output.txt")


def _extract_telemetry(obs: Any) -> str:
    """Extract readable telemetry from observation array."""
    cpu_temp = obs[3] if len(obs) > 3 else 0.0
    thermal_gradient = obs[8] if len(obs) > 8 else 0.0
    memory_drift = obs[9] if len(obs) > 9 else 0.0
    return f"cpu_temp={cpu_temp:.3f}, thermal_gradient={thermal_gradient:.3f}, memory_drift={memory_drift:.3f}"


if __name__ == "__main__":
    main()
