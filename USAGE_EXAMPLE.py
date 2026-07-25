#!/usr/bin/env python3
"""
TITAN Agent System - Usage Example

This script demonstrates how to use the diagnostic agent, recovery agent,
memory system, and episode runner together in a complete workflow.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.diagnostic_agent import DiagnosticAgent
from agent.recovery_agent import RecoveryAgent
from agent.memory import Memory
from agent.run_episode import run_episode
from TITAN_env.interface.action_mapping import COMMAND_TO_ACTION

print("=" * 80)
print("TITAN Multi-Agent Fault Recovery System - Usage Example")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Initialize the agents
# ============================================================================
print("EXAMPLE 1: Initializing Agents")
print("-" * 80)
print("""
# Load your Qwen3 model (this example shows the import pattern)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-1.7B",
    max_seq_length=1024,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Create agent instances
diagnostic_agent = DiagnosticAgent(model, tokenizer)
recovery_agent = RecoveryAgent(model, tokenizer)
memory = Memory(max_size=5)
""")

print("✓ Agents initialized (pseudocode shown above)")
print()

# ============================================================================
# EXAMPLE 2: Diagnostic Agent - Analyze Telemetry
# ============================================================================
print("EXAMPLE 2: Diagnostic Agent - Analyze Telemetry")
print("-" * 80)

# Create sample telemetry data
obs = np.array([
    0.85,    # voltage
    1.2,     # current_draw
    0.95,    # battery_soc
    0.85,    # cpu_temperature (elevated)
    0.75,    # power_temperature (elevated)
    0.88,    # memory_integrity
    0.75,    # cpu_load
    0.05,    # delta_voltage
    0.10,    # thermal_gradient (high)
    0.03,    # memory_drift
    0.25,    # step_fraction
])

print("Telemetry received:")
print(f"  voltage={obs[0]:.2f}")
print(f"  cpu_temperature={obs[3]:.2f} (elevated)")
print(f"  thermal_gradient={obs[8]:.2f} (high)")
print()

# Create diagnostic agent (without actual model for demo)
diag_agent = DiagnosticAgent(None, None)

# Show the prompt that would be sent to the model
memory_history = ["Step 1: fault=none, action=no_action, reward=1.0"]
prompt = diag_agent._build_prompt(obs, memory_history)

print("Diagnostic prompt structure:")
print("-" * 40)
print(prompt[:300] + "...")
print("-" * 40)
print()

# Simulate model response
simulated_response = """<think>CPU temperature is 0.85 and thermal gradient is 0.10, 
indicating significant thermal stress. This requires immediate cooling action.</think>
FAULT: thermal
SEVERITY: 2
CONFIDENCE: high
REASONING: CPU and power temperatures are elevated with high thermal gradient."""

print("Simulated model response:")
print(simulated_response)
print()

# Parse the response
belief = diag_agent._parse_output(simulated_response)
print("Parsed diagnostic result:")
print(f"  fault: {belief['fault']}")
print(f"  severity: {belief['severity']}")
print(f"  confidence: {belief['confidence']}")
print(f"  reasoning: {belief['reasoning']}")
print(f"  think_trace: {belief['think_trace'][:50]}...")
print()

# ============================================================================
# EXAMPLE 3: Recovery Agent - Select Action
# ============================================================================
print("EXAMPLE 3: Recovery Agent - Select Action")
print("-" * 80)

recovery_agent = RecoveryAgent(None, None)

# Use the belief from diagnostic agent
belief = {
    "fault": "thermal",
    "severity": 2,
    "confidence": "high",
    "reasoning": "CPU and power temperatures are elevated with high thermal gradient."
}

print("Using belief state:")
print(f"  fault={belief['fault']}")
print(f"  severity={belief['severity']}")
print(f"  confidence={belief['confidence']}")
print()

# Show the prompt sent to recovery agent
prompt = recovery_agent._build_prompt(belief, memory_history)
print("Recovery prompt structure:")
print("-" * 40)
print(prompt[:400] + "...")
print("-" * 40)
print()

# Simulate model response
simulated_response = """<think>Thermal fault with severity 2 requires cooling. 
The guidelines say thermal fault → thermal_throttle.</think>
THOUGHT: High thermal stress requires immediate reduction of thermal load.
ACTION: thermal_throttle"""

print("Simulated model response:")
print(simulated_response)
print()

# Parse the response
action_result = recovery_agent._parse_output(simulated_response)
print("Parsed recovery result:")
print(f"  action: {action_result['action']}")
print(f"  thought: {action_result['thought']}")
print(f"  think_trace: {action_result['think_trace'][:60]}...")
print()

# ============================================================================
# EXAMPLE 4: Memory - Track History
# ============================================================================
print("EXAMPLE 4: Memory - Track Step History")
print("-" * 80)

memory = Memory(max_size=5)

# Add step summaries
memory.add(1, "none", "no_action", 1.0)
memory.add(2, "thermal", "thermal_throttle", 1.5, hint="Cooling initiated")
memory.add(3, "thermal", "thermal_throttle", 1.3)

print("Memory contents after 3 steps:")
for entry in memory.get():
    print(f"  {entry}")
print()

print("Formatted memory (as single string):")
print("-" * 40)
print(memory.get_formatted())
print("-" * 40)
print()

# ============================================================================
# EXAMPLE 5: Action Mapping
# ============================================================================
print("EXAMPLE 5: Action Mapping - Convert to Environment IDs")
print("-" * 80)

action_name = "thermal_throttle"
action_id = COMMAND_TO_ACTION[action_name]

print(f"Action: {action_name}")
print(f"Environment ID: {action_id}")
print()

print("All available actions:")
for cmd, aid in COMMAND_TO_ACTION.items():
    print(f"  {cmd:20s} → {aid}")
print()

# ============================================================================
# EXAMPLE 6: Full Episode Loop
# ============================================================================
print("EXAMPLE 6: Full Episode Loop Structure")
print("-" * 80)

code_example = """
# Initialize components
diag_agent = DiagnosticAgent(model, tokenizer)
recovery_agent = RecoveryAgent(model, tokenizer)
memory = Memory(max_size=5)

# Run episode
total_reward, steps, history = run_episode(
    env=env,
    diagnostic_agent=diag_agent,
    recovery_agent=recovery_agent,
    memory=memory,
    max_steps=1000
)

# Results
print(f"Episode completed!")
print(f"  Steps survived: {steps}")
print(f"  Total reward: {total_reward:.2f}")
print(f"  History entries: {len(history)}")
for entry in history[-5:]:  # Show last 5 steps
    print(f"    {entry}")
"""

print(code_example)
print()

# ============================================================================
# EXAMPLE 7: Step-by-Step Episode Walkthrough
# ============================================================================
print("EXAMPLE 7: Manual Step-by-Step Episode Loop")
print("-" * 80)

manual_loop = """
# Initialize
obs, info = env.reset()
memory = Memory()
total_reward = 0.0
step = 0

# Main episode loop
while step < 1000:
    step += 1
    
    # 1. Diagnostic analysis
    diag_result = diagnostic_agent.run(obs, memory.get())
    belief = {
        "fault": diag_result["fault"],
        "severity": diag_result["severity"],
        "confidence": diag_result["confidence"],
        "reasoning": diag_result["reasoning"],
    }
    
    # 2. Recovery decision
    action_result = recovery_agent.run(belief, memory.get())
    action_name = action_result["action"]
    
    # 3. Convert to environment action ID
    action_id = COMMAND_TO_ACTION[action_name]
    
    # 4. Execute step
    obs, reward, terminated, truncated, info = env.step(action_id)
    total_reward += reward
    
    # 5. Extract hint from diagnose if applicable
    fault_hint = info.get("fault_hint") if action_name == "diagnose" else None
    
    # 6. Update memory
    memory.add(step, belief["fault"], action_name, reward, hint=fault_hint)
    
    # 7. Optional: Print step info
    print(f"Step {step}: {belief['fault']} → {action_name}, reward={reward:.2f}")
    
    # 8. Check termination
    if terminated or truncated:
        break

print(f"Episode end: {step} steps, total reward: {total_reward:.2f}")
"""

print(manual_loop)
print()

# ============================================================================
# EXAMPLE 8: Using run_episode Function
# ============================================================================
print("EXAMPLE 8: Using the run_episode Helper Function")
print("-" * 80)

simple_example = """
from agent.run_episode import run_episode

# One-line episode execution
total_reward, steps, history = run_episode(
    env=my_env,
    diagnostic_agent=diagnostic_agent,
    recovery_agent=recovery_agent,
    memory=memory,
    max_steps=1000
)

# The run_episode function handles:
# ✓ Reset environment
# ✓ Call diagnostic agent for belief
# ✓ Call recovery agent for action
# ✓ Convert action to environment ID
# ✓ Execute environment step
# ✓ Update memory with hint from diagnose
# ✓ Print formatted output
# ✓ Handle termination/truncation
"""

print(simple_example)
print()

# ============================================================================
# EXAMPLE 9: Complete Integration Example
# ============================================================================
print("EXAMPLE 9: Complete Integration - Full Script Template")
print("-" * 80)

full_template = """
import numpy as np
from unsloth import FastLanguageModel

from agent.diagnostic_agent import DiagnosticAgent
from agent.recovery_agent import RecoveryAgent
from agent.memory import Memory
from agent.run_episode import run_episode
from TITAN_env.core.environment.gym_env import TITANGymEnv
from TITAN_env.core.environment.fault_injection import FaultInjector, INTENSITY_PROFILES

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-1.7B",
    max_seq_length=1024,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Create environment
env = TITANGymEnv(
    fault_injector=FaultInjector(INTENSITY_PROFILES["medium"]),
    reward_version="v3",
    training_mode=False,
)

# Create agents
diagnostic_agent = DiagnosticAgent(model, tokenizer)
recovery_agent = RecoveryAgent(model, tokenizer)
memory = Memory(max_size=5)

# Run episode
print("Starting episode...")
total_reward, steps, history = run_episode(
    env=env,
    diagnostic_agent=diagnostic_agent,
    recovery_agent=recovery_agent,
    memory=memory,
    max_steps=1000,
)

# Show results
print(f"\\nEpisode Results:")
print(f"  Steps survived: {steps}")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Average reward per step: {total_reward/steps:.3f}")

env.close()
"""

print(full_template)
print()

# ============================================================================
# EXAMPLE 10: Debugging & Analysis
# ============================================================================
print("EXAMPLE 10: Debugging & Analysis")
print("-" * 80)

debug_example = """
# Examine diagnostic agent's raw output
diag_result = diagnostic_agent.run(obs, memory.get())
print("Diagnostic raw output:")
print(diag_result["raw_output"])
print()

# Examine recovery agent's raw output
recovery_result = recovery_agent.run(belief, memory.get())
print("Recovery raw output:")
print(recovery_result["raw_output"])
print()

# Examine thinking process
if diag_result["think_trace"]:
    print("Diagnostic thinking:")
    print(diag_result["think_trace"])
print()

if recovery_result["think_trace"]:
    print("Recovery thinking:")
    print(recovery_result["think_trace"])
print()

# Analyze memory history
print("Recent episode history:")
for entry in memory.get_formatted().split("\\n"):
    print(f"  {entry}")
"""

print(debug_example)
print()

print("=" * 80)
print("Summary: TITAN Agent System Usage")
print("=" * 80)
print("""
Key Components:
1. DiagnosticAgent - Analyzes telemetry → Fault diagnosis
2. RecoveryAgent - Considers belief → Action selection
3. Memory - Maintains recent history for context
4. run_episode - Orchestrates full episode execution

Quick Start:
  from agent.run_episode import run_episode
  total_reward, steps, history = run_episode(env, diag, recovery, memory)

For Full Demo:
  python agent/demo.py

For Testing:
  python test_full_integration.py
""")
print()
