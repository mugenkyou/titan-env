"""
TITAN Diagnostic Agent

Analyzes satellite telemetry to identify faults and severity.
"""

import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class DiagnosticAgent:
    """
    DiagnosticAgent uses an LLM to analyze telemetry and diagnose satellite faults.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        """
        Parameters
        ----------
        model : Any
            Qwen3 model prepared for inference.
        tokenizer : Any
            Tokenizer for the model.
        """
        self.model = model
        self.tokenizer = tokenizer

    def run(self, obs: np.ndarray, memory: List[str]) -> Dict[str, Any]:
        """
        Diagnose faults from telemetry observation.

        Parameters
        ----------
        obs : np.ndarray
            Observation array of shape (11,):
            [voltage, current_draw, battery_soc, cpu_temperature, power_temperature,
             memory_integrity, cpu_load, delta_voltage, thermal_gradient, memory_drift, step_fraction]
        memory : List[str]
            Last N step summaries (e.g., from Memory.get())

        Returns
        -------
        Dict[str, Any]
            {
                "fault": str,
                "severity": int,
                "confidence": str,
                "reasoning": str,
                "think_trace": str or None,
                "raw_output": str,
            }
        """
        prompt = self._build_prompt(obs, memory)
        raw_output = self._call_model(prompt)
        result = self._parse_output(raw_output)
        result["raw_output"] = raw_output
        return result

    def _build_prompt(self, obs: np.ndarray, memory: List[str]) -> str:
        """Build the diagnostic prompt from telemetry."""
        voltage, current_draw, battery_soc, cpu_temp, power_temp, \
            memory_integrity, cpu_load, delta_voltage, thermal_gradient, \
            memory_drift, step_fraction = obs

        memory_str = "\n".join(memory) if memory else "(empty)"

        prompt = f"""You are a satellite health diagnostic system. Analyze the current telemetry and identify any faults.

Current Telemetry:
- Voltage: {voltage:.4f}
- Current Draw: {current_draw:.4f}
- Battery SoC: {battery_soc:.4f}
- CPU Temperature: {cpu_temp:.4f}
- Power Temperature: {power_temp:.4f}
- Memory Integrity: {memory_integrity:.4f}
- CPU Load: {cpu_load:.4f}
- Delta Voltage: {delta_voltage:.4f}
- Thermal Gradient: {thermal_gradient:.4f}
- Memory Drift: {memory_drift:.4f}
- Step Fraction: {step_fraction:.4f}

Recent History (last 5 steps):
{memory_str}

Output EXACTLY in this format with no extra text before or after:
FAULT: [thermal/memory/power/latchup/none]
SEVERITY: [1/2/3]
CONFIDENCE: [low/medium/high]
REASONING: [one sentence]
"""
        return prompt

    def _call_model(self, prompt: str) -> str:
        """Call the model to generate a response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )

        text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return text

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse model output to extract FAULT, SEVERITY, CONFIDENCE, REASONING.
        """
        result = {
            "fault": "none",
            "severity": 1,
            "confidence": "low",
            "reasoning": "",
            "think_trace": None,
        }

        think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
        if think_match:
            result["think_trace"] = think_match.group(1).strip()
            # Search structured fields ONLY in text after </think>, since
            # Qwen3 may draft FAULT:/SEVERITY: inside the think block
            # before emitting the real answer.
            search_text = raw_output[think_match.end():]
            if not search_text.strip():
                search_text = raw_output
        else:
            search_text = raw_output

        fault_match = re.search(r"FAULT:\s*(\w+)", search_text, re.IGNORECASE)
        if fault_match:
            fault_candidate = fault_match.group(1).lower()
            if fault_candidate in ["thermal", "memory", "power", "latchup", "none"]:
                result["fault"] = fault_candidate

        severity_match = re.search(r"SEVERITY:\s*([123])", search_text, re.IGNORECASE)
        if severity_match:
            result["severity"] = int(severity_match.group(1))

        confidence_match = re.search(r"CONFIDENCE:\s*(\w+)", search_text, re.IGNORECASE)
        if confidence_match:
            conf_candidate = confidence_match.group(1).lower()
            if conf_candidate in ["low", "medium", "high"]:
                result["confidence"] = conf_candidate

        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", search_text, re.IGNORECASE)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()

        return result
