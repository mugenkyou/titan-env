"""
TITAN Recovery Agent

Generates recovery actions based on diagnostic belief state using Qwen3.
"""

from typing import Any, Dict, List, Optional
import re


class RecoveryAgent:
    """
    RecoveryAgent uses an LLM to select recovery actions based on diagnostic beliefs
    and recent memory of step outcomes.
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
        self.valid_actions = [
            "no_action",
            "reset",
            "memory_scrub",
            "load_shedding",
            "power_cycle",
            "thermal_throttle",
            "isolate",
            "diagnose",
        ]

    def run(self, belief: Dict[str, Any], memory: List[str]) -> Dict[str, Any]:
        """
        Generate a recovery action based on diagnostic belief and memory.

        Parameters
        ----------
        belief : Dict[str, Any]
            Diagnostic state with keys: fault, severity, confidence, reasoning
        memory : List[str]
            Last N step summaries (e.g., from Memory.get())

        Returns
        -------
        Dict[str, Any]
            {
                "thought": str,
                "action": str,
                "think_trace": str or None,
                "raw_output": str,
            }
        """
        prompt = self._build_prompt(belief, memory)
        raw_output = self._call_model(prompt)
        result = self._parse_output(raw_output)
        result["raw_output"] = raw_output
        return result

    def _build_prompt(self, belief: Dict[str, Any], memory: List[str]) -> str:
        """Build the recovery agent prompt."""
        memory_str = "\n".join(memory) if memory else "(empty)"

        guidelines = """
Guidelines:
- thermal fault → use thermal_throttle
- memory fault → use memory_scrub
- power fault → use power_cycle
- latchup → use isolate or reset
- ambiguous or low confidence → use diagnose
- no fault detected → use no_action
- severity 3 (critical) → use isolate immediately

Valid actions: no_action, reset, memory_scrub, load_shedding, power_cycle, thermal_throttle, isolate, diagnose
"""

        prompt = f"""You are a satellite fault recovery decision system. Based on the diagnostic belief and recent history, decide the next recovery action.

Current Diagnostic Belief:
- Fault: {belief.get('fault', 'unknown')}
- Severity: {belief.get('severity', 'unknown')}
- Confidence: {belief.get('confidence', 'unknown')}
- Reasoning: {belief.get('reasoning', '')}

Recent History (last 5 steps):
{memory_str}

{guidelines}

Think carefully before answering. Output EXACTLY in this format:
THOUGHT: [one sentence reasoning]
ACTION: [exactly one action name from the list above]
"""
        return prompt

    def _call_model(self, prompt: str) -> str:
        """Call the model to generate a response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
        )

        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse model output to extract THOUGHT, ACTION, and optional <think> block.
        """
        result = {
            "thought": "",
            "action": "no_action",
            "think_trace": None,
        }

        think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
        if think_match:
            result["think_trace"] = think_match.group(1).strip()

        thought_match = re.search(r"THOUGHT:\s*(.+?)(?:\n|$)", raw_output, re.IGNORECASE)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        action_match = re.search(r"ACTION:\s*(\w+)", raw_output, re.IGNORECASE)
        if action_match:
            action_candidate = action_match.group(1).lower()
            if action_candidate in self.valid_actions:
                result["action"] = action_candidate
            else:
                result["action"] = "no_action"

        return result
