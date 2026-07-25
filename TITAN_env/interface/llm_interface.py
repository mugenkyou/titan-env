"""
TITAN — Phase 2: LLM Interface Layer
TITAN_environment/llm_interface.py

Provides language-based interpretation of observations and parsing of natural
language actions into valid commands. This layer sits on top of the OpenEnv
wrapper without modifying any core simulation logic.

Components:
  1. Observation Renderer: Converts structured observations to human-readable text
  2. Action Vocabulary: Maps natural language synonyms to canonical commands
  3. Action Parser: Extracts valid commands from free-form LLM output
  4. LLM Step Adapter: Helper to bridge LLM output directly to environment steps
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from TITAN_env.interface.models import Action, Observation, Reward


# ============================================================================
# 1. ACTION VOCABULARY LAYER
# ============================================================================
# Canonical action commands mapped to synonyms.
# Each key is the canonical command; each value is a list of accepted aliases.

ACTION_VOCABULARY: Dict[str, List[str]] = {
    "no_action": [
        "no_action",
        "no action",
        "do nothing",
        "wait",
        "idle",
        "noop",
    ],
    "reset": [
        "reset",
        "restart",
        "reboot",
        "system reset",
        "cold start",
    ],
    "memory_scrub": [
        "memory_scrub",
        "scrub",
        "memory scrub",
        "scrub memory",
        "fix memory",
        "repair memory",
        "memory repair",
    ],
    "load_shedding": [
        "load_shedding",
        "reduce load",
        "lower cpu",
        "shed load",
        "reduce cpu",
        "lower workload",
        "reduce workload",
    ],
    "power_cycle": [
        "power_cycle",
        "power cycle",
        "cycle power",
        "toggle power",
        "off and on",
    ],
    "thermal_throttle": [
        "thermal_throttle",
        "thermal throttle",
        "throttle",
        "reduce heat",
        "cool down",
        "slow cpu",
        "cpu throttle",
    ],
    "isolate": [
        "isolate",
        "quarantine",
        "isolate subsystem",
        "fault isolation",
        "isolation",
    ],
}

_ACTION_PRIORITY: Dict[str, int] = {
    "thermal_throttle": 0,
    "memory_scrub": 1,
    "power_cycle": 2,
    "load_shedding": 3,
    "reset": 4,
    "isolate": 5,
    "no_action": 6,
}


# ============================================================================
# 2. OBSERVATION RENDERER
# ============================================================================


def _get_battery_interpretation(battery: float) -> str:
    """Qualitative interpretation of battery level."""
    if battery >= 0.75:
        return "HEALTHY"
    elif battery >= 0.5:
        return "NORMAL"
    elif battery >= 0.25:
        return "LOW"
    else:
        return "CRITICAL"


def _get_temperature_interpretation(cpu_temp: float, power_temp: float) -> str:
    """Qualitative interpretation of temperature readings."""
    max_temp = max(cpu_temp, power_temp)
    if max_temp >= 0.8:
        return "CRITICAL"
    elif max_temp >= 0.6:
        return "HIGH"
    elif max_temp >= 0.3:
        return "NORMAL"
    else:
        return "LOW"


def _get_memory_interpretation(memory: float) -> str:
    """Qualitative interpretation of memory integrity."""
    if memory >= 0.8:
        return "HEALTHY"
    elif memory >= 0.6:
        return "NORMAL"
    elif memory >= 0.3:
        return "DEGRADING"
    else:
        return "CRITICAL"


def _get_signal_interpretation(signal: float) -> str:
    """Qualitative interpretation of signal quality."""
    if signal >= 0.75:
        return "STRONG"
    elif signal >= 0.5:
        return "NORMAL"
    elif signal >= 0.25:
        return "WEAK"
    else:
        return "LOST"


def _get_load_interpretation(cpu_load: float) -> str:
    """Qualitative interpretation of CPU load."""
    if cpu_load >= 0.8:
        return "CRITICAL"
    elif cpu_load >= 0.6:
        return "HIGH"
    elif cpu_load >= 0.3:
        return "NORMAL"
    else:
        return "LOW"


def _severity_label(value: float, inverse: bool = False) -> str:
    """Map normalized values to LOW/MEDIUM/HIGH/CRITICAL consistently."""
    x = max(0.0, min(1.0, float(value)))
    if inverse:
        x = 1.0 - x

    if x >= 0.8:
        return "CRITICAL"
    if x >= 0.6:
        return "HIGH"
    if x >= 0.35:
        return "MEDIUM"
    return "LOW"


def _fault_consequence(fault: str) -> str:
    mapping = {
        "seu": "memory corruption risk",
        "memory": "memory corruption risk",
        "thermal": "thermal failure risk",
        "latchup": "power drain and heat spike risk",
        "power": "power loss risk",
    }
    return mapping.get(fault.lower(), "system instability risk")


def _suggest_actions(obs: Observation) -> List[str]:
    """Recommend a compact set of corrective actions from current state."""
    scores: Dict[str, int] = {}

    for fault in obs.faults:
        fault_key = fault.lower()
        if fault_key in {"seu", "memory"}:
            scores["memory_scrub"] = scores.get("memory_scrub", 0) + 3
        elif fault_key in {"latchup", "power"}:
            scores["power_cycle"] = scores.get("power_cycle", 0) + 3
        elif fault_key == "thermal":
            scores["thermal_throttle"] = scores.get("thermal_throttle", 0) + 3

    if obs.cpu_temp >= 0.65 or obs.power_temp >= 0.65:
        scores["thermal_throttle"] = scores.get("thermal_throttle", 0) + 3
    if obs.cpu_load >= 0.65:
        scores["load_shedding"] = scores.get("load_shedding", 0) + 2
    if obs.memory <= 0.55:
        scores["memory_scrub"] = scores.get("memory_scrub", 0) + 2
    if obs.battery <= 0.30:
        scores["load_shedding"] = scores.get("load_shedding", 0) + 1
        scores["power_cycle"] = scores.get("power_cycle", 0) + 1
    if len(obs.faults) >= 2 or obs.recent_fault_count >= 0.7:
        scores["isolate"] = scores.get("isolate", 0) + 2
    if obs.signal <= 0.3:
        scores["reset"] = scores.get("reset", 0) + 1

    if not scores:
        return ["no_action"]

    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], _ACTION_PRIORITY.get(kv[0], 99), kv[0]),
    )
    return [name for name, _ in ranked[:3]]


def render_observation(obs: Observation) -> str:
    """
    Convert a structured Observation into human-readable semantic text.

    This rendition is designed for LLM comprehension and includes:
      - Battery level with qualitative interpretation
      - Temperature readings with severity assessment
      - Memory integrity with health status
      - Signal quality with connection status
      - CPU load assessment
      - Active faults (if any)

    Args:
        obs: Structured Observation from the TITAN wrapper

    Returns:
        Human-readable string representation of system state
    """
    battery_pct = round(obs.battery * 100, 1)
    battery_status = _get_battery_interpretation(obs.battery)

    cpu_temp_pct = round(obs.cpu_temp * 100, 1)
    power_temp_pct = round(obs.power_temp * 100, 1)
    temp_status = _get_temperature_interpretation(obs.cpu_temp, obs.power_temp)

    memory_pct = round(obs.memory * 100, 1)
    memory_status = _get_memory_interpretation(obs.memory)

    signal_pct = round(obs.signal * 100, 1)
    signal_status = _get_signal_interpretation(obs.signal)

    cpu_load_pct = round(obs.cpu_load * 100, 1)
    load_status = _get_load_interpretation(obs.cpu_load)

    voltage_pct = round(obs.voltage * 100, 1)
    current_draw_pct = round(obs.current_draw * 100, 1)

    fault_count = round(obs.recent_fault_count * 10, 1)

    battery_severity = _severity_label(obs.battery, inverse=True)
    temp_severity = _severity_label(max(obs.cpu_temp, obs.power_temp))
    memory_severity = _severity_label(obs.memory, inverse=True)
    signal_severity = _severity_label(obs.signal, inverse=True)
    load_severity = _severity_label(obs.cpu_load)

    suggested_actions = _suggest_actions(obs)

    causal_notes: List[str] = []
    if obs.cpu_load >= 0.65 and max(obs.cpu_temp, obs.power_temp) >= 0.55:
        causal_notes.append("high CPU load is driving thermal rise")
    if obs.memory <= 0.55 and any(f in {"seu", "memory"} for f in obs.faults):
        causal_notes.append("fault activity is degrading memory integrity")
    if obs.battery <= 0.30 and obs.current_draw >= 0.6:
        causal_notes.append("high current draw is accelerating battery drain")

    risk_score = 0
    if battery_severity in {"HIGH", "CRITICAL"}:
        risk_score += 1
    if temp_severity in {"HIGH", "CRITICAL"}:
        risk_score += 2
    if memory_severity in {"HIGH", "CRITICAL"}:
        risk_score += 1
    if signal_severity in {"HIGH", "CRITICAL"}:
        risk_score += 1
    if len(obs.faults) > 0:
        risk_score += min(len(obs.faults), 2)

    if risk_score >= 5:
        system_risk = "CRITICAL"
        system_state = "unstable"
    elif risk_score >= 3:
        system_risk = "HIGH"
        system_state = "degrading"
    elif risk_score >= 1:
        system_risk = "MEDIUM"
        system_state = "watch"
    else:
        system_risk = "LOW"
        system_state = "stable"

    # Build the lines
    lines = []
    battery_hint = "risk of power failure" if battery_severity in {"HIGH", "CRITICAL"} else "power margin acceptable"
    lines.append(
        f"Battery: {battery_pct}% ({battery_status}, Severity={battery_severity}; {battery_hint})"
    )
    temp_hint = "risk of thermal failure" if temp_severity in {"HIGH", "CRITICAL"} else "thermal envelope acceptable"
    lines.append(
        f"Temperature: CPU {cpu_temp_pct}°C, Power {power_temp_pct}°C ({temp_status}, Severity={temp_severity}; {temp_hint})"
    )
    memory_hint = "risk of memory corruption" if memory_severity in {"HIGH", "CRITICAL"} else "memory health acceptable"
    lines.append(
        f"Memory Integrity: {memory_pct}% ({memory_status}, Severity={memory_severity}; {memory_hint})"
    )
    signal_hint = "link unstable" if signal_severity in {"HIGH", "CRITICAL"} else "link stable"
    lines.append(
        f"Signal Quality: {signal_pct}% ({signal_status}, Severity={signal_severity}; {signal_hint})"
    )
    load_hint = "high load may trigger thermal rise" if load_severity in {"HIGH", "CRITICAL"} else "load within expected range"
    lines.append(
        f"CPU Load: {cpu_load_pct}% ({load_status}, Severity={load_severity}; {load_hint})"
    )
    lines.append(f"Voltage: {voltage_pct}%")
    lines.append(f"Current Draw: {current_draw_pct}%")
    lines.append(f"Recent Faults: {fault_count}")

    if obs.faults:
        faults_str = ", ".join([f.upper() for f in obs.faults])
        consequences = ", ".join(_fault_consequence(f) for f in obs.faults)
        lines.append(f"Active Faults: {faults_str} (consequence: {consequences})")
    else:
        lines.append("Active Faults: NONE")

    lines.append(f"System Risk: {system_risk} ({system_state})")
    if causal_notes:
        lines.append(f"Causal Signals: {'; '.join(causal_notes)}")
    lines.append(f"Suggested Actions: {', '.join(suggested_actions)}")

    return "\n".join(lines)


# ============================================================================
# 3. ACTION PARSER
# ============================================================================


def _normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, strip, remove punctuation."""
    # Convert to lowercase and strip whitespace
    normalized = text.strip().lower()

    # Remove common punctuation and formatting
    normalized = re.sub(r'["\']', "", normalized)  # Remove quotes
    normalized = re.sub(r"[`*_#]", "", normalized)  # Remove markdown wrappers
    normalized = re.sub(r"[^a-z0-9_\s-]", " ", normalized)  # Remove punctuation
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)  # Collapse whitespace

    return normalized.strip()


def _find_matching_command(text: str) -> Optional[str]:
    """
    Search text for any recognized action vocabulary term.

    Tries to match the text against all known action synonyms.
    Returns the canonical command if found, None otherwise.

    Uses a greedy approach: longer matches take precedence.
    Handles word-level matching to deal with intervening words.
    """
    normalized = _normalize_text(text)

    if not normalized:
        return None

    # Split into words for word-level matching
    text_words = set(normalized.split())

    # Find all matching canonical commands
    matches: List[Tuple[str, int, int]] = []  # (command, score, match_length)

    for canonical, synonyms in ACTION_VOCABULARY.items():
        for synonym in synonyms:
            normalized_synonym = _normalize_text(synonym)
            synonym_words = normalized_synonym.split()
            if not normalized_synonym:
                continue

            # Try two matching strategies:
            # 1. Exact substring match (strict)
            if normalized_synonym in normalized:
                score = 100
                matches.append((canonical, score, len(normalized_synonym)))
            # 2. Word-level match - all synonym words appear in text (lenient)
            elif len(synonym_words) > 0 and all(
                word in text_words for word in synonym_words
            ):
                score = 50
                matches.append((canonical, score, len(normalized_synonym)))

    # Return the best command by score, match specificity, then fixed priority.
    if matches:
        matches.sort(
            key=lambda x: (
                -x[1],
                -x[2],
                _ACTION_PRIORITY.get(x[0], 99),
                x[0],
            )
        )
        return matches[0][0]

    return None


def parse_action(text: str) -> Action:
    """
    Extract a valid action command from free-form LLM output.

    This function is robust to:
      - Extra text before/after the command
      - Multiple sentences
      - Formatting variations (punctuation, capitalization)
      - Common phrasing patterns

    If no valid command is found, returns a safe no_action.

    Args:
        text: Free-form text output from an LLM or human

    Returns:
        Action with a valid canonical command string

    Example:
        parse_action("I think we should reset the system")
        -> Action(command="reset")

        parse_action("random gibberish")
        -> Action(command="no_action")
    """
    if not text or not isinstance(text, str):
        return Action(command="no_action")

    # Try to find a matching command
    found_command = _find_matching_command(text)

    if found_command:
        return Action(command=found_command)

    # No valid command found; return safe fallback
    return Action(command="no_action")


# ============================================================================
# 4. LLM STEP ADAPTER
# ============================================================================


def llm_step(
    env,
    llm_output_text: str,
) -> Tuple[str, float, bool, Dict]:
    """
    Bridge function to connect LLM output directly to environment steps.

    This high-level helper:
      1. Parses the LLM output as a natural language action command
      2. Calls env.step() with the parsed action
      3. Renders the resulting observation as human-readable text
      4. Returns (observation_text, reward, done, info)

    This is NOT replacing env.step() — it's an optional convenience layer
    that handles parsing and rendering for LLM workflows.

    Args:
        env: TITANEnv wrapper instance
        llm_output_text: Free-form text from an LLM or agent

    Returns:
        Tuple of (rendered_observation, reward_value, done, info_dict)

    Example:
        obs_text, reward, done, info = llm_step(env, "please reset the system")
        # obs_text is now a human-readable string
        # reward is a float
        # done is a bool
        # info is a dict with additional metadata
    """
    # Parse the LLM output into an Action
    action = parse_action(llm_output_text)

    # Execute the environment step
    observation, reward, done, info = env.step(action)

    # Render the observation to human-readable text
    observation_text = render_observation(observation)

    # Return as a tuple suitable for LLM processing
    return observation_text, reward.value, done, info


# ============================================================================
# 5. HELPER: GET AVAILABLE_COMMANDS
# ============================================================================


def get_available_commands() -> List[str]:
    """
    Return list of all canonical action commands.

    Useful for:
      - Documenting valid actions to LLMs
      - Validating action output programmatically
      - Building action prompts

    Returns:
        List of canonical command strings (e.g., ["no_action", "reset", ...])
    """
    return list(ACTION_VOCABULARY.keys())


def get_action_synonyms(canonical_command: str) -> Optional[List[str]]:
    """
    Get all accepted synonyms for a canonical command.

    Args:
        canonical_command: The canonical action command (e.g., "reset")

    Returns:
        List of accepted synonyms, or None if command not recognized
    """
    return ACTION_VOCABULARY.get(canonical_command)
