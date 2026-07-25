# REASONING_SPEC

## Required Reasoning Format

Use this exact block before selecting a command:

```text
OBS:
- voltage=<0..1>
- current_draw=<0..1>
- battery=<0..1>
- cpu_temp=<0..1>
- power_temp=<0..1>
- memory=<0..1>
- cpu_load=<0..1>
- signal=<0..1>
- recent_fault_count=<0..1>

UNCERTAINTY:
- suspected_faults=[...]
- confidence=<low|medium|high>

DECISION:
- command=<no_action|reset|memory_scrub|load_shedding|power_cycle|thermal_throttle|isolate|diagnose>

JUSTIFICATION:
- expected_primary_effect=<state variable and direction>
- risk_if_wrong=<one concrete risk>
```

## Trigger Conditions

Emit reasoning block when any condition is true:

- `confidence != high`
- `recent_fault_count >= 0.4`
- `max(cpu_temp, power_temp) >= 0.65`
- `memory <= 0.6`
- `battery <= 0.3`
- prior action produced no measurable improvement for 2 consecutive steps

## Decision Guards

- If uncertainty is high and intervention is expensive, prefer `diagnose` before `power_cycle` or `isolate`.
- Never output free-text commands outside the canonical command set.
- If command parse fails, force `no_action` and mark confidence `low` on next step.
