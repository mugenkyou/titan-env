# OBSERVATION_SPEC

## Interface Observation Shape

Top-level object with 10 fields:

- `voltage: float`
- `current_draw: float`
- `battery: float`
- `cpu_temp: float`
- `power_temp: float`
- `memory: float`
- `cpu_load: float`
- `signal: float`
- `recent_fault_count: float`
- `faults: list[str]`

## Variable Semantics

- `voltage`: normalized power bus level.
- `current_draw`: normalized instantaneous electrical demand.
- `battery`: normalized battery state-of-charge (`battery_soc` projection).
- `cpu_temp`: normalized CPU thermal load.
- `power_temp`: normalized power subsystem thermal load.
- `memory`: normalized memory integrity.
- `cpu_load`: normalized compute utilization.
- `signal`: derived proxy `1.0 - current_draw` after clamping.
- `recent_fault_count`: normalized recent fault pressure on `[0, 1]`.
- `faults`: reserved diagnostic list; currently emitted as empty list.

## Normalization Rules

- All scalar fields are clamped to `[0.0, 1.0]` in interface projection.
- `None`, `NaN`, and infinite values are converted to `0.0`.
- No telemetry denormalization is allowed at interface boundary.

## Hidden State Policy

- Root fault variables are latent by default: `seu_flag`, `latchup_flag`, `thermal_fault_flag`, `memory_fault_flag`, `power_fault_flag` are not exposed in standard interface observations.
- Per-fault severities are latent by default and not included in standard observation payload.
- Fault-bearing keys are removed from standard info payload except explicit diagnostic surfaces.
