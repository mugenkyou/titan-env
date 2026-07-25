# REWARD_LOGIC

## Reward Equation

Per-step reward:

`R = 1.5*uptime - 1.0*fault_severity - 0.1*energy_usage - 0.5*recovery_latency - 0.05*action_cost + failure_penalty`

Terminal failure penalty:

`failure_penalty = -30.0 if terminated else 0.0`

## Component Definitions

- `uptime in {0,1}`: `1` only when all fault flags are inactive.
- `fault_severity in [0,1]`: normalized blend of active fault-flag count and thermal excess.
- `energy_usage in [0,1]`: clamped `current_draw`.
- `recovery_latency in [0,1]`: normalized distance from nominal `(cpu_temp=0.30, power_temp=0.25, battery_soc=0.90)`.
- `action_cost in [0,1]`: `raw_action_cost / 1.0` where `1.0` is max raw cost.

## Numerical Bounds

- Maximum non-terminal per-step reward: `+1.5`.
- Typical non-terminal range under faults/actions: approximately `[-3, +1.5]`.
- Failure step adds `-30.0` on top of weighted terms.

## Interpretation Rules

- Positive reward requires low severity, low latency, and controlled action spending.
- High-cost actions are acceptable only when they materially reduce future severity/termination risk.
- `no_action` is free but can be net-negative under escalating latent faults.
