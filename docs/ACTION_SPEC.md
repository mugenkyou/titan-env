# ACTION_SPEC

## Canonical Discrete Actions

- `0 -> no_action`
- `1 -> reset`
- `2 -> memory_scrub`
- `3 -> load_shedding`
- `4 -> power_cycle`
- `5 -> thermal_throttle`
- `6 -> isolate`
- `7 -> diagnose`

## Action Costs (Raw)

- `no_action: 0.00`
- `reset: 0.50`
- `memory_scrub: 0.40`
- `load_shedding: 0.30`
- `power_cycle: 0.80`
- `thermal_throttle: 0.20`
- `isolate: 1.00`
- `diagnose: 0.35`

## Valid Usage Rules

- `reset`: use for low CPU/comms recovery; avoid in healthy state.
- `memory_scrub`: use for memory degradation or SEU suspicion.
- `load_shedding`: use for high load/thermal trend control.
- `power_cycle`: use for persistent over-current/latch-up behavior.
- `thermal_throttle`: use near thermal threshold.
- `isolate`: use under multi-fault/high-severity conditions.
- `diagnose`: use only when symptom ambiguity blocks confident intervention.

## Invalid or Risky Usage

- Unknown command strings are treated as `no_action` (silent fallback).
- `reset` with no active fault induces avoidable CPU-load side effect.
- `power_cycle` without power fault induces avoidable CPU-load side effect.
- `memory_scrub` during thermal fault adds power-temperature stress.
- `isolate` under low fault severity can cause avoidable battery-SOC drop.
- Repeated high-cost actions without state improvement are policy misuse.

## Side Effects

- `reset`: battery cost, comms penalty, load shedding, latch-up recovery boost.
- `memory_scrub`: memory repair trigger, brief CPU-load increase.
- `load_shedding`: thermal relief through reduced load/current.
- `power_cycle`: hard power reset behavior with battery/comms hit.
- `thermal_throttle`: strongest indirect thermal control via load reduction.
- `isolate`: comms disruption, aggressive recovery effect, load reduction.
- `diagnose`: no direct plant mutation; emits diagnostic hint surface.

## Penalties for Misuse

- Immediate reward penalty from weighted action cost term.
- Secondary penalty via degraded downstream state (battery/comms/load/thermal).
- Indirect strategic penalty: higher probability of terminal failure and large terminal penalty.
