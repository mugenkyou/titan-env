# PITCH_CONTEXT

## Problem Framing

- This is not a fully observed control task; root faults are latent and only symptoms are observable.
- Correct control requires diagnosis under uncertainty, not direct reactive thresholding.
- Wrong interventions impose physical side effects and increase downstream failure risk.

## What Must Be Demonstrated

- Stable control policy under hidden-fault dynamics, not scripted fault-label lookup.
- Action selection that balances immediate mitigation with long-horizon survival.
- Robustness across easy/medium/hard tasks with score validity in `(0,1)`.
- Reproducible evaluation traces (`[START]/[STEP]/[END]`) under fixed seed.

## Why This Is Non-Trivial

- Partial observability creates state-aliasing: identical telemetry can correspond to different latent faults.
- Action side effects make greedy one-step optimization unsafe.
- Reward is multi-objective and penalizes both inaction under faults and excessive costly interventions.
- Deployment path must preserve strict contracts while serving mutable simulator state over HTTP.
