# DESIGN_CONSTRAINTS

## Invariants

- Step order is fixed: fault injection -> action application -> physics transition -> fault/threshold flag computation -> failure check -> observation emission.
- Interface telemetry fields are bounded to `[0.0, 1.0]`; non-finite values are coerced to `0.0` before emission.
- Unknown/empty commands resolve to `no_action` and must not throw.
- Episode termination is `failure OR max_steps`; post-termination `step()` is invalid until `reset()`.
- Evaluation output is always clamped to `0.01 <= score <= 0.99`.

## Forbidden Changes

- Do not expose raw fault flags/severity in standard interface observation or sanitized info.
- Do not change discrete action ids or remap existing command strings.
- Do not change endpoint names (`/ping`, `/health`, `/state`, `/reset`, `/step`) or payload envelope keys.
- Do not bypass score clamping or return 0/1 edge values.
- Do not remove request-level locking around singleton environment access.

## Cross-Module Consistency Rules

- `action_mapping.py`, `actions.py::ActionType`, and external contract docs must agree on command semantics.
- Wrapper observation keys must remain aligned with `Observation` model and `openenv.yaml` requirements.
- Reward action-cost normalization must consume canonical `ACTION_COSTS` from `actions.py`.
- If a new action is added internally, update command parser vocabulary, command-id mapping, and deployment contract in one change.
- Any change to termination semantics must be reflected simultaneously in task evaluators and runtime tests.
