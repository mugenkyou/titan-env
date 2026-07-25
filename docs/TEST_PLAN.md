# TEST_PLAN

## Minimal Integration Test Set

- Contract test: `/reset` returns all required observation keys and bounded scalar values.
- Contract test: `/step` response envelope contains `observation`, `reward`, `done`, `info`.
- Action-map test: each canonical command maps to expected discrete id; unknown command maps to `no_action`.
- Hidden-state test: standard observation and sanitized info contain no raw fault keys.
- Reward test: deterministic fixture state produces exact weighted component sum and terminal penalty behavior.
- Episode test: stepping after terminal state raises/returns expected terminal behavior until `reset()`.
- Scoring test: task grader output is clamped to `(0,1)` and never equals boundary values.
- Concurrency test: parallel `/step` requests do not corrupt singleton env state.

## Required Failure Cases

- Invalid JSON body to `/step` returns `400` with `invalid_json`.
- Unknown route returns `404` with `not_found`.
- Empty command payload defaults to safe command path and does not crash.
- High-cost action spam sequence degrades reward and does not produce NaN.
- Missing API credentials triggers fallback inference path without runtime failure.

## Integration Risks This Plan Targets

- Drift between OpenEnv schema and wrapper implementation.
- Drift between action parser vocabulary and action-id mapping.
- Fault information leakage through transport payloads.
- Score contract violations that break evaluator ingestion.
