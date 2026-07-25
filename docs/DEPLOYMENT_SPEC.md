# DEPLOYMENT_SPEC

## Required Endpoints

- `GET /ping` -> `200` `{ "status": "ok" }`
- `GET /health` -> `200` `{ "status": "ok" }`
- `GET /state` -> `200` current typed observation snapshot
- `POST /reset` -> `200` initial observation object
- `POST /step` body `{ "command": "<action>" }` -> `200` step envelope

## Error Behavior

- Invalid JSON on POST -> `400` `{ "error": "invalid_json" }`
- Unknown path -> `404` `{ "error": "not_found" }`

## Port Behavior

- Server binds host `0.0.0.0`.
- Active port = explicit function arg OR env `PORT` OR default `7860`.
- Container/runtime deployments must expose selected port externally.

## Runtime Expectations

- Single in-process environment instance is reused across requests.
- Request handling is lock-guarded for thread safety.
- `POST /reset` reinitializes episode state deterministically for a fixed seed path.
- `POST /step` must be idempotent only with respect to transport retries that are not replayed by client; each accepted step mutates env state exactly once.
- JSON responses are UTF-8 with deterministic key set per endpoint contract.
