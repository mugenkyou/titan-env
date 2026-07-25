# SYSTEM_OVERVIEW

## Purpose

TITAN is a deterministic benchmark for autonomous satellite fault recovery under partial observability.
Agents control recovery actions from normalized telemetry while latent fault causes evolve internally.
System intent is robust diagnosis-informed control, not direct fault-label classification.

## Architecture Boundaries

## Simulation core

Ownership: true plant state, fault injection, action effects, transition order, termination.
Inputs: discrete action id.
Outputs: raw observation dict, done flag, internal info.

## Interface layer

Ownership: command-to-id mapping, observation projection, fault sanitization, OpenEnv contract.
Inputs: command string.
Outputs: typed observation/reward, safe info payload.

## Evaluation layer

Ownership: trajectory recording, task-specific grading, score clamping to strict open interval.
Inputs: trajectory records.
Outputs: scalar score in (0, 1).

## Serving layer

Ownership: HTTP transport, singleton env lifecycle, thread-safe request handling, JSON schemas.
Inputs: REST requests.
Outputs: contract-compliant JSON responses.

## Components That Must Not Be Modified

- `openenv.yaml` action and observation schemas.
- `titan_env/interface/action_mapping.py` canonical command-id map.
- `titan_env/interface/openenv_wrapper.py` fault sanitization behavior and typed response shape.
- `titan_env/evaluation/scoring.py` strict `(0, 1)` clamping rule.
- `titan_env/server/app.py` endpoint paths and response envelope keys.
