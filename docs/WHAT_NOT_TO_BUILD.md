# WHAT_NOT_TO_BUILD

## Banned Features

- Do not add direct fault-label observations to standard agent-facing observation.
- Do not add autonomous multi-action macros per step.
- Do not add external persistence layers (DB, cache, queue) for core simulation runtime.
- Do not add web UI dashboards as part of benchmark runtime path.
- Do not add online learning/training loops inside serving endpoints.

## Overengineering Guards

- No plugin framework for actions or reward terms.
- No dynamic schema negotiation; contracts stay static.
- No multi-agent coordination abstractions.
- No probabilistic belief-state subsystem unless explicitly requested by task owners.

## Irrelevant Additions To Reject

- Authentication/authorization middleware beyond deployment platform defaults.
- Generic observability stacks that alter endpoint latency/behavior contracts.
- Feature work not improving partial-observability control fidelity, evaluation integrity, or deployment compliance.
