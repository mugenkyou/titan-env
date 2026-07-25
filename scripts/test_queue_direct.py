"""Direct in-process test of SimulationManager queue logic — no WS, no
uvicorn, no async timing flakiness. Confirms multi-fault recovery works."""
from __future__ import annotations
import os, sys
from pathlib import Path

# Force FAST_MODE so we use the heuristic path (no LLM cold-start).
os.environ["TITAN_FAST_MODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization.backend.server import SimulationManager  # noqa: E402


def tick(sim: SimulationManager) -> dict:
    """One full UI tick equivalent (decide_fast → apply)."""
    belief, action_id, reason, epoch = sim.decide_fast()
    return sim.apply(belief, action_id, reason, epoch) or {}


def main() -> int:
    sim = SimulationManager()
    sim.reset("none")

    print(f"initial: recovered={sim.faults_recovered} queue={sim._active_faults}")

    print("\n--- INJECT THERMAL ---")
    sim.inject_fault("THERMAL")
    print(f"after inject: queue={sim._active_faults} hold={sim._fault_hold_steps}")
    print("--- INJECT MEMORY ---")
    sim.inject_fault("MEMORY")
    print(f"after inject: queue={sim._active_faults} hold={sim._fault_hold_steps}")
    print("--- INJECT SEU ---")
    sim.inject_fault("SEU")
    print(f"after inject: queue={sim._active_faults} hold={sim._fault_hold_steps}")

    print("\n--- TICK LOOP ---")
    for i in range(20):
        out = tick(sim)
        print(f"tick {i:2d}: step={out.get('step'):>3} "
              f"action={out.get('action_name', '?'):<22} "
              f"reason={out.get('reason'):<20} "
              f"recovered={out.get('faults_recovered')} "
              f"queue_len={out.get('active_fault_count')} "
              f"queue={[af['type'] for af in (out.get('active_faults') or [])]}")
        if not sim._active_faults:
            print("  (queue drained)")
            break

    print(f"\nfinal: faults_seen={sim.faults_seen} "
          f"faults_recovered={sim.faults_recovered}")
    expected = 3
    if sim.faults_recovered >= expected:
        print(f"PASS ({sim.faults_recovered} >= {expected})")
        return 0
    print(f"FAIL ({sim.faults_recovered} < {expected})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
