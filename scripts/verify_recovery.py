"""End-to-end check: inject faults via WebSocket and confirm the backend
credits recoveries. Tests both single-fault and multi-fault scenarios.

Run while `python visualization/backend/server.py` is up:
    python scripts/verify_recovery.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import websockets

WS_URL = "ws://localhost:8000/ws"

FAULT_TYPES = ["THERMAL", "MEMORY", "POWER", "LATCH_UP", "SEU"]


async def drain_for(ws, seconds: float) -> list[dict[str, Any]]:
    """Collect all messages that arrive in `seconds` seconds."""
    out: list[dict[str, Any]] = []
    end = asyncio.get_event_loop().time() + seconds
    while True:
        remaining = end - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        try:
            out.append(json.loads(msg))
        except Exception:  # noqa: BLE001
            pass
    return out


def latest_recovered(msgs: list[dict[str, Any]], fallback: int) -> int:
    for m in reversed(msgs):
        if "faults_recovered" in m:
            return int(m["faults_recovered"])
    return fallback


async def reset(ws) -> int:
    await ws.send(json.dumps({"action": "reset", "profile": "none"}))
    msgs = await drain_for(ws, 1.5)
    return latest_recovered(msgs, 0)


async def test_single(ws) -> tuple[int, int]:
    print("\n=== single-fault tests ===")
    passed = 0
    base = await reset(ws)
    await ws.send(json.dumps({"action": "set_speed", "speed": 4.0}))
    await drain_for(ws, 0.5)

    for ftype in FAULT_TYPES:
        # Re-baseline because the env may have reset between iterations.
        snap = await drain_for(ws, 0.6)
        before = latest_recovered(snap, base)

        print(f"\n--- inject {ftype} (recovered_before={before}) ---")
        await ws.send(json.dumps({"action": "inject_fault",
                                   "fault_type": ftype}))
        msgs = await drain_for(ws, 6.0)
        after = latest_recovered(msgs, before)
        delta = after - before
        ok = delta >= 1
        print(f"[{'PASS' if ok else 'FAIL'}] {ftype}: "
              f"{before}->{after} (delta={delta})")
        if ok:
            passed += 1
        base = after
    return passed, len(FAULT_TYPES)


async def test_multi(ws) -> tuple[int, int]:
    """Inject 3 different faults back-to-back and verify all are recovered."""
    print("\n=== multi-fault test (3 simultaneous injections) ===")

    base = await reset(ws)
    await ws.send(json.dumps({"action": "set_speed", "speed": 4.0}))
    await drain_for(ws, 0.5)
    snap = await drain_for(ws, 0.6)
    before = latest_recovered(snap, base)

    print(f"baseline recovered={before}")
    triple = ["THERMAL", "MEMORY", "SEU"]
    for ftype in triple:
        await ws.send(json.dumps({"action": "inject_fault",
                                   "fault_type": ftype}))
        await asyncio.sleep(0.05)
    print(f"injected {triple} back-to-back")

    # Each fault needs ~3-step hold + 1-step heuristic recovery, with 4Hz
    # tick rate that's ~1s per fault. Allow plenty of slack.
    msgs = await drain_for(ws, 12.0)
    after = latest_recovered(msgs, before)

    # Track active-fault queue depth over time so we can see drainage.
    depths = [m.get("active_fault_count") for m in msgs
              if "active_fault_count" in m]
    print(f"queue depth trace (every msg): "
          f"first={depths[:1]} max={max(depths) if depths else 0} "
          f"last={depths[-1:] if depths else 'n/a'}")

    delta = after - before
    expected = len(triple)
    ok = delta >= expected
    print(f"[{'PASS' if ok else 'FAIL'}] multi-injection: "
          f"{before}->{after} (delta={delta}, expected>={expected})")
    return (1 if ok else 0), 1


async def main() -> int:
    print(f"connecting to {WS_URL} ...")
    async with websockets.connect(WS_URL) as ws:
        s_pass, s_total = await test_single(ws)
        m_pass, m_total = await test_multi(ws)

    print(f"\n=== summary: single {s_pass}/{s_total}, multi {m_pass}/{m_total} ===")
    return 0 if (s_pass == s_total and m_pass == m_total) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
