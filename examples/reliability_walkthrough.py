"""Run: uv run python examples/reliability_walkthrough.py. No API key or model required."""

import json
from pathlib import Path
import tempfile
from context_lab.store import Store
from context_lab.runtime import Runtime
from context_lab.types import Action, Plan, count
from context_lab.projection import prompt, required_view
from context_lab.evidence import pack_evidence
from context_lab.memory import SimpleMemory


def main():
    Path("runs").mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="reliability-", dir="runs")).resolve()
    store = Store(root / "state")
    old = store.append("task", "old result\n" * 100, turn=1, tool="read")
    store.append("task", "current constraint " * 400, turn=2, kind="user")
    with Runtime(store, "task", capacity=10000) as runtime:
        runtime.governor.stage(
            Plan("task", store.snapshot("task").revision, [Action(old.id, "fold")], "example")
        )
        deferred = runtime.poll_gc()
    with Runtime(Store(root / "state"), "task", capacity=10000) as runtime:
        resumed = runtime.poll_gc(expired=True)
    assert deferred["status"] == "pending" and resumed["committed"]
    snap = store.snapshot("task")
    states = required_view(snap, hide_unrelated=True)
    base = count(prompt(snap, states, dynamic="compare"))
    packed = pack_evidence(
        snap,
        states,
        "compare",
        [
            {"id": "large", "text": "large evidence " * 1000, "sources": [old.id]},
            {"id": "small", "text": "Use an idempotency key.", "sources": [old.id]},
        ],
        capacity=base + 170,
        body_budget=10000,
        output_reserve=20,
    )
    assert packed["packing"]["selected_ids"] == ["small"]
    memory = SimpleMemory(store, "facts")

    def fact(value, turn, **extra):
        return store.append(
            "facts",
            value,
            turn=turn,
            kind="user",
            metadata={
                "facts": [
                    {
                        "entity": "Ada",
                        "slot": "plan",
                        "value": value,
                        "date": "2026-09-07",
                        "text": "Ada plan " + value,
                        **extra,
                    }
                ]
            },
        )

    a = fact("A", 1)
    aid = memory.ingest([a])[0]
    b = fact("B", 2)
    memory.ingest([b])
    c = fact("C", 3, supersedes=[aid])
    memory.ingest([c])
    memory.ingest([a, b, c])
    memories = memory.retrieve("Ada", entity="Ada")["memories"]
    assert {m["value"] for m in memories} == {"B", "C"}
    result = {
        "note": "synthetic mechanism checks; not semantic accuracy or real billing",
        "deferred": deferred,
        "resumed": resumed,
        "packing": packed["packing"],
        "surviving_memory_values": sorted(m["value"] for m in memories),
    }
    (root / "trace.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(root / "trace.json"), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
