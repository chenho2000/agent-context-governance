import json
import pytest
from context_lab.store import Store
from context_lab.types import Action, Plan, StalePlan, count
from context_lab.governance import Governor
from context_lab.runtime import Runtime
from context_lab.projection import prompt, required_view
from context_lab.memory import SimpleMemory
from context_lab.providers import ReplayModel
from context_lab.evidence import pack_evidence


def setup(store):
    old = store.append("s", "old data\n" * 100, turn=1, tool="read")
    store.append("s", "current goal", turn=2, kind="user")
    return old


def test_full_request_budget_skips_large_and_preserves_sources(tmp_path):
    store = Store(tmp_path)
    setup(store)
    snap = store.snapshot("s")
    states = required_view(snap, hide_unrelated=True)
    base = count(prompt(snap, states, dynamic="q"))
    rows = [
        {"id": "large", "text": "big word " * 500, "sources": ["original-large"]},
        {"id": "small", "text": "fact", "sources": ["original-small"]},
    ]
    result = pack_evidence(snap, states, "q", rows, capacity=base + 160, body_budget=10000, output_reserve=20)
    assert result["units"] + 20 <= base + 160
    assert result["packing"]["selected_ids"] == ["small"]
    assert result["packing"]["omitted"][0]["reason"] == "serialized_request_budget"
    assert "original-small" in result["prompt"]
    assert result["packing"]["body_units"] == count("fact")


def test_conflicting_duplicate_evidence_rejected(tmp_path):
    store = Store(tmp_path)
    setup(store)
    snap = store.snapshot("s")
    with pytest.raises(ValueError, match="conflicting"):
        pack_evidence(
            snap,
            snap.states,
            "q",
            [{"id": "same", "text": "A"}, {"id": "same", "text": "B"}],
            capacity=5000,
            body_budget=5000,
        )


def test_pending_retried_after_restart_without_replanning(tmp_path):
    store = Store(tmp_path)
    old = setup(store)
    # Keep a long unchanged protected suffix: one valid-cache call does not repay its rewrite.
    store.append("s", "current constraints " * 400, turn=2, kind="user")
    with Runtime(store, "s", capacity=10000) as runtime:
        plan = Plan("s", store.snapshot("s").revision, [Action(old.id, "fold")], "test")
        runtime.governor.stage(plan)
        assert runtime.poll_gc()["status"] == "pending"
    with Runtime(Store(tmp_path), "s", capacity=10000) as restarted:
        assert restarted.start_gc()["status"] == "pending"
        result = restarted.poll_gc(expired=True)
        assert result["status"] == "committed"
        assert restarted.store.pending("s") is None


def test_stale_pending_discarded_and_newer_proposal_preserved(tmp_path):
    store = Store(tmp_path)
    old = setup(store)
    first = Plan("s", store.snapshot("s").revision, [Action(old.id, "fold")], "old")
    store.save_pending(first)
    second = Plan("s", first.revision, [Action(old.id, "keep")], "new")
    store.save_pending(second)
    assert not store.discard_pending(first, reason="late failure")
    assert store.pending("s").planner == "new"
    store.append("s", "new requirement", turn=3, kind="user")
    with Runtime(store, "s") as runtime:
        assert runtime.resume_pending()["status"] == "stale"
    assert store.pending("s") is None


def test_commit_checks_identity_not_only_source_revision(tmp_path, monkeypatch):
    store = Store(tmp_path)
    old = setup(store)
    g = Governor(store)
    first = Plan("s", store.snapshot("s").revision, [Action(old.id, "fold")], "old")
    g.stage(first)
    second = Plan("s", first.revision, [Action(old.id, "keep")], "replacement")
    original = store.commit

    def replace_before_commit(*args, **kwargs):
        store.save_pending(second)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "commit", replace_before_commit)
    try:
        with pytest.raises(StalePlan, match="replaced"):
            g.commit("s", expired=True)
        assert store.snapshot("s").states[old.id].mode == "active"
        assert store.pending("s").planner == "replacement"
    finally:
        g.close()


def fact(value, date="2026-09-07", **extra):
    return {
        "entity": "Ada",
        "slot": "plan",
        "value": value,
        "date": date,
        "text": "Ada plan " + value,
        **extra,
    }


def append_fact(store, value, turn, **extra):
    return store.append("s", value, turn=turn, kind="user", metadata={"facts": [fact(value, **extra)]})


def test_explicit_correction_preserves_unrelated_versions_and_replays(tmp_path):
    store = Store(tmp_path)
    memory = SimpleMemory(store, "s")
    a = append_fact(store, "A", 1)
    aid = memory.ingest([a])[0]
    b = append_fact(store, "B", 2)
    bid = memory.ingest([b])[0]
    c = append_fact(store, "C", 3, supersedes=[aid])
    cid = memory.ingest([c])[0]
    memory.ingest([a, b, c])
    values = {m["value"] for m in memory.retrieve("Ada", entity="Ada")["memories"]}
    assert values == {"B", "C"}
    assert next(m for m in store.memories("s") if m["id"] == cid)["supersedes"] == [aid]
    assert bid not in next(m for m in store.memories("s") if m["id"] == cid)["supersedes"]


def test_window_validation_is_atomic_and_cycle_rolls_back(tmp_path):
    store = Store(tmp_path)
    memory = SimpleMemory(store, "s")
    obj = store.append("s", "source", turn=1, kind="user")
    with pytest.raises(ValueError, match="provenance"):
        memory.ingest(
            [obj],
            model=ReplayModel([{"facts": [fact("A", sources=[obj.id]), fact("B", sources=["foreign"])]}]),
        )
    assert store.memories("s") == []
    a = append_fact(store, "A", 2)
    aid = memory.ingest([a])[0]
    b = append_fact(store, "B", 3, supersedes=[aid])
    bid = memory.ingest([b])[0]
    revision_before = json.dumps(store.memories("s"), sort_keys=True)
    reverse = append_fact(store, "A", 4, supersedes=[bid])
    with pytest.raises(ValueError, match="cycle"):
        memory.ingest([reverse])
    assert json.dumps(store.memories("s"), sort_keys=True) == revision_before


def test_no_match_is_not_reciprocal_rank_evidence_and_vectors_batched(tmp_path):
    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, texts):
            self.calls.append(texts)
            return [[0.0, 0.0] for _ in texts]

    store = Store(tmp_path)
    obj = append_fact(store, "A", 1)
    encoder = Encoder()
    memory = SimpleMemory(store, "s", encoder)
    memory.ingest([obj])
    result = memory.retrieve(
        "why nowhere", model=ReplayModel([{"queries": ["nowhere", "absent"], "top_k": 3}])
    )
    assert result["memories"] == []
    assert len(encoder.calls) == 1
