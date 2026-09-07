import sqlite3
import pytest
from context_lab.store import Store
from context_lab.types import Action, Plan, GovernanceError, StalePlan, BudgetError
from context_lab.governance import Governor, parse_xml
from context_lab.projection import messages, prompt, dependency_closure
from context_lab.preprocess import preprocess
from context_lab.runtime import Runtime


@pytest.fixture
def setup(tmp_path):
    store = Store(tmp_path)
    old = store.append(
        "s",
        "original code\n" * 200,
        turn=1,
        tool="read",
        metadata={"path": "a.py", "version": "v1", "range": [1, 200]},
    )
    current = store.append("s", "Preserve exact cents", turn=2, kind="user")
    return store, old, current


def test_fold_restart_recovery_protocol_and_raw_immutable(setup):
    store, old, _ = setup
    governor = Governor(store)
    snap = store.snapshot("s")
    plan = parse_xml(f'<gc><fold target="{old.id}" /></gc>', snap)
    governor.stage(plan)
    assert governor.commit("s", expired=True)["committed"]
    restarted = Store(store.root)
    assert restarted.recover("s", old.id) == old.content
    msg = messages(restarted.snapshot("s"))
    assert msg[0]["tool_calls"][0]["id"] == msg[1]["tool_call_id"]
    assert "context_ref" in msg[0]["tool_calls"][0]["function"]["arguments"]
    with store.connect() as db, pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM objects WHERE id=?", (old.id,))
    store.sidecar_path(old).write_text("corrupt")
    with pytest.raises(ValueError, match="integrity"):
        restarted.recover("s", old.id)
    governor.close()


def test_protected_foreign_duplicate_stale_and_busy(setup):
    store, old, current = setup
    g = Governor(store)

    def plan(actions):
        return Plan("s", store.snapshot("s").revision, actions, "test")

    for actions in [
        [Action(current.id, "fold")],
        [Action("foreign:tool:1", "fold")],
        [Action(old.id, "fold"), Action(old.id, "fold")],
    ]:
        with pytest.raises(GovernanceError):
            g.rehearse(plan(actions))
    stale = g.fork("s").result()
    store.append("s", "new", turn=3, kind="user")
    with pytest.raises(StalePlan):
        g.stage(stale)
    store.set_busy("s", 1)
    g.stage(plan([Action(old.id, "fold")]))
    with pytest.raises(ValueError, match="running"):
        g.commit("s", expired=True)
    assert store.snapshot("s").states[old.id].mode == "active"
    g.close()


def test_dedupe_version_range_and_alias_liveness(setup):
    store, old, _ = setup
    same = store.append("s", old.content, turn=2, tool="read", metadata=old.metadata)
    changed = store.append("s", old.content, turn=2, tool="read", metadata={**old.metadata, "version": "v2"})
    partial = store.append("s", old.content, turn=2, tool="read", metadata={**old.metadata, "range": [5, 8]})
    store.append("s", "next", turn=3, kind="user")
    preprocess(store, "s", limit=10000)
    snap = store.snapshot("s")
    assert snap.states[same.id].mode == "alias"
    assert snap.states[changed.id].mode == "active"
    assert snap.states[partial.id].mode == "active"
    assert old.id in dependency_closure(snap)
    g = Governor(store)
    g.stage(Plan("s", snap.revision, [Action(old.id, "fold")], "test"))
    g.commit("s", expired=True)
    assert store.recover("s", same.id) == old.content
    g.close()


def test_sparse_and_dependency_protection(tmp_path):
    store = Store(tmp_path)
    evidence = store.append(
        "s",
        "@@ -1 +1 @@\n-old\n+new",
        turn=1,
        tool="diff",
        metadata={"redundant_log": True, "obsolete": True},
    )
    store.append("s", "now", turn=2, kind="user", dependencies=(evidence.id,))
    g = Governor(store)
    for action in ["prune", "mask", "distill"]:
        with pytest.raises(GovernanceError):
            g.rehearse(
                Plan(
                    "s", store.snapshot("s").revision, [Action(evidence.id, action, summary="short")], "test"
                )
            )
    g.close()


def test_dcp_purge_preserves_error_and_distill(tmp_path):
    store = Store(tmp_path)
    error = store.append(
        "s",
        "Validation failed: invalid ID",
        turn=1,
        tool="api",
        arguments={"huge": "x" * 200},
        metadata={"purge_input": True},
    )
    text = store.append("s", "long background " * 100, turn=1, tool="read")
    store.append("s", "now", turn=2, kind="user")
    g = Governor(store)
    plan = Plan(
        "s",
        store.snapshot("s").revision,
        [Action(error.id, "mask"), Action(text.id, "distill", summary="Background summary")],
        "test",
    )
    snap, states, _ = g.rehearse(plan)
    rendered = messages(snap, states)
    assert "huge" not in rendered[0]["tool_calls"][0]["function"]["arguments"]
    assert error.content in rendered[1]["content"]
    assert store.recover("s", text.id) == text.content
    g.close()


def test_whole_turn_hard_budget_and_restore(setup):
    store, old, _ = setup
    g = Governor(store)
    g.stage(Plan("s", store.snapshot("s").revision, [Action("turn:1", "fold")], "test"))
    with pytest.raises(BudgetError):
        g.commit("s", hard_limit=1)
    assert g.commit("s", expired=True)["committed"]
    runtime = Runtime(store, "s", capacity=100000)
    assert runtime.recover_into_view(old.id) == old.content
    assert old.content.replace("\n", "\\n") in prompt(store.snapshot("s"))
    runtime.close()
    g.close()


def test_alias_cycle_and_corrupt_fold_rejected_at_commit(setup):
    from context_lab.types import ViewState

    store, old, current = setup
    snap = store.snapshot("s")
    with pytest.raises(ValueError, match="identical earlier"):
        store.commit(snap, {old.id: ViewState("alias", source_id=old.id)}, event={})
    with pytest.raises(ValueError, match="identical earlier"):
        store.commit(snap, {current.id: ViewState("alias", source_id=old.id)}, event={})
    store.persist_sidecar(old)
    store.sidecar_path(old).write_text("wrong")
    with pytest.raises(ValueError, match="integrity"):
        store.commit(snap, {old.id: ViewState("folded")}, event={})
