import json
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from context_lab.store import Store
from context_lab.types import Plan, Action, GovernanceError, StalePlan
from context_lab.projection import protected, messages, prompt_zones, compaction_request
from context_lab.governance import Governor
from context_lab.planning import GC_PROMPT, tiered_plan
from context_lab.runtime import Runtime
from context_lab.fusion import FusionGraph
from context_lab.summaries import TurnSummaries, consolidate_patterns
from context_lab.providers import ReplayModel
from context_lab.cache import ZonedCache, Prices, forecast_value
from context_lab.preprocess import clean_terminal, preprocess
from context_lab.flywheel import clean_session


def scenario(tmp_path):
    store = Store(tmp_path)
    old = store.append("s", "evidence line\n" * 100, turn=1, tool="read")
    current = store.append("s", "use this evidence", turn=2, kind="user", dependencies=(old.id,))
    return store, old, current


def test_user_fold_only_and_media_instruction_write_guards(tmp_path):
    store = Store(tmp_path)
    user = store.append(
        "s", "old resolved goal", turn=1, kind="user", metadata={"resolved": True, "obsolete": True}
    )
    media = store.append(
        "s", "image attachment metadata", turn=1, tool="screenshot", metadata={"media_type": "image"}
    )
    instruction = store.append(
        "s", "instructions", turn=1, tool="read", arguments={"path": "/repo/AGENTS.md"}
    )
    write = store.append("s", "write succeeded", turn=1, tool="write_file")
    store.append("s", "now", turn=2, kind="user")
    snap = store.snapshot("s")
    assert all(protected(obj, snap) for obj in [media, instruction, write])
    g = Governor(store)
    for obj, action in [(user, "prune"), (media, "fold"), (instruction, "fold"), (write, "fold")]:
        with pytest.raises(GovernanceError):
            g.rehearse(Plan("s", snap.revision, [Action(obj.id, action)], "test"))
    g.close()


def test_mask_preserves_tails_and_aged_error_stack(tmp_path):
    store = Store(tmp_path)
    log = store.append(
        "s", "".join(f"INFO {i}\n" for i in range(100)), turn=1, tool="test", metadata={"redundant_log": True}
    )
    err = store.append(
        "s",
        'Traceback\n  File "a.py", line 1\nError',
        turn=1,
        tool="test",
        arguments={"large": "x" * 100},
        metadata={"error": True},
    )
    store.append("s", "now", turn=4, kind="user")
    g = Governor(store)
    snap, states, _ = g.rehearse(g.plan(store.snapshot("s")))
    assert "INFO 0\n" in states[log.id].display and "INFO 99\n" in states[log.id].display
    assert "INFO 50\n" not in states[log.id].display
    rendered = messages(snap, states)
    assert any(err.content in m.get("content", "") for m in rendered)
    assert "large" not in rendered[2]["tool_calls"][0]["function"]["arguments"]
    g.close()


def test_retrieve_first_keeps_dependency_and_mode_switch(tmp_path):
    store, old, _ = scenario(tmp_path)
    graph = FusionGraph().build(store.snapshot("s"))
    runtime = Runtime(store, "s", capacity=5000)
    result = runtime.context_for("evidence", graph, budget=100)
    assert result["mode"] == "retrieve_first"
    assert old.id in result["prompt"] and "evidence line" in result["prompt"]
    assert runtime.context_for("evidence", available=False)["mode"] == "in_place"
    assert len([e for e in store.events("s") if e["kind"] == "mode_switch"]) == 2
    runtime.close()


def test_nonblocking_start_poll_and_stale(tmp_path):
    store, _, _ = scenario(tmp_path)
    started, release = threading.Event(), threading.Event()

    class Model:
        def generate(self, *args):
            started.set()
            assert release.wait(3)
            return {"actions": []}

    runtime = Runtime(store, "s")
    try:
        assert runtime.start_gc(event="idle", model=Model())["status"] == "started"
        assert started.wait(1)
        assert runtime.poll_gc()["status"] == "planning"
        store.append("s", "new constraint", turn=3, kind="user")
        release.set()
        runtime.pending_future.result(timeout=2)
        assert runtime.poll_gc()["status"] == "stale"
    finally:
        release.set()
        runtime.close()


def test_model_trigger_and_tiered_planner(tmp_path):
    store, old, _ = scenario(tmp_path)

    class Encoder:
        def classify(self, text, labels):
            return {"label": "redundant background", "score": 0.98}

    plan = tiered_plan(store.snapshot("s"), classifier=Encoder())
    assert plan.actions[0].action == "fold" and plan.trace[0]["tier"] == "encoder"
    g = Governor(store)
    g.rehearse(plan)
    g.close()
    runtime = Runtime(store, "s")
    assert runtime.compress_context(start_turn=1, end_turn=1, expired=True)["committed"]
    assert store.recover("s", old.id) == old.content
    runtime.close()
    assert all(f"{i}." in GC_PROMPT for i in range(1, 7))


def test_tiered_local_and_llm_abstention(tmp_path):
    store = Store(tmp_path)
    old = store.append("s", "background " * 200, turn=1, tool="read")
    store.append("s", "new", turn=2, kind="user")
    snap = store.snapshot("s")
    assert not tiered_plan(snap).actions
    plan = tiered_plan(snap, local_model=ReplayModel([{"summary": "brief", "confident": True}]))
    assert plan.actions[0].action == "distill"
    with pytest.raises(ValueError, match="outside"):
        tiered_plan(snap, planner_model=ReplayModel([{"actions": [{"target": "foreign", "action": "fold"}]}]))
    plan = tiered_plan(snap, planner_model=ReplayModel([{"actions": [{"target": old.id, "action": "fold"}]}]))
    assert plan.trace[0]["tier"] == "planner"


def test_fusion_edges_persistence_tuning_and_gc(tmp_path):
    store, old, current = scenario(tmp_path)
    graph = FusionGraph().build(store.snapshot("s"))
    assert any(r == "contains" for edges in graph.edges.values() for _, r, _ in edges)
    assert old.id in graph.live_objects({current.id})
    graph.save(store)
    loaded = FusionGraph.load(store)
    loaded.validate_snapshot(store.snapshot("s"))
    episode = "s:episode:1"
    loaded.tune(inactive=[episode])
    assert episode not in loaded.retrieve("evidence", budget=5000)["candidate_ids"]
    store.append("s", "changed", turn=3, kind="user")
    with pytest.raises(StalePlan):
        loaded.validate_snapshot(store.snapshot("s"))


def test_fusion_model_edges_are_scoped(tmp_path):
    store = Store(tmp_path)
    old = store.append("s", "cause", turn=1, tool="read", metadata={"obsolete": True})
    current = store.append("s", "effect", turn=2, kind="user")
    model = ReplayModel([{"edges": [{"source": current.id, "target": old.id, "relation": "causal"}]}])
    graph = FusionGraph().build(store.snapshot("s"), model=model)
    g = Governor(store, graph)
    with pytest.raises(GovernanceError, match="live dependency"):
        g.rehearse(Plan("s", store.snapshot("s").revision, [Action(old.id, "prune")], "test"))
    g.close()
    with pytest.raises(ValueError, match="model edge"):
        FusionGraph().build(
            store.snapshot("s"),
            model=ReplayModel([{"edges": [{"source": "foreign", "target": old.id, "relation": "causal"}]}]),
        )


def test_structured_summary_exact_spine_and_sources(tmp_path):
    store = Store(tmp_path)
    obj = store.append("s", "不得重复扣款", turn=1, kind="user", metadata={"constraint": True})
    summaries = TurnSummaries(store)
    reply = {field: [] for field in summaries.fields}
    result = summaries.write("s", 1, ReplayModel([reply]))
    assert result["exact_constraints"][0]["text"] == obj.content
    reply["goals"] = [{"text": "fake", "sources": ["foreign"]}]
    with pytest.raises(ValueError, match="source"):
        summaries.write("s", 1, ReplayModel([reply]))
    assert len(store.blocks("summaries:s")) == 1


def test_slow_clock_and_golden_trace(tmp_path):
    store = Store(tmp_path / "db")
    for session in ["a", "b"]:
        store.append(session, "goal", turn=1, kind="user")
        store.log(session, "failure", {"category": "stale-file"})
    patterns = consolidate_patterns(store, ["a", "b"])
    assert patterns["patterns"][0]["support"] == 2
    retry = store.append("a", "dead retry", turn=1, tool="read", metadata={"retry": True})
    side = store.append("a", "audit write", turn=1, tool="write", metadata={"retry": True})
    result = clean_session(
        store, "a", verified_result="fixed", required_ids=[], directory=tmp_path / "export"
    )
    assert retry.id in result["removed"] and side.id in result["kept"]
    assert json.loads((tmp_path / "export" / "benchmark.json").read_text())["result"] == "fixed"


def test_zoned_cache_expiry_dynamic_and_forecast(tmp_path):
    store, _, _ = scenario(tmp_path)
    snap = store.snapshot("s")
    cache = ZonedCache()
    first = cache.request(prompt_zones(snap, dynamic="A"), now=0)
    second = cache.request(prompt_zones(snap, dynamic="B"), now=1)
    expired = cache.request(prompt_zones(snap, dynamic="C"), now=400)
    assert first["read"] == 0 and second["read"] > expired["read"] > 0
    assert second["uncached"] > 0
    assert forecast_value(5000, 4500, hit_probability=1) < 0
    assert forecast_value(5000, 4500, hit_probability=0) > 0
    assert forecast_value(5000, 4500, prices=Prices(read=0.5), future_calls=20) > 0
    request = compaction_request(snap, "summarize")
    original = compaction_request(snap, "different instruction")
    assert request["messages"][:-1] == original["messages"][:-1]


def test_atomic_memory_source_merge(tmp_path):
    store = Store(tmp_path)

    def write(i):
        store.put_memory({"id": "one", "session": "s", "sources": [str(i)], "supersedes": []})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(12)))
    assert len(store.memories("s")[0]["sources"]) == 12


def test_terminal_clean_and_tracker_reuses_folded_source(tmp_path):
    assert clean_terminal("⠋ loading\rOK   \n\n\n") == "OK\n\n"
    store, old, _ = scenario(tmp_path)
    # Use a separate versioned source for this scenario.
    source = store.append(
        "s",
        "code\n" * 100,
        turn=2,
        tool="read",
        metadata={"path": "a.py", "version": "v1", "range": [1, 100]},
    )
    store.append("s", "next", turn=3, kind="user")
    preprocess(store, "s", limit=50)
    other = store.append("s", source.content, turn=3, tool="read", metadata=source.metadata)
    store.append("s", "next", turn=4, kind="user")
    preprocess(store, "s", limit=10000)
    assert store.snapshot("s").states[other.id].source_id == source.id


def test_information_gate_conservative(tmp_path):
    from context_lab.gating import information_gate

    store = Store(tmp_path)
    greeting = store.append("s", "thanks!", turn=1, kind="user")
    assert not information_gate([greeting])["keep"]
    fact = store.append("s", "The error occurred at 12:34.", turn=2, kind="user")
    assert information_gate([fact])["keep"]
    constrained = store.append("s", "thanks", turn=3, kind="user", metadata={"constraint": True})
    assert information_gate([constrained])["keep"]


def test_encoder_adapter_contract(monkeypatch):
    import sys
    from types import SimpleNamespace
    from context_lab.providers import EncoderClassifier

    def factory(task, **kwargs):
        assert task == "zero-shot-classification"
        return lambda text, candidate_labels, multi_label: {"labels": candidate_labels, "scores": [0.9, 0.1]}

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=factory))
    assert EncoderClassifier("test").classify("x", ["a", "b"]) == {"label": "a", "score": 0.9}


def test_symbol_search(tmp_path):
    from context_lab.codemap import CodeMap

    root = tmp_path / "code"
    root.mkdir()
    (root / "a.py").write_text('def f():\n    """idempotent payment implementation"""\n    pass\n')
    index = CodeMap(Store(tmp_path / "db"), root)
    index.build()
    assert index.search("idempotent")[0]["name"] == "f"


def test_methodology_end_to_end(tmp_path):
    from context_lab.methodology import run

    report = run(tmp_path)
    trace = json.loads((report / "trace.json").read_text())
    assert trace["commit"]["committed"]
    assert trace["fallback"]["mode"] == "in_place"
    assert trace["summary"]["exact_constraints"]


def test_new_dependency_revives_hidden_evidence(tmp_path):
    from context_lab.types import ViewState

    store = Store(tmp_path)
    old = store.append("s", "CRITICAL OLD EVIDENCE", turn=1, tool="read", metadata={"obsolete": True})
    store.append("s", "next", turn=2, kind="user")
    store.commit(store.snapshot("s"), {old.id: ViewState("hidden")}, event={"test": "prior prune"})
    store.append("s", "now use the old evidence", turn=3, kind="user", dependencies=(old.id,))
    runtime = Runtime(store, "s")
    assert old.content in runtime.context_for("compare", available=False)["prompt"]
    runtime.close()


def test_model_cannot_choose_cache_state(tmp_path):
    from context_lab.memgpt import MemoryAgent

    store, _, _ = scenario(tmp_path)
    runtime = Runtime(store, "s")
    agent = MemoryAgent(store, "s", runtime=runtime)
    with pytest.raises(ValueError, match="cache state"):
        agent.tool("compress_context", {"start_turn": 1, "end_turn": 1, "expired": True})
    runtime.close()


def test_coverage_references():
    from context_lab.coverage import audit

    assert not audit(check=True)["failures"]


def test_optional_embedding_and_graph_encoder_contract(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from context_lab.retrieval import SentenceEmbedding

    class FakeModel:
        def __init__(self, name):
            pass

        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings
            return SimpleNamespace(tolist=lambda: [[1.0, 0.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel))
    assert SentenceEmbedding("test").encode(["q"]) == [[1.0, 0.0]]
    store = Store(tmp_path)
    store.append("s", "old", turn=1, tool="read")
    store.append("s", "now", turn=2, kind="user")

    class Encoder:
        def classify(self, text, labels):
            return {"label": "semantic", "score": 0.99}

    graph = FusionGraph().build(store.snapshot("s"), classifier=Encoder())
    assert graph.build_trace[0]["tier"] == "encoder"


def test_model_memory_query_plan_and_compaction_transport(tmp_path):
    from context_lab.memory import SimpleMemory

    store = Store(tmp_path)
    obj = store.append(
        "s",
        "Ada changed plan",
        turn=1,
        kind="user",
        metadata={"facts": [{"entity": "Ada", "slot": "plan", "value": "new", "text": "Ada changed plan"}]},
    )
    memory = SimpleMemory(store, "s")
    memory.ingest([obj])
    result = memory.retrieve(
        "why Ada", model=ReplayModel([{"queries": ["Ada plan", "Ada changed"], "top_k": 1}])
    )
    assert len(result["plan"]["queries"]) == 2 and len(result["memories"]) == 1


def test_retrieval_rejects_intervening_append(tmp_path):
    store, _, _ = scenario(tmp_path)

    class RacingRetriever:
        def retrieve(self, query, budget):
            store.append("s", "new constraint while retrieving", turn=3, kind="user")
            return {"evidence": []}

    with Runtime(store, "s") as runtime:
        with pytest.raises(StalePlan, match="during retrieval"):
            runtime.retrieve_first("query", RacingRetriever())


def test_corrupt_alias_cycle_fails_cleanly(tmp_path):
    from context_lab.types import canonical

    store, old, _ = scenario(tmp_path)
    with store.connect() as db:
        db.execute(
            "INSERT INTO views VALUES(?,?)", (old.id, canonical({"mode": "alias", "source_id": old.id}))
        )
    with pytest.raises(ValueError, match="cycle"):
        store.recover("s", old.id)


def test_cache_rejects_nonfinite_inputs():
    from context_lab.cache import commit_value, PrefixCache

    with pytest.raises(ValueError):
        commit_value(float("nan"), 10)
    with pytest.raises(ValueError):
        PrefixCache(ttl=-1)


def test_tracker_does_not_alias_object_to_itself(tmp_path):
    from context_lab.preprocess import FileStateTracker

    store = Store(tmp_path)
    obj = store.append(
        "s", "text", turn=1, tool="read", metadata={"path": "a", "version": "1", "range": [1, 1]}
    )
    tracker = FileStateTracker()
    snap = store.snapshot("s")
    assert tracker.observe(obj, snap) is None
    assert tracker.observe(obj, snap) is None


def test_ollama_tool_protocol_and_stable_compaction_prefix(tmp_path):
    from context_lab.providers import ollama_messages

    store, _, _ = scenario(tmp_path)
    request = compaction_request(store.snapshot("s"), "compact")
    translated = ollama_messages(request["messages"])
    assert translated[:-1] == ollama_messages(request["messages"][:-1])
    calls = [m for m in translated if m.get("tool_calls")]
    results = [m for m in translated if m["role"] == "tool"]
    assert isinstance(calls[0]["tool_calls"][0]["function"]["arguments"], dict)
    assert results[0]["tool_name"] == "read"
    assert any("tool_call_id" in m for m in request["messages"])
