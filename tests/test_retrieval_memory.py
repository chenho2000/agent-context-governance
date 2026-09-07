import pytest
from context_lab.demo import graph_fixture
from context_lab.store import Store
from context_lab.retrieval import Prism, bm25, route
from context_lab.memory import SimpleMemory
from context_lab.providers import ReplayModel
from context_lab.memgpt import MemoryAgent, SleepWorker
from context_lab.types import BudgetError
from context_lab.runtime import Runtime


def test_prism_path_intent_budget_persistence(tmp_path):
    graph = graph_fixture()
    result = graph.retrieve("为什么 Ada 改变支付方案", budget=80)
    assert result["intent"] == "causal"
    assert result["units"] <= 80
    assert any(e["id"] == "e2" for e in result["evidence"])
    assert any(e["relation"] == "causal" and e["step"] == 0.4 for e in result["trace"])
    assert not graph.retrieve("支付", budget=0)["evidence"]
    store = Store(tmp_path)
    graph.save(store)
    assert Prism.load(Store(tmp_path)).retrieve("支付") == graph.retrieve("支付")
    assert bm25("needle", ["needle hay", "hay hay"])[0] > bm25("needle", ["needle hay", "hay hay"])[1]
    assert route("变化")[0] == "evolution"


def test_model_rerank_invalid_ids_rejected():
    graph = graph_fixture()
    with pytest.raises(ValueError, match="IDs"):
        graph.retrieve("为什么", model=ReplayModel([{"ids": ["invented"]}]))


def test_memory_date_synthesis_correction_and_scope(tmp_path):
    store = Store(tmp_path)

    def fact(value, date, correction=False):
        return {
            "entity": "Ada",
            "slot": "plan",
            "value": value,
            "date": date,
            "text": "Ada plan " + value,
            "correction": correction,
        }

    a = store.append(
        "s",
        "first",
        turn=1,
        kind="user",
        metadata={"date": "2026-09-06", "facts": [fact("retry", "yesterday")]},
    )
    b = store.append(
        "s",
        "repeat",
        turn=2,
        kind="user",
        metadata={"date": "2026-09-06", "facts": [fact("retry", "2026-09-05")]},
    )
    memory = SimpleMemory(store, "s")
    ids = memory.ingest([a, b], window=2, stride=1)
    assert len(ids) == 1
    assert store.memories("s")[0]["sources"] == [a.id, b.id]
    c = store.append(
        "s",
        "corrected",
        turn=3,
        kind="user",
        metadata={"date": "2026-09-06", "facts": [fact("idempotency", "2026-09-06", True)]},
    )
    memory.ingest([c])
    result = memory.retrieve("Ada plan", entity="Ada")
    assert len(result["memories"]) == 1
    assert result["memories"][0]["value"] == "idempotency"
    assert len(memory.retrieve("Ada", include_superseded=True)["memories"]) == 2
    assert not memory.retrieve("Ada", date_to="2026-09-04")["memories"]
    with pytest.raises(ValueError, match="foreign"):
        SimpleMemory(store, "other").ingest([a])
    worker = SleepWorker(store)
    worker.start("s").result()
    worker.close()
    assert len(Store(tmp_path).memories("s")) == 2


def test_hallucinated_source_and_ambiguous_date(tmp_path):
    store = Store(tmp_path)
    obj = store.append("s", "Ada", turn=1, kind="user")
    memory = SimpleMemory(store, "s")
    fact = {"text": "Ada", "entity": "Ada", "slot": "x", "value": "y", "sources": ["foreign"]}
    with pytest.raises(ValueError, match="provenance"):
        memory.ingest([obj], model=ReplayModel([{"facts": [fact]}]))
    fact.update(sources=[obj.id], date="yesterday")
    with pytest.raises(ValueError, match="anchor"):
        memory.ingest([obj], model=ReplayModel([{"facts": [fact]}]))


def test_memgpt_paging_heartbeat_and_caps(tmp_path):
    store = Store(tmp_path)
    for n in range(8):
        store.append("s", f"needle {n}", turn=n + 1, kind="user")
    agent = MemoryAgent(store, "s", capacity=60)
    assert agent.tool("recall_search", {"query": "needle"})["has_more"]
    assert len(agent.tool("recall_search", {"query": "needle", "page": 2})["results"]) == 2
    with pytest.raises(BudgetError):
        agent.tool("core_replace", {"name": "x", "text": "many words " * 100})
    agent.capacity = 1000
    result = agent.run(
        ReplayModel([{"tool": "recall_search", "args": {"query": "needle"}, "request_heartbeat": True}]),
        "needle",
        max_steps=1,
    )
    assert result["limit_reached"]
    runtime = Runtime(store, "s", capacity=1)
    with pytest.raises(BudgetError):
        runtime.retrieve_first("needle", graph_fixture())
    runtime.close()


def test_model_directed_archive_and_explicit_cross_session(tmp_path):
    store = Store(tmp_path)
    obj = store.append("old", "Ada uses idempotency", turn=1, kind="user")
    agent = MemoryAgent(store, "old", capacity=3000)
    agent.tool(
        "archival_insert",
        {
            "source_ids": [obj.id],
            "facts": [
                {
                    "text": obj.content,
                    "entity": "Ada",
                    "slot": "plan",
                    "value": "idempotency",
                    "date": "2026-09-06",
                    "sources": [obj.id],
                }
            ],
        },
    )
    fresh = MemoryAgent(Store(tmp_path), "new", capacity=3000, archival_session="old")
    assert fresh.tool("archival_search", {"query": "Ada"})["memories"][0]["sources"] == [obj.id]
    assert fresh.tool("recall_search", {"query": "Ada"})["results"] == []
