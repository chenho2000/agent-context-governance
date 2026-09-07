"""Repeatable end-to-end experiment; all figures come from the current run."""

from pathlib import Path
import json
import tempfile
from .store import Store
from .types import Plan, Action, count, canonical
from .projection import prompt, messages
from .preprocess import preprocess
from .governance import Governor
from .cache import PrefixCache, commit_value
from .retrieval import Prism
from .memory import SimpleMemory
from .memgpt import MemoryAgent, SleepWorker
from .providers import ReplayModel
from .codemap import CodeMap
from .flywheel import evaluate_trial, export
from .compression import cascade
from .kv import demo as kv_demo
from .runtime import Runtime


def graph_fixture():
    graph = Prism()
    for key, layer, text in [
        ("ada", "entity", "Ada payment owner"),
        ("payment", "facet_point", "Ada payment plan"),
        ("retry", "facet", "Risk of duplicate charges from payment retries"),
        ("e1", "episode", "On September 5, Ada changed payment retries to use idempotency keys."),
        (
            "e2",
            "episode",
            "Reason: tests showed that payment retries without idempotency keys cause duplicate charges.",
        ),
        ("e3", "episode", "On September 6, Ada discussed the database connection pool size."),
    ]:
        graph.node(key, layer, text)
    graph.edge("ada", "payment", "belongs_to")
    graph.edge("payment", "retry", "belongs_to")
    graph.edge("retry", "e1", "belongs_to")
    graph.edge("e1", "e2", "causal")
    graph.edge("e1", "e3", "temporal")
    graph.edge("e2", "e1", "evolution")
    return graph


def run(parent="runs"):
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="demo-", dir=parent))
    store, session = Store(output / "state"), "payment"
    constraint = store.append(
        session,
        "Never charge twice; amounts must be exact to the cent.",
        turn=1,
        kind="user",
        metadata={"constraint": True},
    )
    code = "def charge(amount):\n    return gateway.charge(amount)\n" + "# legacy implementation\n" * 120
    first = store.append(
        session,
        code,
        turn=1,
        tool="read_file",
        metadata={"path": "payment.py", "version": "v1", "range": [1, 122]},
    )
    duplicate = store.append(
        session,
        code,
        turn=2,
        tool="read_file",
        metadata={"path": "payment.py", "version": "v1", "range": [1, 122]},
    )
    log = store.append(session, "INFO retry\n" * 120, turn=2, tool="test", metadata={"redundant_log": True})
    obsolete = store.append(
        session,
        "Network instability is the only cause. Rejected.",
        turn=2,
        tool="search",
        metadata={"obsolete": True},
    )
    sparse = store.append(
        session,
        'Traceback\n  File "payment.py", line 2\nDuplicateCharge',
        turn=2,
        tool="test",
        metadata={"exact_now": True},
    )
    store.append(
        session,
        "Fix this with idempotency keys and review the old code later.",
        turn=3,
        kind="user",
        dependencies=(first.id,),
    )
    raw = prompt(store.snapshot(session))
    pretrace = preprocess(store, session, limit=10000)
    before = prompt(store.snapshot(session))
    governor = Governor(store)
    plan = Plan(
        session,
        store.snapshot(session).revision,
        [Action(first.id, "fold"), Action(log.id, "mask"), Action(obsolete.id, "prune")],
        "teaching-fixture",
    )
    rehearsal = governor.stage(plan)
    deferred = governor.commit(session, future_calls=1)
    committed = (
        governor.commit(session, future_calls=1, expired=True) if not deferred["committed"] else deferred
    )
    after = prompt(store.snapshot(session))
    recovered = store.recover(session, duplicate.id)
    assertions = {
        "alias_recovery_exact": recovered == code,
        "constraint_preserved": constraint.content in after,
        "sparse_preserved": any(
            sparse.content in m.get("content", "") for m in messages(store.snapshot(session))
        ),
    }
    failures = {}
    for name, action in [
        ("protected", Action(constraint.id, "prune")),
        ("sparse_mask", Action(sparse.id, "mask")),
        ("foreign_id", Action("other:tool:1", "fold")),
    ]:
        try:
            governor.rehearse(Plan(session, store.snapshot(session).revision, [action], "negative-test"))
        except ValueError as error:
            failures[name] = str(error)
    stale = governor.fork(session).result()
    store.append(session, "Continue checking the test results.", turn=4, kind="user")
    try:
        governor.stage(stale)
    except ValueError as error:
        failures["stale_plan"] = str(error)
    governor.close()
    cache = PrefixCache()
    cache_trace = [
        cache.request(text, now=t) for t, text in [(0, before), (1, before), (2, after), (400, after)]
    ]
    graph = graph_fixture()
    graph.save(store)
    query = "Why did Ada change the payment plan?"
    retrieval = {
        label: graph.retrieve(query, budget=80, **flags)
        for label, flags in [
            ("all", {}),
            ("without_N1", {"n1": False}),
            ("without_N2", {"n2": False}),
            ("without_N3", {"n3": False}),
            ("without_N4", {"n4": False}),
        ]
    }
    rows = []
    for i, fact in enumerate(
        [
            {
                "entity": "Ada",
                "slot": "plan",
                "value": "idempotency",
                "date": "yesterday",
                "text": "Ada changed the payment plan to use idempotency keys on 2026-09-05.",
            },
            {
                "entity": "Ada",
                "slot": "reason",
                "value": "duplicate_charge",
                "date": "2026-09-06",
                "text": "Ada changed the payment plan because tests found duplicate charges.",
            },
        ]
    ):
        rows.append(
            store.append(
                "memory",
                fact["text"],
                turn=i + 1,
                kind="user",
                metadata={"date": "2026-09-06", "facts": [fact], "resolved": True},
            )
        )
    memory = SimpleMemory(store, "memory")
    memory_ids = memory.ingest(rows)
    memory_result = memory.retrieve(query, entity="Ada", budget=100)
    agent = MemoryAgent(store, "memory", capacity=3000)
    agent_result = agent.run(
        ReplayModel(
            [
                {
                    "tool": "core_replace",
                    "args": {"name": "task", "text": "Check the payment plan"},
                    "request_heartbeat": True,
                },
                {"tool": "archival_search", "args": {"query": query}, "request_heartbeat": True},
                {"answer": "Replay example: Ada adopted idempotency keys to avoid duplicate charges."},
            ]
        ),
        query,
    )
    worker = SleepWorker(store)
    sleep_ids = worker.start("memory").result()
    worker.close()
    sample = output / "sample_code"
    sample.mkdir()
    (sample / "payment.py").write_text(
        "def charge(amount, key):\n    return gateway.charge(amount, key=key)\n", encoding="utf-8"
    )
    cmap = CodeMap(store, sample)
    code_index = cmap.build()
    symbol = cmap.get_symbol("payment.py", "charge")
    prefix = {"visible_ids": [constraint.id, first.id], "query": "Fix payment"}
    future = {"required_ids": [constraint.id, first.id]}
    trials = [
        evaluate_trial(prefix, "fold", future, [constraint.id], [first.id]),
        evaluate_trial(prefix, "prune", future, [constraint.id]),
    ]
    flywheel = export(trials, output / "training_candidates")
    runtime = Runtime(store, session, capacity=4000)
    retrieve_first = runtime.retrieve_first(query, graph, budget=80)
    runtime.close()
    trace = {
        "unit_note": "lexical teaching units, not provider tokens; synthetic fixture, not paper benchmark",
        "sizes": {"raw": count(raw), "preprocessed": count(before), "governed": count(after)},
        "preprocess": pretrace,
        "rehearsal": rehearsal,
        "cache_valid_decision": deferred,
        "commit": committed,
        "assertions": assertions,
        "negative_cases": failures,
        "cache_requests": cache_trace,
        "cache_examples": {
            "valid": commit_value(5000, 4500),
            "expired": commit_value(5000, 4500, expired=True),
        },
        "prism_ablations": retrieval,
        "simplemem": memory_result,
        "memory_ids": memory_ids,
        "memgpt": agent_result,
        "sleep_ids": sleep_ids,
        "codemap": code_index,
        "symbol": symbol,
        "flywheel": flywheel,
        "kv": kv_demo(),
        "retrieve_first": retrieve_first,
        "cascade": cascade("INFO hello\n" * 10, 50),
        "optional_neural_inference": "not run; install extras and explicitly choose models",
        "objects": [o.to_dict() for o in store.snapshot(session).objects],
        "events": store.events(session),
    }
    (output / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, text in [
        ("01-raw", raw),
        ("02-preprocessed", before),
        ("03-governed", after),
        ("04-recovered", recovered),
    ]:
        (output / (name + ".txt")).write_text(text, encoding="utf-8")
    report = f"""# Context Lab experiment

This report uses an offline synthetic case. Figures are teaching lexical units, not real model tokens; answer accuracy and provider bills were not measured.

## Inspect the changes

| View | Length |
|---|---:|
| Raw projection | {count(raw)} |
| After preprocessing | {count(before)} |
| After governance | {count(after)} |

Compare `01-raw.txt`, `02-preprocessed.txt`, and `03-governed.txt` in order. See `04-recovered.txt` for the complete original code. All original objects remain in `state/context.sqlite3`.

## Governance and caching

With a valid cache: `{canonical(deferred)}`.

Final commit: `{canonical(committed)}`. Cache cost and context length are evaluated separately; see cache_requests in trace.json for cache hit counts.

## Constraints and negative cases

Automatic assertions: `{canonical(assertions)}`.

Rejection reasons: `{canonical(failures)}`.

## Retrieval and memory

PRISM with all modules returned: {", ".join(e["id"] for e in retrieval["all"]["evidence"])}. Budget: 80; actual usage: {retrieval["all"]["units"]}. Inspect prism_ablations to compare results, paths, and candidate sets with N1–N4 removed; this small sample does not establish that any module is universally better.

SimpleMem wrote {len(memory_ids)} facts with provenance. The combined results from three views are in simplemem. Input facts are explicitly annotated here; only the optional model branch extracts facts automatically from natural language.

MemGPT uses ReplayModel to run core_replace → archival_search → answer as a scripted replay. SleepWorker ingests the facts again in the background; deduplication keeps the total at {len(store.memories("memory"))}.

## Structural reads, training candidates, and internal mechanisms

codemap stores version hashes and function line numbers; symbol returns the exact function. training_candidates contains {flywheel["sft"]} synthetic SFT candidates and {flywheel["dpo"]} synthetic DPO candidates, not a trained model.

The kv field demonstrates how appending preserves the prefix and editing affects later layers. retrieve_first shows a complete request in the alternative mode.

## Continue experimenting

From the project root, run `uv run context-lab inspect --state {output / "state"} --session payment` to inspect the governed view. Run `uv run context-lab query --state {output / "state"} --session memory "Why did Ada change the payment plan?"` to query the persisted memory again.

See [trace.json](trace.json) for all events, parameters, data, and ablation outputs. Real Ollama, Sentence Transformers, and LLMLingua inference was not run in this offline case.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    if not all(assertions.values()):
        raise AssertionError(assertions)
    return output
