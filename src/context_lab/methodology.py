"""Revision lab: one executable walk-through connecting the article's proposed components."""

import json
from pathlib import Path
import tempfile
from .store import Store
from .fusion import FusionGraph
from .runtime import Runtime
from .summaries import TurnSummaries, consolidate_patterns
from .planning import tiered_plan
from .projection import prompt_zones, compaction_request
from .cache import ZonedCache, forecast_value
from .flywheel import clean_session
from .gating import information_gate


def run(parent="runs"):
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="methodology-", dir=parent))
    store = Store(output / "state")
    constraint = store.append(
        "lab", "Never charge twice; integer cents only.", turn=1, kind="user", metadata={"constraint": True}
    )
    old = store.append(
        "lab",
        "def charge(amount):\n    return gateway(amount)\n" + "# historical code\n" * 100,
        turn=1,
        tool="read_file",
        metadata={"path": "pay.py", "version": "v1", "range": [1, 102]},
    )
    store.append(
        "lab", "INFO retry\n" * 100, turn=1, tool="test", metadata={"redundant_log": True, "retry": True}
    )
    store.append(
        "lab", "Use the original function to compare the fix.", turn=2, kind="user", dependencies=(old.id,)
    )
    snap = store.snapshot("lab")
    graph = FusionGraph().build(snap)
    graph.save(store)
    summary = TurnSummaries(store).write("lab", 1)
    tiered = tiered_plan(snap)
    cache = ZonedCache()
    zones = [cache.request(prompt_zones(snap, dynamic=d), now=t) for d, t in [("a", 0), ("b", 1), ("c", 400)]]
    runtime = Runtime(store, "lab", capacity=5000)
    runtime.governor.graph = graph
    committed = runtime.compress_context(start_turn=1, end_turn=1, expired=True)
    # A view commit does not invalidate the graph's raw-source fingerprint.
    graph.validate_snapshot(store.snapshot("lab"))
    retrieved = runtime.context_for("charge", graph, budget=100)
    fallback = runtime.context_for("charge", available=False)
    runtime.close()
    for session in ["lab", "other"]:
        store.create_session(session)
        store.log(session, "failure", {"category": "repeated-read"})
    patterns = consolidate_patterns(store, ["lab", "other"])
    cleaned = clean_session(
        store,
        "lab",
        verified_result="synthetic: required evidence retained",
        required_ids=[constraint.id, old.id],
        directory=output / "candidates",
    )
    trace = {
        "fusion_nodes": graph.nodes,
        "fusion_edges": graph.edges,
        "summary": summary,
        "tiered": tiered.to_dict(),
        "gate": information_gate(snap.objects),
        "zone_costs": zones,
        "forecast": {str(p): forecast_value(5000, 4500, hit_probability=p) for p in [0, 0.5, 1]},
        "compaction_request": compaction_request(snap, "Summarize as JSON, preserving constraints."),
        "commit": committed,
        "retrieved": retrieved,
        "fallback": fallback,
        "patterns": patterns,
        "cleaned": cleaned,
    }
    (output / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# Methodology lab / 方法论组合实验

This is a deterministic teaching run, not a paper benchmark or a live model evaluation.
本次仅使用规则、真实存储和合成材料；没有神经模型调用。

| Step | Mechanism | Where to inspect in trace.json |
|---|---|---|
| 1 | Objects → Episodes → contains + typed dependencies | fusion_nodes / fusion_edges |
| 2 | Source-cited JSON summary + exact constraint spine | summary |
| 3 | Conservative gating + confidence cascade | gate / tiered |
| 4 | Per-zone TTL and planning-time expected value | zone_costs / forecast |
| 5 | Append-only structured compaction request | compaction_request |
| 6 | Model-triggerable bounded governance | commit |
| 7 | Retrieve-first keeps live dependencies | retrieved |
| 8 | External-store outage → in-place fallback | fallback |
| 9 | Cross-session patterns + clean trace | patterns / cleaned |

提交结果：`{committed}`。

可检索模式：`{retrieved["mode"]}`；断开外部检索后的模式：`{fallback["mode"]}`。

精确约束有 {len(summary["exact_constraints"])} 条。慢循环找到 {len(patterns["patterns"])} 个有两个独立会话支持的符号模式；这不证明因果。

[完整运行轨迹](trace.json) · [原理到实现指南](../../docs/原理到实现.md)
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return output
