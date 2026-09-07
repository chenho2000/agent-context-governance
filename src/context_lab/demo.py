"""可重复的端到端实验；所有数字由本次运行产生。"""

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
        ("ada", "entity", "Ada 支付负责人"),
        ("payment", "facet_point", "Ada 支付方案 payment"),
        ("retry", "facet", "支付重试的重复扣款风险"),
        ("e1", "episode", "9月5日 Ada 将支付重试改为幂等键。"),
        ("e2", "episode", "原因：测试表明没有幂等键时，支付重试导致重复扣款。"),
        ("e3", "episode", "9月6日 Ada 讨论了数据库连接池大小。"),
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
        session, "不要重复扣款；金额精确到分。", turn=1, kind="user", metadata={"constraint": True}
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
        session, "网络波动是唯一原因。已否定。", turn=2, tool="search", metadata={"obsolete": True}
    )
    sparse = store.append(
        session,
        'Traceback\n  File "payment.py", line 2\nDuplicateCharge',
        turn=2,
        tool="test",
        metadata={"exact_now": True},
    )
    store.append(session, "用幂等键修复，稍后查看旧代码。", turn=3, kind="user", dependencies=(first.id,))
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
    store.append(session, "继续核对测试结果。", turn=4, kind="user")
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
    query = "为什么 Ada 改变支付方案？"
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
                "text": "Ada 在 2026-09-05 将支付方案改为幂等键。",
            },
            {
                "entity": "Ada",
                "slot": "reason",
                "value": "duplicate_charge",
                "date": "2026-09-06",
                "text": "Ada 的支付方案变更原因是测试发现重复扣款。",
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
                    "args": {"name": "task", "text": "核对支付方案"},
                    "request_heartbeat": True,
                },
                {"tool": "archival_search", "args": {"query": query}, "request_heartbeat": True},
                {"answer": "回放示例：为了避免重复扣款，Ada 采用幂等键。"},
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
    prefix = {"visible_ids": [constraint.id, first.id], "query": "修复支付"}
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
    report = f"""# Context Lab 本次实验

本报告由离线合成案例生成。数字是教学词法单位，不是真实模型 token；未测量回答准确率或供应商账单。

## 从这里看变化

| 视图 | 长度 |
|---|---:|
| 原始投影 | {count(raw)} |
| 预处理后 | {count(before)} |
| 治理后 | {count(after)} |

按顺序比较 `01-raw.txt`、`02-preprocessed.txt`、`03-governed.txt`，在 `04-recovered.txt` 查看旧版本完整代码。所有原始对象仍在 `state/context.sqlite3`。

## 治理与缓存

缓存有效时：`{canonical(deferred)}`。

最终提交：`{canonical(committed)}`。缓存代价与上下文长度分开判断；具体命中量见 trace.json 的 cache_requests。

## 约束与反例

自动断言：`{canonical(assertions)}`。

拒绝原因：`{canonical(failures)}`。

## 检索与记忆

PRISM 全模块返回：{", ".join(e["id"] for e in retrieval["all"]["evidence"])}。预算 80，实际 {retrieval["all"]["units"]}。查看 prism_ablations 比较 N1—N4 的移除结果、路径和候选集；小样本不证明任何模块普遍更好。

SimpleMem 写入 {len(memory_ids)} 条带来源事实，三视图合并结果在 simplemem。这里输入事实有显式标注；可选模型分支才自动抽取自然语言。

MemGPT 使用 ReplayModel 跑 core_replace → archival_search → 回答，明确是脚本回放。SleepWorker 后台再次摄取，去重后仍是 {len(store.memories("memory"))} 条。

## 结构读取、训练候选与内部机制

codemap 保存版本哈希和函数行号，symbol 返回精确函数。training_candidates 含 {flywheel["sft"]} 条 SFT、{flywheel["dpo"]} 条 DPO 合成候选，不是训练完成的模型。

kv 字段展示追加保持前缀、编辑影响后续层的因果关系。retrieve_first 展示另一种模式的完整请求。

## 继续实验

在项目根目录运行 `uv run context-lab inspect --state {output / "state"} --session payment` 查看治理后的视图；运行 `uv run context-lab query --state {output / "state"} --session memory "为什么 Ada 改变支付方案？"` 重新检索持久化记忆。

完整事件、参数、数据与消融输出见 [trace.json](trace.json)。真实 Ollama、Sentence Transformers、LLMLingua 推理未在此离线案例中运行。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    if not all(assertions.values()):
        raise AssertionError(assertions)
    return output
