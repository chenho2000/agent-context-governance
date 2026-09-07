"""context-lab CLI：所有网络后端由显式参数启用。"""

import argparse
import importlib.util
import json
from pathlib import Path
from .store import Store


def main(argv=None):
    parser = argparse.ArgumentParser(description="Context Lab 上下文治理学习实验台")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="离线跑所有核心机制并生成报告")
    demo.add_argument("--output", default="runs")
    sub.add_parser("doctor", help="只检查依赖，不下载模型")
    coverage = sub.add_parser("coverage", help="检查方法清单中的实现与测试入口")
    coverage.add_argument("--check", action="store_true")
    lab = sub.add_parser("methodology", help="运行融合方案与改进实验")
    lab.add_argument("--output", default="runs")
    for name in (
        "ingest",
        "inspect",
        "recover",
        "plan",
        "commit",
        "resume",
        "query",
        "memory-ingest",
        "events",
        "tick",
        "summarize",
        "compact-request",
    ):
        p = sub.add_parser(name)
        p.add_argument("--state", default=".context-lab")
        p.add_argument("--session", default="learning")
        if name == "ingest":
            p.add_argument("file")
        if name == "recover":
            p.add_argument("object_id")
        if name in {"plan", "memory-ingest", "query", "tick", "summarize", "compact-request"}:
            p.add_argument("--ollama-model")
            p.add_argument("--ollama-url", default="http://localhost:11434")
        if name == "summarize":
            p.add_argument("--turn", type=int, required=True)
        if name == "compact-request":
            p.add_argument("--instruction", default="Summarize as JSON; preserve constraints and source IDs.")
        if name == "query":
            p.add_argument("query")
            p.add_argument("--budget", type=int, default=300)
            p.add_argument("--embedding-model")
            p.add_argument("--entity")
        if name in {"commit", "tick", "resume"}:
            p.add_argument("--future-calls", type=int, default=1)
            p.add_argument("--expired", action="store_true")
            p.add_argument("--capacity", type=int, default=4000)
    compress = sub.add_parser("compress")
    compress.add_argument("file")
    compress.add_argument(
        "--method", choices=["rules", "local", "llmlingua", "longllmlingua", "llmlingua2"], default="rules"
    )
    compress.add_argument("--model")
    compress.add_argument("--device", default="cpu")
    compress.add_argument("--question", default="")
    compress.add_argument("--rate", type=float, default=0.5)
    compress.add_argument("--budget", type=int, default=200)
    code = sub.add_parser("codemap")
    code.add_argument("root")
    code.add_argument("--state", default=".context-lab")
    code.add_argument("--file")
    code.add_argument("--symbol")
    code.add_argument("--query")
    code.add_argument("--embedding-model")
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
        if result is not None:
            print(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
        parser.exit(2, f"context-lab: {type(error).__name__}: {error}\n")


def dispatch(args):
    if args.command == "demo":
        from .demo import run

        return str(run(args.output) / "report.md")
    if args.command == "methodology":
        from .methodology import run

        return str(run(args.output) / "report.md")
    if args.command == "coverage":
        from .coverage import audit

        return audit(check=args.check)
    if args.command == "doctor":
        return {
            "python_stdlib_core": True,
            "ollama": "HTTP adapter available; server/model not probed",
            "llmlingua_installed": importlib.util.find_spec("llmlingua") is not None,
            "sentence_transformers_installed": importlib.util.find_spec("sentence_transformers") is not None,
            "note": "Installed does not mean model weights are available or inference was validated.",
        }
    if args.command == "compress":
        from .providers import Lingua, Ollama
        from .compression import compress

        text = Path(args.file).read_text(encoding="utf-8")
        if args.method.startswith("llmlingua") or args.method == "longllmlingua":
            return Lingua(args.method, args.model, args.device).compress(
                text, rate=args.rate, question=args.question
            )
        if args.method == "local" and not args.model:
            raise ValueError("local method requires --model")
        return compress(
            text, method=args.method, budget=args.budget, model=Ollama(args.model) if args.model else None
        )
    store = Store(args.state)
    if args.command == "codemap":
        from .codemap import CodeMap

        index = CodeMap(store, args.root)
        if bool(args.file) != bool(args.symbol):
            raise ValueError("--file and --symbol must be supplied together")
        if args.query:
            from .retrieval import SentenceEmbedding

            embedder = SentenceEmbedding(args.embedding_model) if args.embedding_model else None
            return index.search(args.query, embedder=embedder)
        return index.get_symbol(args.file, args.symbol) if args.symbol else index.build()
    if args.command == "ingest":
        rows = json.loads(Path(args.file).read_text(encoding="utf-8"))
        return {"ids": [store.append(args.session, **row).id for row in rows]}
    if args.command == "inspect":
        from .projection import prompt

        return prompt(store.snapshot(args.session))
    if args.command == "recover":
        return store.recover(args.session, args.object_id)
    if args.command == "events":
        return store.events(args.session)
    model = None
    if getattr(args, "ollama_model", None):
        from .providers import Ollama

        model = Ollama(args.ollama_model, args.ollama_url)
    if args.command == "summarize":
        from .summaries import TurnSummaries

        return TurnSummaries(store).write(args.session, args.turn, model)
    if args.command == "compact-request":
        from .projection import compaction_request

        request = compaction_request(store.snapshot(args.session), args.instruction)
        return model.generate_messages(request["messages"], request["tools"]) if model else request
    if args.command in {"query", "memory-ingest"}:
        from .memory import SimpleMemory

        embedder = None
        if getattr(args, "embedding_model", None):
            from .retrieval import SentenceEmbedding

            embedder = SentenceEmbedding(args.embedding_model)
        memory = SimpleMemory(store, args.session, embedder)
        if args.command == "memory-ingest":
            return {"ids": memory.ingest(store.snapshot(args.session).objects, model=model)}
        return memory.retrieve(args.query, budget=args.budget, entity=args.entity, model=model)
    if args.command in {"tick", "resume"}:
        from .runtime import Runtime

        runtime = Runtime(store, args.session, args.capacity)
        try:
            if args.command == "resume":
                return runtime.resume_pending(future_calls=args.future_calls, expired=args.expired)
            return runtime.tick(model=model, future_calls=args.future_calls, expired=args.expired)
        finally:
            runtime.close()
    from .governance import Governor

    governor = Governor(store)
    try:
        if args.command == "plan":
            plan = governor.fork(args.session, model).result()
            return {"plan": plan.to_dict(), "rehearsal": governor.stage(plan)}
        if args.command == "commit":
            return governor.commit(
                args.session, future_calls=args.future_calls, expired=args.expired, hard_limit=args.capacity
            )
    finally:
        governor.close()


if __name__ == "__main__":
    main()
