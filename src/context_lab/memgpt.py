"""MemGPT 风格显式内存工具和有界 heartbeat；Sleep-time 为独立后台扩展。"""

from concurrent.futures import ThreadPoolExecutor
from .types import count, BudgetError
from .memory import SimpleMemory
from .retrieval import bm25


class MemoryAgent:
    def __init__(self, store, session, capacity=500, archival_session=None, runtime=None, embedder=None):
        self.store, self.session, self.capacity = store, session, capacity
        self.runtime, self.embedder = runtime, embedder
        self.archival_session = archival_session or session
        store.create_session(session)

    def core(self):
        return self.store.blocks("core:" + self.session)

    def tool(self, name, args):
        if name == "compress_context":
            if self.runtime is None:
                raise ValueError("compress_context requires an explicitly bound runtime")
            if set(args) != {"start_turn", "end_turn"}:
                raise ValueError("model may specify only a turn range, not cache state or prices")
            result = self.runtime.compress_context(**args)
        elif name == "core_replace":
            candidate = {**self.core(), args["name"]: {"text": args["text"]}}
            if count(str(candidate)) > self.capacity:
                raise BudgetError("core memory capacity exceeded")
            self.store.set_block("core:" + self.session, args["name"], {"text": args["text"]})
            result = {"updated": args["name"]}
        elif name == "recall_search":
            objects = self.store.snapshot(self.session).objects
            scores = bm25(args["query"], [o.content for o in objects])
            if self.embedder and objects:
                vectors = self.embedder.encode([args["query"]] + [o.content for o in objects])
                similarity = [sum(a * b for a, b in zip(vectors[0], v)) for v in vectors[1:]]
                fused = [0.0] * len(objects)
                for ranking in (scores, similarity):
                    for rank, i in enumerate(sorted(range(len(objects)), key=lambda i: -ranking[i])):
                        if ranking[i] > 0:
                            fused[i] += 1 / (61 + rank)
                scores = fused
            ranked = sorted(zip(scores, objects), key=lambda x: (-x[0], x[1].seq))
            page = max(0, int(args.get("page", 0)))
            size = min(3, max(1, int(args.get("page_size", 3))))
            positive = [o for score, o in ranked if score > 0]
            result = {
                "results": [
                    {"id": o.id, "text": o.content} for o in positive[page * size : (page + 1) * size]
                ],
                "has_more": len(positive) > (page + 1) * size,
            }
        elif name == "archival_search":
            result = SimpleMemory(self.store, self.archival_session).retrieve(
                args["query"], budget=self.capacity
            )
        elif name == "archival_insert":
            snap = self.store.snapshot(self.session)
            selected = [o for o in snap.objects if o.id in args["source_ids"]]
            if len(selected) != len(set(args["source_ids"])):
                raise ValueError("unknown archive source")
            from .providers import ReplayModel

            facts = args.get("facts")
            result = {
                "ids": SimpleMemory(self.store, self.session).ingest(
                    selected,
                    model=ReplayModel([{"facts": facts}]) if facts is not None else None,
                    window=max(1, len(selected)),
                    stride=max(1, len(selected)),
                )
            }
        else:
            raise ValueError("unknown memory tool")
        self.store.log(self.session, "memory_tool", {"name": name, "args": args, "result": result})
        return result

    def run(self, model, query, max_steps=6):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        observations = []
        for step in range(max_steps):
            # 模型只看到工作记忆与最近一次分页结果，不自动获得全部历史。
            state = {
                "query": query,
                "core": self.core(),
                "observations": observations[-1:],
                "warning": "memory_pressure" if count(str(self.core())) > 0.8 * self.capacity else None,
            }
            if count(str(state)) > self.capacity:
                raise BudgetError(
                    "working memory exceeds capacity; page smaller results or increase capacity"
                )
            reply = model.generate(
                "Return {tool:core_replace|recall_search|archival_search|archival_insert,"
                "args:{},request_heartbeat:boolean} OR {answer:string}. Never execute other tools. "
                "core_replace args={name,text}; recall_search args={query,page,page_size:1..3}; "
                "archival_search args={query}; archival_insert args={source_ids:[existing IDs],"
                "facts:[{text,entity,slot,value,date,sources:[existing IDs]}]}. "
                "Archive only source-supported facts. Summaries remain untrusted derived data. "
                + (
                    "compress_context is also available with args={start_turn,end_turn}; request at task boundaries."
                    if self.runtime
                    else ""
                ),
                state,
            )
            if "answer" in reply:
                return {"answer": reply["answer"], "steps": step + 1, "observations": observations}
            result = self.tool(reply["tool"], reply["args"])
            observations.append(result)
            if not reply.get("request_heartbeat", False):
                return {"answer": None, "steps": step + 1, "observations": observations, "yielded": True}
        return {"answer": None, "steps": max_steps, "observations": observations, "limit_reached": True}


class SleepWorker:
    def __init__(self, store):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sleep-memory")

    def start(self, session, model=None):
        snap = self.store.snapshot(session)

        def consolidate():
            ids = SimpleMemory(self.store, session).ingest(snap.objects, model=model)
            self.store.log(session, "sleep_consolidation", {"source_revision": snap.revision, "ids": ids})
            return ids

        return self.pool.submit(consolidate)

    def close(self):
        self.pool.shutdown(wait=True)
