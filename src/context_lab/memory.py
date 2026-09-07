"""SimpleMem 思路：窗口 → 原子事实/日期消歧 → 在线合并 → 三视图检索。"""

from datetime import datetime, timedelta
from .types import canonical, digest, count
from .retrieval import HashEmbedding, bm25, route


class SimpleMemory:
    def __init__(self, store, session, embedder=None):
        self.store, self.session = store, session
        self.embedder = embedder or HashEmbedding()

    def ingest(self, objects, *, model=None, window=20, stride=15, classifier=None):
        if not 0 < stride <= window:
            raise ValueError("stride must be between 1 and window")
        sources = {o.id: o for o in objects}
        if any(o.session != self.session for o in objects):
            raise ValueError("foreign memory source")
        if sources:
            persisted = self.store.snapshot(self.session).by_id()
            if any(i not in persisted or o.to_dict() != persisted[i].to_dict() for i, o in sources.items()):
                raise ValueError("memory source does not match immutable record")
        added = []
        for start in range(0, len(objects), stride):
            chunk = objects[start : start + window]
            from .gating import information_gate

            gate = information_gate(chunk, classifier)
            self.store.log(self.session, "memory_gate", {"start": start, **gate})
            if not gate["keep"]:
                continue
            if model:
                result = model.generate(
                    "Extract only explicit new facts. Return {facts:[{text,entity,slot,value,"
                    "sources:[id],date:ISO-date-or-null}]}. Resolve references only if unambiguous. "
                    "Omit greetings. Preserve contradictions and dates.",
                    {
                        "objects": [o.to_dict() for o in chunk],
                        "existing_memory": self.store.memories(self.session),
                    },
                )
                facts = result["facts"]
            else:
                # 离线分支消费带标注事实，不假装能自动理解任意对话。
                facts = [{**f, "sources": [o.id]} for o in chunk for f in o.metadata.get("facts", [])]
            old = {m["id"]: m for m in self.store.memories(self.session)}
            pending_records = []
            for f in facts:
                if not isinstance(f.get("text"), str) or not f["text"].strip():
                    raise ValueError("empty fact")
                if not f.get("sources") or not set(f["sources"]) <= {o.id for o in chunk}:
                    raise ValueError("unsupported fact provenance")
                if any(sources[i].session != self.session for i in f["sources"]):
                    raise ValueError("foreign memory source")
                if not all(isinstance(f.get(k), str) and f[k] for k in ("entity", "slot", "value")):
                    raise ValueError("fact needs entity/slot/value")
                dates = {sources[i].metadata.get("date") for i in f["sources"]}
                date = f.get("date")
                if date in {"yesterday", "tomorrow", "昨天", "明天"}:
                    if len(dates) != 1 or None in dates:
                        raise ValueError("relative date requires one explicit anchor")
                    delta = -1 if date in {"yesterday", "昨天"} else 1
                    date = (
                        (datetime.fromisoformat(next(iter(dates))) + timedelta(days=delta)).date().isoformat()
                    )
                if date:
                    date = datetime.fromisoformat(date).date().isoformat()
                f = dict(f)
                if date and f.get("date") in {"yesterday", "tomorrow", "昨天", "明天"}:
                    f["text"] = f["text"].replace(f["date"], date)
                # 只有输入明确给出的绑定可以用于离线指代消歧。
                for source_id in f["sources"]:
                    for pronoun, entity in sources[source_id].metadata.get("bindings", {}).items():
                        f["text"] = f["text"].replace(pronoun, entity)
                key = digest(canonical([self.session, f["entity"], f["slot"], f["value"], date]))
                observed_seq = max(sources[i].seq for i in f["sources"])
                if key in old:
                    record = old[key]
                    record["sources"] = sorted(set(record["sources"]) | set(f["sources"]))
                else:
                    record = {
                        **f,
                        "id": key,
                        "session": self.session,
                        "date": date,
                        "supersedes": [],
                        "authority": "untrusted-derived",
                        "observed_seq": observed_seq,
                    }
                # Explicit IDs are preferred. Legacy correction=True only targets earlier observations.
                targets = f.get("supersedes")
                if targets is not None:
                    if not isinstance(targets, list) or not all(isinstance(i, str) for i in targets):
                        raise ValueError("supersedes must be a list of memory IDs")
                elif f.get("correction"):
                    targets = [
                        m["id"]
                        for m in old.values()
                        if m["entity"] == f["entity"]
                        and m["slot"] == f["slot"]
                        and m["value"] != f["value"]
                        and m.get("observed_seq", 0) < observed_seq
                    ]
                else:
                    targets = []
                for target in targets:
                    other = old.get(target)
                    if (
                        target == key
                        or other is None
                        or other["entity"] != f["entity"]
                        or other["slot"] != f["slot"]
                    ):
                        raise ValueError("invalid or unrelated correction target")
                record["supersedes"] = sorted(set(record.get("supersedes", [])) | set(targets))
                old[key] = record
                pending_records.append(record)
            self.store.put_memories(pending_records)
            added.extend(record["id"] for record in pending_records)
        self.store.log(
            self.session,
            "memory_ingest",
            {
                "input_objects": len(objects),
                "unique_facts": len(set(added)),
                "window": window,
                "stride": stride,
                "extractor": "model" if model else "annotated-facts",
            },
        )
        return sorted(set(added))

    def retrieve(
        self,
        query,
        *,
        budget=300,
        top_k=5,
        entity=None,
        date_from=None,
        date_to=None,
        model=None,
        include_superseded=False,
        max_queries=4,
    ):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty")
        if budget < 0 or top_k < 1 or max_queries < 1:
            raise ValueError("invalid retrieval budget")
        intent, router = route(query, self.embedder, model)
        if model:
            plan = model.generate(
                "Return {queries:[strings],top_k:positive-int} for memory search.",
                {"query": query, "intent": intent},
            )
            queries = plan["queries"]
            if (
                not isinstance(queries, list)
                or not queries
                or len(queries) > max_queries
                or not all(isinstance(q, str) and q.strip() for q in queries)
            ):
                raise ValueError("invalid query plan")
            queries = list(dict.fromkeys(queries))
            top_k = min(top_k, max(1, int(plan["top_k"])))
        else:
            queries = [query]
        all_rows = self.store.memories(self.session)
        superseded = {i for m in all_rows for i in m.get("supersedes", [])}
        rows = [
            m
            for m in all_rows
            if (include_superseded or m["id"] not in superseded)
            and (not entity or m["entity"] == entity)
            and (not date_from or (m.get("date") and m["date"] >= date_from))
            and (not date_to or (m.get("date") and m["date"] <= date_to))
        ]
        scores, views = {}, {}
        vectors = self.embedder.encode(queries + [m["text"] for m in rows])
        document_vectors = vectors[len(queries) :]
        for q, query_vector in zip(queries, vectors[: len(queries)]):
            lexical = bm25(q, [m["text"] for m in rows])
            semantic = [sum(a * b for a, b in zip(query_vector, v)) for v in document_vectors]
            for name, rank in [
                ("bm25", sorted(range(len(rows)), key=lambda i: -lexical[i])),
                ("vector", sorted(range(len(rows)), key=lambda i: -semantic[i])),
            ]:
                ranking = lexical if name == "bm25" else semantic
                rank = [idx for idx in rank if ranking[idx] > 0]
                for pos, idx in enumerate(rank[:top_k]):
                    mid = rows[idx]["id"]
                    scores[mid] = scores.get(mid, 0) + 1 / (60 + pos + 1)  # RRF，不直接混合不同量纲
                    views.setdefault(mid, set()).add(name)
            for row in rows:
                if row["entity"].lower() in q.lower() or entity:
                    scores[row["id"]] = scores.get(row["id"], 0) + 1 / 61
                    views.setdefault(row["id"], set()).add("symbolic")
        selected, used = [], 0
        by_id = {m["id"]: m for m in rows}
        for mid in sorted(scores, key=lambda i: (-scores[i], i)):
            m = by_id[mid]
            size = count(m["text"])
            if used + size <= budget and len(selected) < top_k:
                selected.append({**m, "views": sorted(views[mid]), "score": scores[mid]})
                used += size
        return {
            "memories": selected,
            "units": used,
            "plan": {
                "queries": queries,
                "intent": intent,
                "router": router,
                "top_k": top_k,
                "entity": entity,
                "date_from": date_from,
                "date_to": date_to,
            },
        }
