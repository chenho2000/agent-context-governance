"""PRISM 启发的四层图检索。路径和评分全部可观察，不声称复现论文指标。"""

from collections import Counter
import hashlib
import heapq
import math
import re
from .types import count, BudgetError


def terms(text):
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())


class HashEmbedding:
    """非神经词袋哈希，仅用于离线跑通控制流。"""

    def encode(self, texts):
        result = []
        for text in texts:
            vector = [0.0] * 128
            for term in terms(text):
                h = hashlib.sha256(term.encode()).digest()
                vector[int.from_bytes(h[:2], "big") % 128] += 1 if h[2] % 2 else -1
            norm = math.sqrt(sum(x * x for x in vector)) or 1
            result.append([x / norm for x in vector])
        return result


class SentenceEmbedding:
    def __init__(self, model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def bm25(query, documents, k1=1.5, b=0.75):
    bags = [Counter(terms(d)) for d in documents]
    avg = sum(sum(x.values()) for x in bags) / max(len(bags), 1) or 1
    query_terms = set(terms(query))
    dfs = Counter(word for bag in bags for word in bag if word in query_terms)
    result = []
    for bag in bags:
        score = 0.0
        for word in query_terms:
            df = dfs[word]
            idf = math.log(1 + (len(bags) - df + 0.5) / (df + 0.5))
            tf = bag[word]
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * sum(bag.values()) / avg))
        result.append(score)
    return result


def route(query, embedder=None, model=None):
    prototypes = {
        "causal": "why reason cause 为什么 原因 导致",
        "temporal": "when before after date 何时 之前 之后 日期",
        "evolution": "changed formerly now 更新 变化 更正 现在",
        "semantic": "what detail explain 什么 内容 解释",
    }
    for intent in ("causal", "temporal", "evolution"):
        if any(word in query.lower() for word in prototypes[intent].split()):
            return intent, "keyword"
    if model:
        answer = model.generate("Return {intent:causal|temporal|evolution|semantic}.", {"query": query})
        if answer.get("intent") not in prototypes:
            raise ValueError("invalid intent")
        return answer["intent"], "llm"
    vectors = (embedder or HashEmbedding()).encode([query] + list(prototypes.values()))
    scores = [sum(a * b for a, b in zip(vectors[0], v)) for v in vectors[1:]]
    return list(prototypes)[max(range(4), key=lambda i: scores[i])], "prototype"


class Prism:
    layers = ("entity", "facet_point", "facet", "episode")
    relations = {"belongs_to", "semantic", "temporal", "causal", "evolution", "involves_entity"}

    def __init__(self, embedder=None):
        self.nodes, self.edges = {}, {}
        self.embedder = embedder or HashEmbedding()

    def node(self, key, layer, text, **metadata):
        if layer not in self.layers:
            raise ValueError("unknown layer")
        self.nodes[key] = {"id": key, "layer": layer, "text": text, **metadata}
        return key

    def edge(self, source, target, relation, cost=1.0):
        if (
            source not in self.nodes
            or target not in self.nodes
            or relation not in self.relations
            or cost <= 0
        ):
            raise ValueError("invalid typed edge")
        self.edges.setdefault(source, []).append((target, relation, cost))

    def save(self, store, namespace="prism"):
        store.set_block(namespace, "graph", {"nodes": self.nodes, "edges": self.edges})

    @classmethod
    def load(cls, store, namespace="prism", embedder=None):
        obj = cls(embedder)
        data = store.blocks(namespace)["graph"]
        obj.nodes, obj.edges = data["nodes"], data["edges"]
        return obj

    def retrieve(
        self, query, *, budget=300, anchors=1, max_hops=4, n1=True, n2=True, n3=True, n4=True, model=None
    ):
        if budget < 0 or anchors < 1 or max_hops < 0:
            raise ValueError("invalid retrieval budget")
        intent, router = route(query, self.embedder, model) if n4 else ("semantic", "disabled")
        ids = [i for i, node in self.nodes.items() if not node.get("inactive")]
        if not ids:
            return {"evidence": [], "trace": [], "units": 0, "intent": intent}
        vectors = self.embedder.encode([query] + [self.nodes[i]["text"] for i in ids])
        similarities = {i: sum(a * b for a, b in zip(vectors[0], v)) for i, v in zip(ids, vectors[1:])}
        starts = []
        for layer in self.layers if n1 else ("episode",):
            candidates = [i for i in ids if self.nodes[i]["layer"] == layer]
            starts.extend(sorted(candidates, key=lambda i: (-similarities[i], i))[:anchors])
        heap = [(max(0.0, 1 - similarities[i]), i, 0, 0, (i,)) for i in starts]
        heapq.heapify(heap)
        best, episodes, trace = {}, {}, []
        while heap:
            cost, node, hops, bridges, path = heapq.heappop(heap)
            state = node, hops, bridges
            if cost >= best.get(state, float("inf")):
                continue
            best[state] = cost
            if self.nodes[node]["layer"] == "episode":
                if node not in episodes or cost < episodes[node]["cost"]:
                    episodes[node] = {**self.nodes[node], "cost": cost, "path": list(path)}
            if not n1 or hops >= max_hops:
                continue
            for target, relation, base in self.edges.get(node, []):
                if self.nodes[target].get("inactive"):
                    continue
                bridge = int(relation != "belongs_to")
                if bridges + bridge > 1 or target in path:
                    continue
                discount = 0.4 if n2 and relation == intent else 1.0
                step = base * discount
                trace.append({"from": node, "to": target, "relation": relation, "step": step})
                heapq.heappush(heap, (cost + step, target, hops + 1, bridges + bridge, (*path, target)))
        candidates = sorted(episodes.values(), key=lambda e: (e["cost"], e["id"]))
        # N3 只读取 episode 内容，不能用路径成本冒充内容相关性。
        if n3 and model and candidates:
            answer = model.generate(
                "Return {ids:[episode IDs]} ranked only by content relevance. Do not invent IDs.",
                {"query": query, "episodes": [{"id": e["id"], "text": e["text"]} for e in candidates]},
            )
            order = answer["ids"]
            if len(set(order)) != len(order) or not set(order) <= {e["id"] for e in candidates}:
                raise ValueError("invalid evidence IDs")
            by_id = {e["id"]: e for e in candidates}
            candidates = [by_id[i] for i in order]
        elif n3:
            scores = bm25(query, [e["text"] for e in candidates])
            candidates = [e for _, e in sorted(zip(scores, candidates), key=lambda p: (-p[0], p[1]["id"]))]
        selected, used = [], 0
        for episode in candidates:
            size = count(episode["text"])
            if used + size <= budget:
                selected.append(episode)
                used += size
        if used > budget:
            raise BudgetError("evidence exceeds budget")
        return {
            "evidence": selected,
            "units": used,
            "intent": intent,
            "router": router,
            "trace": trace,
            "candidate_ids": [e["id"] for e in candidates],
            "modules": {"N1": n1, "N2": n2, "N3": n3, "N4": n4},
        }
