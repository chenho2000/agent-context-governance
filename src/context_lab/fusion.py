"""Article's proposed fusion: Episode retrieval + addressable objects + typed dependency edges."""

from .retrieval import Prism
from .types import canonical, digest, StalePlan


class FusionGraph(Prism):
    layers = (*Prism.layers, "context_object")
    relations = Prism.relations | {"contains", "depends_on"}

    def __init__(self, embedder=None):
        super().__init__(embedder)
        self.source_fingerprint = None
        self.build_trace = []

    @staticmethod
    def fingerprint(snap):
        return digest(canonical([o.to_dict() for o in snap.objects]))

    def validate_snapshot(self, snap):
        if self.source_fingerprint != self.fingerprint(snap):
            raise StalePlan("graph source objects changed; rebuild graph")

    def build(self, snap, *, classifier=None, model=None, confidence=0.9):
        if not 0 < confidence <= 1:
            raise ValueError("invalid graph confidence")
        self.nodes, self.edges, self.build_trace = {}, {}, []
        self.source_fingerprint = self.fingerprint(snap)
        turns = {}
        for obj in snap.objects:
            turns.setdefault(obj.turn, []).append(obj)
            self.node(obj.id, "context_object", obj.content, sources=[obj.id])
        for turn, objects in turns.items():
            episode = f"{snap.session}:episode:{turn}"
            entity = f"{snap.session}:entity:session"
            point = f"{snap.session}:point:task"
            facet = f"{snap.session}:facet:history"
            for key, layer, text in [
                (entity, "entity", snap.session),
                (point, "facet_point", "task"),
                (facet, "facet", "conversation history"),
            ]:
                if key not in self.nodes:
                    self.node(key, layer, text)
            self.node(
                episode, "episode", "\n".join(o.content for o in objects), sources=[o.id for o in objects]
            )
            for a, b in [(entity, point), (point, facet), (facet, episode)]:
                if (b, "belongs_to", 1.0) not in self.edges.get(a, []):
                    self.edge(a, b, "belongs_to")
            for obj in objects:
                self.edge(episode, obj.id, "contains")
                for dep in obj.dependencies:
                    self.edge(obj.id, dep, "depends_on")
        # Temporal order is a rule; causal order is never inferred from adjacency alone.
        for previous, current in zip(snap.objects, snap.objects[1:]):
            self.edge(previous.id, current.id, "temporal")
        uncertain = []
        for previous, current in zip(snap.objects, snap.objects[1:]):
            pair = {
                "source": current.id,
                "target": previous.id,
                "text": current.content + "\nPREVIOUS:\n" + previous.content,
            }
            if previous.id in current.dependencies:
                self.build_trace.append({**pair, "tier": "rules", "relation": "depends_on"})
                continue
            if classifier:
                prediction = classifier.classify(
                    pair["text"], ["semantic", "causal", "evolution", "unrelated"]
                )
                if prediction["score"] >= confidence:
                    if prediction["label"] != "unrelated":
                        self.edge(current.id, previous.id, prediction["label"])
                    self.build_trace.append({**pair, "tier": "encoder", **prediction})
                    continue
            uncertain.append(pair)
        if model and uncertain:
            answer = model.generate(
                "Return {edges:[{source,target,relation}]}. Only classify supplied pairs. "
                "relation=semantic|causal|evolution|unrelated. Never invent dependencies.",
                {"pairs": uncertain},
            )
            allowed = {(p["source"], p["target"]) for p in uncertain}
            seen = set()
            for edge in answer["edges"]:
                pair = edge["source"], edge["target"]
                relation = edge["relation"]
                if (
                    pair not in allowed
                    or pair in seen
                    or relation not in {"semantic", "causal", "evolution", "unrelated"}
                ):
                    raise ValueError("invalid model edge")
                seen.add(pair)
                if relation != "unrelated":
                    self.edge(*pair, relation)
                self.build_trace.append({**edge, "tier": "llm"})
        return self

    def live_objects(self, roots):
        # Causal/semantic links conservatively ADD liveness, never erase an explicit dependency.
        live, stack = set(), list(roots)
        while stack:
            node = stack.pop()
            if node in live:
                continue
            live.add(node)
            stack.extend(
                target
                for target, relation, _ in self.edges.get(node, [])
                if relation in {"depends_on", "causal", "semantic", "contains"}
            )
        return {i for i in live if self.nodes.get(i, {}).get("layer") == "context_object"}

    def save(self, store, namespace="fusion"):
        store.set_block(
            namespace,
            "graph",
            {
                "nodes": self.nodes,
                "edges": self.edges,
                "source_fingerprint": self.source_fingerprint,
                "build_trace": self.build_trace,
            },
        )

    @classmethod
    def load(cls, store, namespace="fusion", embedder=None):
        obj = cls(embedder)
        data = store.blocks(namespace)["graph"]
        obj.nodes, obj.edges = data["nodes"], data["edges"]
        obj.source_fingerprint, obj.build_trace = data["source_fingerprint"], data["build_trace"]
        return obj

    def tune(self, *, inactive=(), edge_costs=()):
        """Validated retrieval-only edits; do not change object liveness or raw data."""
        if not set(inactive) <= set(self.nodes):
            raise ValueError("unknown node")
        updates = []
        for source, target, relation, cost in edge_costs:
            edges = self.edges.get(source, [])
            if cost <= 0 or not any(t == target and r == relation for t, r, _ in edges):
                raise ValueError("unknown edge or invalid cost")
            updates.append((source, target, relation, float(cost)))
        for node in inactive:
            self.nodes[node]["inactive"] = True
        for source, target, relation, cost in updates:
            self.edges[source] = [
                (t, r, cost if t == target and r == relation else c) for t, r, c in self.edges[source]
            ]
