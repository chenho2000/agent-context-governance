"""Versioned, source-cited JSON turn summaries and a separate slow pattern clock."""

from collections import defaultdict
from .types import canonical, digest


class TurnSummaries:
    fields = ("goals", "decisions", "constraints", "open_questions", "artifacts")

    def __init__(self, store):
        self.store = store

    def write(self, session, turn, model=None):
        snap = self.store.snapshot(session)
        objects = [o for o in snap.objects if o.turn == turn]
        if not objects:
            raise ValueError("unknown turn")
        allowed = {o.id for o in objects}
        if model:
            response = model.generate(
                "Return JSON with goals, decisions, constraints, open_questions, artifacts. "
                "Each field is a list of {text:string,sources:[object IDs]}. Use only supported facts. "
                "Preserve negations/numbers and leave uncertain items open.",
                {"objects": [o.to_dict() for o in objects]},
            )
        else:
            response = {field: [] for field in self.fields}
            for obj in objects:
                field = (
                    "constraints"
                    if obj.metadata.get("constraint")
                    else "goals"
                    if obj.kind == "user"
                    else "artifacts"
                    if obj.kind == "tool"
                    else "decisions"
                )
                response[field].append({"text": obj.content, "sources": [obj.id]})
        for field in self.fields:
            if not isinstance(response.get(field), list):
                raise ValueError("missing structured summary field")
            for item in response[field]:
                if (
                    not isinstance(item.get("text"), str)
                    or not item.get("sources")
                    or not set(item["sources"]) <= allowed
                ):
                    raise ValueError("invalid summary source")
        # Exact constraints are a separate spine even if the model omits or paraphrases them.
        spine = [{"text": o.content, "sources": [o.id]} for o in objects if o.metadata.get("constraint")]
        record = {field: response[field] for field in self.fields}
        record.update(
            session=session,
            turn=turn,
            exact_constraints=spine,
            source_hash=digest(canonical([o.to_dict() for o in objects])),
            authority="untrusted-derived",
            method="model" if model else "extractive",
        )
        self.store.set_block("summaries:" + session, f"{turn}:{record['source_hash']}", record)
        return record


def consolidate_patterns(store, sessions, min_support=2):
    """Cross-session connected components over shared symbolic failure labels. Not SimpleMem v3."""
    if min_support < 2:
        raise ValueError("patterns require independent support")
    sessions = sorted(set(sessions))
    by_label, adjacency = defaultdict(set), defaultdict(set)
    for session in sessions:
        for event in store.events(session):
            if event["kind"] == "failure":
                by_label[event["data"]["category"]].add(session)
    for group in by_label.values():
        for session in group:
            adjacency[session].update(group - {session})
    visited, communities = set(), []
    for session in sessions:
        if session in visited:
            continue
        stack, group = [session], set()
        while stack:
            node = stack.pop()
            if node in group:
                continue
            group.add(node)
            stack.extend(adjacency[node] - group)
        visited |= group
        if len(group) >= min_support:
            communities.append(sorted(group))
    patterns = [
        {"category": label, "sessions": sorted(group), "support": len(group)}
        for label, group in sorted(by_label.items())
        if len(group) >= min_support
    ]
    result = {
        "communities": communities,
        "patterns": patterns,
        "method": "symbolic connected-components; associations, not causal conclusions",
    }
    store.set_block("slow-clock", digest(canonical(sessions)), result)
    return result
