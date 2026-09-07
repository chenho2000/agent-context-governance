"""Pack complete evidence against the actual serialized request, without slicing individual facts."""

from dataclasses import dataclass
from .types import BudgetError, canonical, count
from .projection import prompt


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    sources: tuple[str, ...]

    @classmethod
    def from_row(cls, row):
        if not isinstance(row, dict) or not all(
            isinstance(row.get(k), str) and row[k] for k in ("id", "text")
        ):
            raise ValueError("evidence requires nonempty string id/text")
        sources = row.get("sources", [row["id"]])
        if (
            not isinstance(sources, (list, tuple))
            or not sources
            or not all(isinstance(s, str) and s for s in sources)
        ):
            raise ValueError("invalid evidence sources")
        return cls(row["id"], row["text"], tuple(dict.fromkeys(sources)))

    def render(self):
        return canonical(
            {
                "evidence_id": self.id,
                "source_ids": list(self.sources),
                "authority": "untrusted-evidence",
                "text": self.text,
            }
        )


def pack_evidence(snap, states, query, rows, *, capacity, body_budget, output_reserve=0):
    if not all(
        isinstance(n, int) and not isinstance(n, bool) for n in (capacity, body_budget, output_reserve)
    ):
        raise ValueError("budgets must be integer teaching units")
    if capacity < 1 or min(body_budget, output_reserve) < 0 or output_reserve >= capacity:
        raise ValueError("invalid budget or output reserve")
    available = capacity - output_reserve
    base = prompt(snap, states, dynamic=query)
    if count(base) > available:
        raise BudgetError("protected context and query alone exceed capacity after output reserve")
    parsed = [Evidence.from_row(row) for row in rows]
    unique = {}
    for item in parsed:
        if item.id in unique and item != unique[item.id]:
            raise ValueError("conflicting payloads for one evidence ID")
        unique[item.id] = item
    selected, omitted, bodies, rendered = [], [], 0, []
    composed = base
    for item in unique.values():
        if bodies + count(item.text) > body_budget:
            omitted.append({"id": item.id, "reason": "body_budget"})
            continue
        trial = prompt(snap, states, dynamic=query + "\n" + "\n".join([*rendered, item.render()]))
        if count(trial) > available:
            omitted.append({"id": item.id, "reason": "serialized_request_budget"})
            continue
        selected.append(item.id)
        bodies += count(item.text)
        rendered.append(item.render())
        composed = trial
    return {
        "prompt": composed,
        "units": count(composed),
        "packing": {
            "base_units": count(base),
            "body_units": bodies,
            "selected_ids": selected,
            "omitted": omitted,
            "output_reserve": output_reserve,
            "available_input_units": available,
            "policy": "rank-order greedy; whole evidence; no claim of globally optimal selection",
        },
    }
