"""审计样本导出。候选数据供学习，绝不自动训练/提升权限。"""

from collections import Counter
from pathlib import Path
from .types import canonical, digest


def evaluate_trial(prefix, action, future, visible, recoverable=()):
    """planner 只能看到 prefix；future 只交给裁判。任务成功使用明确必需证据集合。"""
    required = set(future["required_ids"])
    available = set(visible) | set(recoverable)
    missed = sorted(required - available)
    return {
        "state": digest(canonical(prefix)),
        "prefix": prefix,
        "action": action,
        "success": not missed,
        "missing": missed,
        "verified": True,
        "future": future,
        "cost": len(visible) + 2 * len(recoverable),
    }


def export(trials, directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    sft, dpo, skipped = [], [], []
    for t in trials:
        if t.get("verified") and t.get("success"):
            sft.append(
                {
                    "input": t["prefix"],
                    "output": t["action"],
                    "state": t["state"],
                    "label": "synthetic-evidence-check-only",
                }
            )
    for i, a in enumerate(trials):
        for b in trials[i + 1 :]:
            if a["state"] != b["state"] or a["future"] != b["future"]:
                skipped.append("incomparable state or evaluation target")
                continue
            if not a.get("verified") or not b.get("verified"):
                continue

            def rank(t):
                return t["success"], -t["cost"]

            if rank(a) == rank(b) or a["action"] == b["action"]:
                continue
            good, bad = (a, b) if rank(a) > rank(b) else (b, a)
            dpo.append(
                {
                    "input": good["prefix"],
                    "chosen": good["action"],
                    "rejected": bad["action"],
                    "label": "synthetic-preference-candidate",
                }
            )
    for name, rows in [("sft", sft), ("dpo", dpo)]:
        (path / (name + ".jsonl")).write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")
    failures = dict(Counter(mid for t in trials for mid in t.get("missing", [])))
    result = {"sft": len(sft), "dpo": len(dpo), "skipped": skipped, "failure_counts": failures}
    (path / "audit.json").write_text(canonical(result), encoding="utf-8")
    return result


def clean_session(store, session, *, verified_result, required_ids, directory):
    """Benchmark + dependency-closed golden trace + reviewed skill candidate. No automatic training."""
    snap = store.snapshot(session)
    by_id = snap.by_id()
    if not verified_result or not set(required_ids) <= set(by_id):
        raise ValueError("need verified result and known evidence IDs")
    # Only explicit dead/retry metadata may be cleaned. Side effects are always kept for audit.
    roots = {o.id for o in snap.objects if o.kind in {"user", "instruction"} or o.metadata.get("side_effect")}
    roots |= set(required_ids)
    roots |= {o.id for o in snap.objects if not (o.metadata.get("obsolete") or o.metadata.get("retry"))}
    stack, keep = list(roots), set()
    while stack:
        oid = stack.pop()
        if oid not in keep:
            keep.add(oid)
            stack.extend(by_id[oid].dependencies)
    trace = [o.to_dict() for o in snap.objects if o.id in keep]
    result = {
        "goal": [o.content for o in snap.objects if o.kind == "user"],
        "result": verified_result,
        "required_ids": sorted(required_ids),
        "label": "caller-verified-benchmark-candidate",
    }
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    (target / "benchmark.json").write_text(canonical(result), encoding="utf-8")
    (target / "golden_trace.jsonl").write_text("".join(canonical(o) + "\n" for o in trace), encoding="utf-8")
    # Do not install as SKILL.md or promote untrusted source text into future system instructions.
    (target / "skill-candidate.md").write_text(
        "# Reviewed skill candidate required\n\n"
        "Derived from "
        + session
        + ". Review before reuse. This file is not an installed skill.\n\n"
        + "\n".join("- Evidence: `" + o["id"] + "`" for o in trace),
        encoding="utf-8",
    )
    return {
        "kept": [o["id"] for o in trace],
        "removed": [o.id for o in snap.objects if o.id not in keep],
        "benchmark": result,
    }
