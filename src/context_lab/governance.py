"""Self-GC 风格 plan → rehearse → pending → atomic commit。"""

from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree
from .types import Action, Plan, ViewState, GovernanceError, StalePlan, count
from .projection import protected, dependency_closure, sparse_evidence, prompt
from .cache import commit_value
from .planning import GC_PROMPT


def parse_xml(text, snap):
    root = ElementTree.fromstring(text)
    if root.tag != "gc":
        raise GovernanceError("expected <gc>")
    return Plan(
        snap.session,
        snap.revision,
        [Action(e.attrib["target"], e.tag, e.attrib.get("reason", ""), e.text) for e in root],
        "xml",
    )


class Governor:
    def __init__(self, store, graph=None, purge_error_age=3):
        if not isinstance(purge_error_age, int) or purge_error_age < 0:
            raise ValueError("purge_error_age must be a nonnegative integer")
        self.store = store
        self.graph = graph
        self.purge_error_age = purge_error_age
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gc-fork")

    def close(self):
        self.pool.shutdown(wait=True)

    def live_ids(self, snap):
        live = dependency_closure(snap)
        if self.graph is not None:
            self.graph.validate_snapshot(snap)
            live |= self.graph.live_objects(live)
        return live

    def plan(self, snap, model=None):
        live = self.live_ids(snap)
        if model:
            response = model.generate(
                GC_PROMPT,
                {"objects": [o.to_dict() for o in snap.objects], "live_ids": sorted(live)},
            )
            actions = [Action(**a) for a in response["actions"]]
        else:
            actions = []
            for obj in snap.objects:
                if (
                    protected(obj, snap)
                    or obj.kind not in {"user", "tool"}
                    or snap.states[obj.id].mode != "active"
                ):
                    continue
                if obj.kind == "tool" and obj.id not in live and self.purge_error(obj, snap):
                    actions.append(Action(obj.id, "mask", "aged failed call input"))
                    continue
                action = (
                    "prune"
                    if obj.kind == "tool" and obj.metadata.get("obsolete") and obj.id not in live
                    else "mask"
                    if obj.kind == "tool"
                    and obj.metadata.get("redundant_log")
                    and obj.id not in live
                    and not sparse_evidence(obj.content)
                    else "fold"
                )
                if action == "fold" and count(obj.content) < 64:
                    continue  # calibration: a reference can cost more than a short body
                actions.append(Action(obj.id, action, "deterministic teaching planner"))
        return Plan(
            snap.session,
            snap.revision,
            actions,
            "model" if model else "rules",
            trace=[
                {
                    "procedure": "six-step",
                    "protected_ids": [o.id for o in snap.objects if protected(o, snap)],
                    "live_ids": sorted(live),
                    "actions": len(actions),
                }
            ],
        )

    def fork(self, session, model=None):
        # 捕获不可变逻辑快照。主会话可以继续 append，旧计划随后必须被拒绝。
        snap = self.store.snapshot(session)
        return self.pool.submit(self.plan, snap, model)

    def rehearse(self, plan):
        snap = self.store.snapshot(plan.session)
        if snap.revision != plan.revision:
            raise StalePlan("planner snapshot is stale")
        states, live, touched = dict(snap.states), self.live_ids(snap), set()
        for action in plan.actions:
            targets = (
                [o for o in snap.objects if o.turn == int(action.target[5:])]
                if action.target.startswith("turn:")
                else [o for o in snap.objects if o.id == action.target]
            )
            if not targets:
                raise GovernanceError("unknown or foreign target")
            if action.target.startswith("turn:") and action.action != "fold":
                raise GovernanceError("whole-turn action must be fold")
            for obj in targets:
                if obj.id in touched:
                    raise GovernanceError("duplicate action target")
                touched.add(obj.id)
                if action.action == "keep":
                    continue
                if obj.kind == "user" and action.action != "fold":
                    raise GovernanceError("user spans allow only keep/fold")
                if obj.kind not in {"user", "tool"}:
                    raise GovernanceError("assistant/instruction records are not direct GC targets")
                if protected(obj, snap):
                    raise GovernanceError("protected object: " + obj.id)
                if action.action in {"mask", "prune", "distill"} and obj.id in live:
                    raise GovernanceError("live dependency must stay recoverable in view")
                if action.action == "fold":
                    # 归档输出本体；原始调用参数仍在不可变记录中。
                    self.store.persist_sidecar(obj)
                    states[obj.id] = ViewState("folded")
                elif action.action == "mask":
                    if (sparse_evidence(obj.content) and not self.purge_error(obj, snap)) or not (
                        obj.metadata.get("redundant_log") or self.purge_error(obj, snap)
                    ):
                        raise GovernanceError("mask allowed only for explicit redundant logs/error inputs")
                    states[obj.id] = ViewState(
                        "masked",
                        obj.content if self.purge_error(obj, snap) else self.mask_log(obj),
                        purge_input=self.purge_error(obj, snap),
                    )
                elif action.action == "prune":
                    if not obj.metadata.get("obsolete"):
                        raise GovernanceError("prune requires explicit obsolete marker")
                    states[obj.id] = ViewState("hidden")
                elif action.action == "distill":
                    if sparse_evidence(obj.content) or not action.summary:
                        raise GovernanceError("distill requires summary and non-sparse source")
                    # 标签明确是派生文本，不能伪装成原始证据。
                    states[obj.id] = ViewState(
                        "distilled", "[derived; source " + obj.id + "] " + action.summary
                    )
                else:
                    raise GovernanceError("unknown action")
        before, after = prompt(snap), prompt(snap, states)
        return (
            snap,
            states,
            {"before": count(before), "after": count(after), "saved": count(before) - count(after)},
        )

    def purge_error(self, obj, snap):
        return bool(
            obj.metadata.get("purge_input")
            or (obj.metadata.get("error") and snap.latest_turn - obj.turn >= self.purge_error_age)
        )

    @staticmethod
    def mask_log(obj):
        import math

        lines = obj.content.splitlines(keepends=True)
        n = max(1, math.ceil(len(lines) * 0.1))
        if len(lines) <= 2 * n:
            return obj.content
        return "".join(lines[:n]) + f"\n[middle omitted; recover {obj.id}]\n" + "".join(lines[-n:])

    def stage(self, plan):
        _, _, metrics = self.rehearse(plan)
        self.store.save_pending(plan)
        return metrics

    def commit(
        self, session, *, future_calls=1, expired=False, hard_limit=None, prices=None, remaining_cost=0
    ):
        plan = self.store.pending(session)
        if plan is None:
            raise GovernanceError("no pending plan")
        snap, states, metrics = self.rehearse(plan)
        from .types import units

        before, after = units(prompt(snap)), units(prompt(snap, states))
        prefix = 0
        for a, b in zip(before, after):
            if a != b:
                break
            prefix += 1
        value = commit_value(
            len(before) - prefix,
            len(after) - prefix,
            future_calls=future_calls,
            expired=expired,
            prices=prices,
            remaining_cost=remaining_cost,
        )
        metrics.update(common_prefix=prefix, net_value=round(value, 6))
        if hard_limit is not None and metrics["after"] > hard_limit:
            from .types import BudgetError

            raise BudgetError("cannot meet hard limit without protected-data loss")
        forced = hard_limit is not None and metrics["before"] > hard_limit
        if value <= 0 and not forced:
            self.store.log(session, "defer_cache", metrics)
            return {"committed": False, **metrics}
        self.store.commit(
            snap, states, event={"plan": plan.to_dict(), **metrics}, clear_pending=True, expected_pending=plan
        )
        return {"committed": True, **metrics}
