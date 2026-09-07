"""把四组件串起来：预处理、文件状态、分层规划、缓存成本边界。"""

from .governance import Governor
from .preprocess import preprocess
from .projection import prompt, required_view
from .types import count, BudgetError, ViewState, StalePlan


class Runtime:
    def __init__(self, store, session, capacity=4000, threshold=0.8, graph=None):
        if capacity < 1 or not 0 < threshold <= 1:
            raise ValueError("invalid runtime capacity")
        self.store, self.session, self.capacity, self.threshold = store, session, capacity, threshold
        self.governor = Governor(store, graph=graph)
        self.pending_future = None
        self.mode = "in_place"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.governor.close()

    def tick(self, *, event="boundary", model=None, future_calls=1, expired=False):
        status = self.start_gc(event=event, model=model)
        if status["status"] == "pending":
            return self.resume_pending(future_calls=future_calls, expired=expired)
        if status["status"] not in {"started", "planning"}:
            return status
        from concurrent.futures import wait

        wait([self.pending_future])  # Blocking convenience; poll_gc remains nonblocking.
        return self.poll_gc(future_calls=future_calls, expired=expired)

    def start_gc(self, *, event="boundary", model=None):
        """Nonblocking host entry point. Call poll_gc at later safe boundaries."""
        if self.pending_future is not None:
            return {"status": "planning"}
        if self.store.pending(self.session) is not None:
            return {"status": "pending", "next": "resume_pending"}
        snap = self.store.snapshot(self.session)
        if snap.busy:
            return {"status": "busy"}
        if count(prompt(snap)) < self.capacity * self.threshold and event not in {
            "tool_burst",
            "phase_end",
            "idle",
            "model",
        }:
            return {"status": "below_threshold"}
        self.pending_future = self.governor.fork(self.session, model)
        return {"status": "started", "revision": snap.revision}

    def poll_gc(self, *, future_calls=1, expired=False):
        if self.pending_future is None:
            return self.resume_pending(future_calls=future_calls, expired=expired)
        if not self.pending_future.done():
            return {"status": "planning"}
        future, self.pending_future = self.pending_future, None
        try:
            self.governor.stage(future.result())
            return self.resume_pending(future_calls=future_calls, expired=expired)
        except StalePlan as error:
            self.store.log(self.session, "stale_gc", {"reason": str(error)})
            return {"status": "stale", "reason": str(error)}

    def resume_pending(self, *, future_calls=1, expired=False):
        """Reevaluate a persisted plan without paying for a second planner call."""
        plan = self.store.pending(self.session)
        if plan is None:
            return {"status": "no_job"}
        if self.store.snapshot(self.session).busy:
            return {"status": "busy"}
        try:
            result = self.governor.commit(
                self.session, future_calls=future_calls, expired=expired, hard_limit=self.capacity
            )
            return {"status": "committed" if result["committed"] else "pending", **result}
        except StalePlan as error:
            self.store.discard_pending(plan, reason=str(error))
            return {"status": "stale", "reason": str(error)}

    def compress_context(self, *, start_turn, end_turn, future_calls=1, expired=False):
        if start_turn > end_turn:
            raise ValueError("invalid turn range")
        snap = self.store.snapshot(self.session)
        eligible = {o.id for o in snap.objects if start_turn <= o.turn <= end_turn}
        plan = self.governor.plan(snap)
        plan.actions = [a for a in plan.actions if a.target in eligible]
        plan.planner = "model-triggered/rule-planned"
        self.governor.stage(plan)
        return self.governor.commit(
            self.session, future_calls=future_calls, expired=expired, hard_limit=self.capacity
        )

    def context_for(self, query, retriever=None, *, available=True, budget=300):
        """Explicit availability signal or operational retrieval failure triggers fallback."""
        old = self.mode
        try:
            if not available or retriever is None:
                raise ConnectionError("external retrieval unavailable")
            result = self.retrieve_first(query, retriever, budget)
            self.mode = "retrieve_first"
        except (ConnectionError, TimeoutError, FileNotFoundError) as error:
            snap = self.store.snapshot(self.session)
            text = prompt(
                snap, required_view(snap, extra_required=self.governor.live_ids(snap)), dynamic=query
            )
            if count(text) > self.capacity:
                raise BudgetError("in-place fallback exceeds capacity; govern before retry") from error
            result = {"prompt": text, "units": count(text), "fallback_reason": str(error)}
            self.mode = "in_place"
        if old != self.mode:
            self.store.log(self.session, "mode_switch", {"from": old, "to": self.mode})
        return {"mode": self.mode, **result}

    def first_projection(self, limit=600):
        return preprocess(self.store, self.session, limit)

    def retrieve_first(self, query, retriever, budget=300, *, output_reserve=0):
        snap = self.store.snapshot(self.session)
        if hasattr(retriever, "validate_snapshot"):
            retriever.validate_snapshot(snap)
        result = retriever.retrieve(query, budget=budget)
        rows = result.get("evidence", result.get("memories", []))
        # hard cap 涵盖固定上下文、来源标记、当前问题与证据，不只算 top-K。
        if self.store.snapshot(self.session).revision != snap.revision:
            raise StalePlan("session changed during retrieval; rebuild the request")
        from .projection import dependency_closure

        extra = retriever.live_objects(dependency_closure(snap)) if hasattr(retriever, "live_objects") else ()
        states = required_view(snap, hide_unrelated=True, extra_required=extra)
        from .evidence import pack_evidence

        packed = pack_evidence(
            snap,
            states,
            query,
            rows,
            capacity=self.capacity,
            body_budget=budget,
            output_reserve=output_reserve,
        )
        if self.store.snapshot(self.session).revision != snap.revision:
            raise StalePlan("session changed while packing evidence")
        return {**packed, "retrieval": result}

    def recover_into_view(self, object_id):
        text = self.store.recover(self.session, object_id)
        snap = self.store.snapshot(self.session)
        states = dict(snap.states)
        states[object_id] = ViewState("active")
        if count(prompt(snap, states)) > self.capacity:
            raise BudgetError("recovery exceeds capacity; read selected lines or increase budget")
        self.store.commit(snap, states, event={"recover_into_view": object_id})
        return text
