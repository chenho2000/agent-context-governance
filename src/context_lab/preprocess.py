"""context-diet/cctx/DCP 思路的可恢复本地预处理。"""

import re
from .types import ViewState, count, digest
from .projection import protected, sparse_evidence


def clean(text):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    # 不去重不相邻行，不截掉 diff/表格/错误栈。
    lines, out = text.splitlines(keepends=True), []
    for line in lines:
        if out and line == out[-1] and line.startswith(("INFO ", "DEBUG ")):
            continue
        out.append(line)
    return "".join(out)


def preprocess(store, session, limit=600):
    snap = store.snapshot(session)
    states, trace = dict(snap.states), []
    tracker = FileStateTracker()
    for obj in snap.objects:
        previous = tracker.observe(obj, snap)
        if states[obj.id].mode != "active":
            continue
        m = obj.metadata
        if not protected(obj, snap):
            if previous:
                states[obj.id] = ViewState("alias", source_id=previous)
            elif count(obj.content) > limit and not sparse_evidence(obj.content):
                store.persist_sidecar(obj)
                states[obj.id] = ViewState("folded")
            elif not sparse_evidence(obj.content):
                text = clean_terminal(obj.content) if m.get("format") == "terminal" else clean(obj.content)
                if text != obj.content:
                    states[obj.id] = ViewState("cleaned", text)
            if states[obj.id].mode != "active":
                trace.append({"id": obj.id, "mode": states[obj.id].mode})
    store.commit(snap, states, event={"preprocess": trace})
    return trace


class FileStateTracker:
    """Session-local exact read identity; caller supplies versions, never infers by path alone."""

    def __init__(self):
        self.session = None
        self.reads = {}
        self._snapshot = None
        self._objects = {}

    def observe(self, obj, snap):
        if self.session != snap.session:
            self.session, self.reads = snap.session, {}
        if self._snapshot is not snap:
            self._snapshot, self._objects = snap, snap.by_id()
        m = obj.metadata
        if obj.session != snap.session or obj.id not in self._objects:
            raise ValueError("foreign file observation")
        if not all(m.get(k) is not None for k in ("path", "version", "range")):
            return None
        key = (m["path"], m["version"], str(m["range"]), digest(obj.content))
        previous = self.reads.get(key)
        if previous == obj.id:
            return None
        if previous and previous in snap.states and snap.states[previous].mode != "hidden":
            return previous
        self.reads[key] = obj.id
        return None


def clean_terminal(text):
    """Opt-in terminal cleanup; never apply trailing-space stripping to code or sparse evidence."""
    if sparse_evidence(text):
        return text
    text = re.sub(r"(?m)^[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]\s*[^\n]*\r", "", text)
    text = clean(text)
    text = re.sub(r"[ \t]+(?=\r?$)", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text)
