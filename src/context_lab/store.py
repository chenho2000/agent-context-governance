"""SQLite 事件库 + 非破坏性视图 + 可校验 sidecar；提交使用版本 CAS。"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import re
import sqlite3
import tempfile
from typing import Iterator

from .types import ContextObject, Plan, Snapshot, StalePlan, ViewState, canonical, digest


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "context.sqlite3"
        with self.connect() as con:
            con.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0,
                    busy INTEGER NOT NULL DEFAULT 0 CHECK(busy >= 0));
                CREATE TABLE IF NOT EXISTS objects (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(id),
                    seq INTEGER NOT NULL, body TEXT NOT NULL, sha256 TEXT NOT NULL,
                    UNIQUE(session,seq));
                CREATE TRIGGER IF NOT EXISTS immutable_update BEFORE UPDATE ON objects
                    BEGIN SELECT RAISE(ABORT, 'raw objects are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_delete BEFORE DELETE ON objects
                    BEGIN SELECT RAISE(ABORT, 'raw objects are immutable'); END;
                CREATE TABLE IF NOT EXISTS views (
                    object_id TEXT PRIMARY KEY REFERENCES objects(id), body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pending (
                    session TEXT PRIMARY KEY REFERENCES sessions(id), body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL,
                    kind TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS blocks (
                    namespace TEXT NOT NULL, name TEXT NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(namespace,name));
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def create_session(self, session: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session):
            raise ValueError("session 只能包含字母、数字、下划线或短横线")
        with self.connect() as con:
            con.execute("INSERT OR IGNORE INTO sessions(id) VALUES(?)", (session,))

    def append(
        self,
        session: str,
        content: str,
        *,
        turn: int,
        kind: str = "tool",
        metadata: dict | None = None,
        dependencies: tuple[str, ...] = (),
        tool: str = "",
        arguments: dict | None = None,
    ) -> ContextObject:
        if not isinstance(content, str):
            raise ValueError("text runtime requires a string; binary media need a separate artifact store")
        metadata = dict(metadata or {})
        path = str(metadata.get("path", (arguments or {}).get("path", "")))
        if Path(path).name in {"AGENTS.md", "SKILL.md", "CLAUDE.md"}:
            metadata["protected"] = True
        if tool.lower() in {"write", "write_file", "edit", "edit_file", "delete", "delete_file"}:
            metadata["side_effect"] = True
        if kind not in {"user", "assistant", "tool", "instruction"} or turn < 0:
            raise ValueError("invalid object kind or turn")
        if kind == "tool" and not tool:
            raise ValueError("tool objects require a tool name")
        self.create_session(session)
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            seq = con.execute(
                "SELECT COALESCE(MAX(seq),0)+1 FROM objects WHERE session=?", (session,)
            ).fetchone()[0]
            for dep in dependencies:
                if not con.execute(
                    "SELECT 1 FROM objects WHERE id=? AND session=?", (dep, session)
                ).fetchone():
                    raise ValueError(f"unknown or foreign dependency: {dep}")
            obj = ContextObject(
                f"{session}:{kind}:{seq}",
                session,
                seq,
                turn,
                kind,
                content,
                metadata or {},
                dependencies,
                tool,
                arguments or {},
                f"call_{session}_{seq}" if kind == "tool" else "",
            )
            body = canonical(obj.to_dict())
            con.execute("INSERT INTO objects VALUES(?,?,?,?,?)", (obj.id, session, seq, body, digest(body)))
            con.execute("UPDATE sessions SET revision=revision+1 WHERE id=?", (session,))
            con.execute(
                "INSERT INTO events(session,kind,body) VALUES(?,?,?)",
                (session, "append", canonical({"id": obj.id, "turn": turn})),
            )
        return obj

    def snapshot(self, session: str) -> Snapshot:
        with self.connect() as con:
            con.execute("BEGIN")
            row = con.execute("SELECT * FROM sessions WHERE id=?", (session,)).fetchone()
            if row is None:
                raise ValueError(f"unknown session: {session}")
            objects = []
            states = {}
            for data in con.execute(
                "SELECT o.*,v.body AS view_body FROM objects o LEFT JOIN views v "
                "ON o.id=v.object_id WHERE o.session=? ORDER BY o.seq",
                (session,),
            ):
                if digest(data["body"]) != data["sha256"]:
                    raise ValueError("raw object integrity failure")
                obj = ContextObject.from_dict(json.loads(data["body"]))
                objects.append(obj)
                states[obj.id] = (
                    ViewState(**json.loads(data["view_body"])) if data["view_body"] else ViewState()
                )
            return Snapshot(session, row["revision"], tuple(objects), states, row["busy"])

    def log(self, session: str, kind: str, data: dict) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO events(session,kind,body) VALUES(?,?,?)", (session, kind, canonical(data))
            )

    def events(self, session: str) -> list[dict]:
        with self.connect() as con:
            return [
                {"seq": r["seq"], "kind": r["kind"], "data": json.loads(r["body"])}
                for r in con.execute("SELECT * FROM events WHERE session=? ORDER BY seq", (session,))
            ]

    def set_busy(self, session: str, count: int) -> None:
        if count < 0:
            raise ValueError("busy cannot be negative")
        with self.connect() as con:
            cur = con.execute("UPDATE sessions SET busy=?,revision=revision+1 WHERE id=?", (count, session))
            if cur.rowcount != 1:
                raise ValueError("unknown session")

    def sidecar_path(self, obj: ContextObject) -> Path:
        return self.root / "sidecars" / obj.session / (digest(obj.id) + ".txt")

    def persist_sidecar(self, obj: ContextObject) -> str:
        path = self.sidecar_path(obj)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = obj.content.encode("utf-8")
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("sidecar payload collision or corruption")
        else:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
                temp.write(payload)
                temp.flush()
                import os

                os.fsync(temp.fileno())
                temporary = Path(temp.name)
            temporary.replace(path)
        return str(path.relative_to(self.root))

    def recover(self, session: str, object_id: str) -> str:
        snap = self.snapshot(session)
        objects = snap.by_id()
        requested, visited = object_id, set()
        while True:
            if object_id in visited:
                raise ValueError("alias recovery cycle")
            visited.add(object_id)
            obj = objects.get(object_id)
            if obj is None:
                raise ValueError("unknown object in this session")
            state = snap.states[object_id]
            if state.mode != "alias":
                break
            source = objects.get(state.source_id)
            if source is None or source.content != obj.content:
                raise ValueError("alias recovery integrity failure")
            object_id = state.source_id
        if state.mode == "folded":
            payload = self.sidecar_path(obj).read_bytes()
            if payload != obj.content.encode("utf-8"):
                raise ValueError("sidecar integrity failure")
            content = payload.decode("utf-8")
        else:
            content = obj.content  # raw audit recovery is explicit, including hidden objects
        self.log(
            session,
            "recover",
            {"id": requested, "resolved_id": object_id, "mode": state.mode, "sha256": digest(content)},
        )
        return content

    def save_pending(self, plan: Plan) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO pending VALUES(?,?) ON CONFLICT(session) DO UPDATE SET body=excluded.body",
                (plan.session, canonical(plan.to_dict())),
            )
        self.log(plan.session, "pending", plan.to_dict())

    def pending(self, session: str) -> Plan | None:
        with self.connect() as con:
            row = con.execute("SELECT body FROM pending WHERE session=?", (session,)).fetchone()
        return Plan.from_dict(json.loads(row[0])) if row else None

    def commit(
        self,
        snapshot: Snapshot,
        states: dict[str, ViewState],
        *,
        event: dict,
        clear_pending: bool = False,
        require_idle: bool = True,
        expected_pending: Plan | None = None,
    ) -> int:
        """同一事务验证 revision/busy 并切换视图；原始 objects 表从不修改。"""
        from dataclasses import asdict

        ids = set(snapshot.by_id())
        if not set(states) <= ids:
            raise ValueError("foreign view target")
        objects = snapshot.by_id()
        merged = {**snapshot.states, **states}
        for oid, state in merged.items():
            if state.mode not in {"active", "cleaned", "alias", "folded", "masked", "hidden", "distilled"}:
                raise ValueError("invalid view mode")
            if state.mode == "alias":
                source = objects.get(state.source_id)
                obj = objects[oid]
                if source is None or source.seq >= obj.seq or source.content != obj.content:
                    raise ValueError("alias must refer to an identical earlier object")
                if merged[source.id].mode == "hidden":
                    raise ValueError("alias source cannot disappear from the view")
            if state.mode == "folded":
                if self.sidecar_path(objects[oid]).read_bytes() != objects[oid].content.encode("utf-8"):
                    raise ValueError("folded sidecar integrity failure")
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT revision,busy FROM sessions WHERE id=?", (snapshot.session,)).fetchone()
            if row["revision"] != snapshot.revision:
                raise StalePlan("snapshot changed; regenerate/rehearse the plan")
            if expected_pending is not None:
                current = con.execute(
                    "SELECT body FROM pending WHERE session=?", (snapshot.session,)
                ).fetchone()
                if current is None or current[0] != canonical(expected_pending.to_dict()):
                    raise StalePlan("pending plan was replaced; do not commit an older proposal")
            if require_idle and row["busy"]:
                raise ValueError("tools still running; no safe boundary")
            for oid, state in states.items():
                con.execute(
                    "INSERT INTO views VALUES(?,?) ON CONFLICT(object_id) DO UPDATE SET body=excluded.body",
                    (oid, canonical(asdict(state))),
                )
            con.execute("UPDATE sessions SET revision=revision+1 WHERE id=?", (snapshot.session,))
            if clear_pending:
                con.execute("DELETE FROM pending WHERE session=?", (snapshot.session,))
            con.execute(
                "INSERT INTO events(session,kind,body) VALUES(?,?,?)",
                (snapshot.session, "commit", canonical(event)),
            )
        return snapshot.revision + 1

    def put_memory(self, memory: dict) -> None:
        self.put_memories([memory])

    def put_memories(self, memories: list[dict]) -> None:
        """One validated window commits atomically; concurrent source sets merge in the transaction."""
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            for memory in memories:
                row = con.execute("SELECT body FROM memories WHERE id=?", (memory["id"],)).fetchone()
                if row:
                    prior = json.loads(row[0])
                    if prior["session"] != memory["session"]:
                        raise ValueError("foreign memory overwrite")
                    memory = {
                        **memory,
                        "sources": sorted(set(prior.get("sources", [])) | set(memory.get("sources", []))),
                        "supersedes": sorted(
                            set(prior.get("supersedes", [])) | set(memory.get("supersedes", []))
                        ),
                    }
                con.execute(
                    "INSERT INTO memories VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body",
                    (memory["id"], memory["session"], canonical(memory)),
                )
            # A correction cycle would hide every fact in it. Reject the entire window.
            for session in {m["session"] for m in memories}:
                records = {
                    m["id"]: m
                    for row in con.execute("SELECT body FROM memories WHERE session=?", (session,))
                    for m in [json.loads(row[0])]
                }
                indegree = {key: 0 for key in records}
                for record in records.values():
                    for target in record.get("supersedes", []):
                        if target not in records:
                            raise ValueError("unknown correction target")
                        indegree[target] += 1
                stack = [key for key, degree in indegree.items() if degree == 0]
                visited = 0
                while stack:
                    key = stack.pop()
                    visited += 1
                    for target in records[key].get("supersedes", []):
                        indegree[target] -= 1
                        if indegree[target] == 0:
                            stack.append(target)
                if visited != len(records):
                    raise ValueError("correction cycle")

    def discard_pending(self, plan: Plan, *, reason: str) -> bool:
        """Compare the full plan before removing it; never discard a newer replacement."""
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT body FROM pending WHERE session=?", (plan.session,)).fetchone()
            if row is None or row[0] != canonical(plan.to_dict()):
                return False
            con.execute("DELETE FROM pending WHERE session=?", (plan.session,))
            con.execute(
                "INSERT INTO events(session,kind,body) VALUES(?,?,?)",
                (plan.session, "discard_pending", canonical({"reason": reason, "revision": plan.revision})),
            )
        return True

    def memories(self, session: str | None = None) -> list[dict]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT body FROM memories" + (" WHERE session=?" if session else "") + " ORDER BY id",
                (session,) if session else (),
            )
            return [json.loads(row[0]) for row in rows]

    def set_block(self, namespace: str, name: str, value: dict) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO blocks VALUES(?,?,?) ON CONFLICT(namespace,name) DO UPDATE SET body=excluded.body",
                (namespace, name, canonical(value)),
            )
        self.log(namespace, "block_update", {"name": name, "value": value})

    def blocks(self, namespace: str) -> dict[str, dict]:
        with self.connect() as con:
            return {
                row[0]: json.loads(row[1])
                for row in con.execute(
                    "SELECT name,body FROM blocks WHERE namespace=? ORDER BY name", (namespace,)
                )
            }
