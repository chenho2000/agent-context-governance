"""原始对象不可变；编辑只改变 ViewState。所有 ID 都在会话内验证。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def units(text: str) -> list[str]:
    """保留空白的确定性教学分词；不是任何供应商的真实 tokenizer。"""
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\S\n]+|\n|.", text)


def count(text: str) -> int:
    return len(units(text))


@dataclass(frozen=True)
class ContextObject:
    id: str
    session: str
    seq: int
    turn: int
    kind: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ContextObject:
        return cls(**{**data, "dependencies": tuple(data.get("dependencies", ()))})


@dataclass(frozen=True)
class ViewState:
    mode: str = "active"  # active, cleaned, alias, folded, masked, hidden, distilled
    display: str | None = None
    source_id: str | None = None
    purge_input: bool = False


@dataclass(frozen=True)
class Snapshot:
    session: str
    revision: int
    objects: tuple[ContextObject, ...]
    states: dict[str, ViewState]
    busy: int = 0

    @property
    def latest_turn(self) -> int:
        return max((obj.turn for obj in self.objects), default=0)

    def by_id(self) -> dict[str, ContextObject]:
        return {obj.id: obj for obj in self.objects}


@dataclass(frozen=True)
class Action:
    target: str  # object ID, or turn:N for whole-turn fold
    action: str
    reason: str = ""
    summary: str | None = None


@dataclass
class Plan:
    session: str
    revision: int
    actions: list[Action]
    planner: str
    trace: list[dict] = field(default_factory=list)
    planner_cost: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Plan:
        return cls(**{**data, "actions": [Action(**x) for x in data["actions"]]})


class GovernanceError(ValueError):
    pass


class StalePlan(GovernanceError):
    pass


class BudgetError(GovernanceError):
    pass
