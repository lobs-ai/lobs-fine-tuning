"""
Canonical trajectory format used between every pipeline stage.

Deliberately not Anthropic's wire format — a flat list of typed turns is
much easier to filter, transform, and re-render into either Anthropic-style
(for Claude inference) or OpenAI/Qwen-style (for student training and vLLM
inference).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class ToolCall:
    """A single tool invocation produced by an assistant turn."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """A single tool result fed back to the model in a tool-result user turn."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class UserTurn:
    """The first user turn (the task prompt) or any later user-injected text."""

    role: Literal["user"] = "user"
    text: str = ""


@dataclass
class AssistantTurn:
    """An assistant response. May contain text, tool calls, or both."""

    role: Literal["assistant"] = "assistant"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolResultTurn:
    """The user-role turn that carries tool results back to the model."""

    role: Literal["tool_result"] = "tool_result"
    results: list[ToolResult] = field(default_factory=list)


Turn = UserTurn | AssistantTurn | ToolResultTurn


@dataclass
class TrajectoryMeta:
    run_id: str
    agent_type: str
    model: str = ""
    succeeded: bool = False
    total_turns: int = 0
    stop_reason: str = ""
    duration_seconds: int = 0
    timestamp: str = ""
    source: Literal["summary", "full_jsonl", "gateway_db", "manual"] = "manual"
    source_path: str = ""
    # Tools that were available to the agent during the run. We re-render
    # this exact set into the system prompt at training time so the student
    # sees the same tool surface the teacher saw.
    tools_offered: list[str] = field(default_factory=list)
    # Optional system prompt the agent was launched with, captured verbatim.
    system_prompt: str = ""
    # Anything else worth preserving (cost, usage, error message, …).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    meta: TrajectoryMeta
    turns: list[Turn] = field(default_factory=list)

    # ── (de)serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": asdict(self.meta),
            "turns": [_turn_to_dict(t) for t in self.turns],
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        meta = TrajectoryMeta(**data["meta"])
        turns = [_turn_from_dict(t) for t in data["turns"]]
        return cls(meta=meta, turns=turns)

    # ── derived properties ───────────────────────────────────────────────────

    @property
    def assistant_turns(self) -> list[AssistantTurn]:
        return [t for t in self.turns if isinstance(t, AssistantTurn)]

    @property
    def tool_calls(self) -> list[ToolCall]:
        out: list[ToolCall] = []
        for t in self.turns:
            if isinstance(t, AssistantTurn):
                out.extend(t.tool_calls)
        return out

    @property
    def tool_names_used(self) -> set[str]:
        return {c.name for c in self.tool_calls}


def _turn_to_dict(t: Turn) -> dict[str, Any]:
    if isinstance(t, UserTurn):
        return {"role": "user", "text": t.text}
    if isinstance(t, AssistantTurn):
        return {
            "role": "assistant",
            "text": t.text,
            "tool_calls": [asdict(c) for c in t.tool_calls],
        }
    if isinstance(t, ToolResultTurn):
        return {"role": "tool_result", "results": [asdict(r) for r in t.results]}
    raise TypeError(f"unknown turn type: {type(t)}")


def _turn_from_dict(d: dict[str, Any]) -> Turn:
    role = d["role"]
    if role == "user":
        return UserTurn(text=d.get("text", ""))
    if role == "assistant":
        return AssistantTurn(
            text=d.get("text", ""),
            tool_calls=[ToolCall(**c) for c in d.get("tool_calls", [])],
        )
    if role == "tool_result":
        return ToolResultTurn(results=[ToolResult(**r) for r in d.get("results", [])])
    raise ValueError(f"unknown role: {role!r}")


# ── file IO helpers ──────────────────────────────────────────────────────────


def write_jsonl(path: Path, trajectories: list[Trajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for traj in trajectories:
            f.write(traj.to_json_line() + "\n")


def read_jsonl(path: Path) -> list[Trajectory]:
    out: list[Trajectory] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Trajectory.from_dict(json.loads(line)))
    return out
