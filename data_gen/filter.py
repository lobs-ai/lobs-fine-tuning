"""
Phase 3 — apply hard filters to canonical trajectories.

Per SPEC.md §8 hard filters (drop trajectory entirely):
- did not terminate cleanly (no final assistant text turn after tool calls)
- any malformed tool call (unknown tool, missing required field, etc.)
- > 15 turns total
- immediate same-tool-same-args repeat (loop)

Soft (LLM-judge) filters are deferred per DECISIONS.md §"Soft filters: deferred".
Add them in `filter_soft.py` if hard-filter pass rate is too high.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_gen.trajectory import (
    AssistantTurn,
    ToolResultTurn,
    Trajectory,
    UserTurn,
    read_jsonl,
    write_jsonl,
)
from tools.schemas import ALLOWED_TOOLS, TOOL_SCHEMAS

MAX_TURNS = 15  # spec §8


@dataclass
class FilterStats:
    seen: int = 0
    no_terminal_text: int = 0
    too_many_turns: int = 0
    malformed_tool_call: int = 0
    immediate_loop: int = 0
    no_assistant_turns: int = 0
    kept: int = 0
    drop_reasons: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.drop_reasons is None:
            self.drop_reasons = Counter()

    def report(self) -> str:
        return (
            f"seen={self.seen} kept={self.kept} "
            f"no_terminal_text={self.no_terminal_text} "
            f"too_many_turns={self.too_many_turns} "
            f"malformed_tool_call={self.malformed_tool_call} "
            f"immediate_loop={self.immediate_loop} "
            f"no_assistant_turns={self.no_assistant_turns}"
        )


def _missing_required_fields(name: str, args: dict[str, Any]) -> list[str]:
    schema = TOOL_SCHEMAS.get(name)
    if schema is None:
        return []  # unknown tool is caught separately
    required = schema.get("input_schema", {}).get("required", []) or []
    return [r for r in required if r not in args]


def _has_terminal_text(traj: Trajectory) -> bool:
    """
    Spec §8: trajectory must terminate via the agent's final answer.
    Squad terminates when the assistant produces a text-only turn (no
    tool_use). So: the last AssistantTurn must have non-empty text and
    zero tool calls.
    """
    last_assistant: AssistantTurn | None = None
    for turn in traj.turns:
        if isinstance(turn, AssistantTurn):
            last_assistant = turn
    if last_assistant is None:
        return False
    return bool(last_assistant.text.strip()) and not last_assistant.tool_calls


def _is_immediate_loop(traj: Trajectory) -> bool:
    """
    Catch the cheapest form of stuck behavior: same tool, same arguments,
    twice in a row across the entire trajectory (per spec §8).
    """
    prev: tuple[str, str] | None = None
    for turn in traj.turns:
        if not isinstance(turn, AssistantTurn):
            continue
        for tc in turn.tool_calls:
            args_key = json.dumps(tc.arguments, sort_keys=True)
            cur = (tc.name, args_key)
            if prev is not None and cur == prev:
                return True
            prev = cur
    return False


def _malformed_call_reason(traj: Trajectory) -> str | None:
    """
    Return a reason string if any tool call is malformed, else None.
    """
    for turn in traj.turns:
        if not isinstance(turn, AssistantTurn):
            continue
        for tc in turn.tool_calls:
            if tc.name not in ALLOWED_TOOLS:
                return f"unknown_tool:{tc.name}"
            missing = _missing_required_fields(tc.name, tc.arguments or {})
            if missing:
                return f"missing_required:{tc.name}.{missing[0]}"
            if not isinstance(tc.arguments, dict):
                return f"non_dict_args:{tc.name}"
    return None


def _has_orphan_tool_results(traj: Trajectory) -> bool:
    """
    Every ToolResult should reference an immediately-preceding tool_use_id.
    If we find a result with no matching call, the trajectory is broken
    (probably a parsing bug upstream).
    """
    pending_ids: set[str] = set()
    for turn in traj.turns:
        if isinstance(turn, AssistantTurn):
            for tc in turn.tool_calls:
                pending_ids.add(tc.id)
        elif isinstance(turn, ToolResultTurn):
            for r in turn.results:
                if r.tool_use_id and r.tool_use_id not in pending_ids:
                    return True
                pending_ids.discard(r.tool_use_id)
    return False


def _starts_with_user(traj: Trajectory) -> bool:
    return bool(traj.turns) and isinstance(traj.turns[0], UserTurn)


def _count_assistant_turns(traj: Trajectory) -> int:
    return sum(1 for t in traj.turns if isinstance(t, AssistantTurn))


def keep(traj: Trajectory, stats: FilterStats) -> bool:
    stats.seen += 1
    n_assistant = _count_assistant_turns(traj)
    if n_assistant == 0:
        stats.no_assistant_turns += 1
        stats.drop_reasons["no_assistant_turns"] += 1
        return False
    if n_assistant > MAX_TURNS:
        stats.too_many_turns += 1
        stats.drop_reasons["too_many_turns"] += 1
        return False
    reason = _malformed_call_reason(traj)
    if reason is not None:
        stats.malformed_tool_call += 1
        stats.drop_reasons[f"malformed:{reason.split(':', 1)[0]}"] += 1
        return False
    if _has_orphan_tool_results(traj):
        stats.malformed_tool_call += 1
        stats.drop_reasons["orphan_tool_result"] += 1
        return False
    if _is_immediate_loop(traj):
        stats.immediate_loop += 1
        stats.drop_reasons["immediate_loop"] += 1
        return False
    if not _has_terminal_text(traj):
        stats.no_terminal_text += 1
        stats.drop_reasons["no_terminal_text"] += 1
        return False
    if not _starts_with_user(traj):
        # spec doesn't list this explicitly but it's structurally required
        stats.drop_reasons["no_initial_user"] += 1
        return False
    stats.kept += 1
    return True


def main() -> int:
    global MAX_TURNS  # noqa: PLW0603 — single CLI knob; intentional
    parser = argparse.ArgumentParser(description="Phase 3 hard filter on canonical trajectories.")
    parser.add_argument("--in", dest="in_path", required=True, help="canonical trajectories JSONL")
    parser.add_argument("--out", dest="out_path", required=True, help="filtered output JSONL")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help=f"drop trajectories with more than this many assistant turns (default {MAX_TURNS})",
    )
    args = parser.parse_args()
    MAX_TURNS = args.max_turns

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    trajectories = read_jsonl(in_path)
    stats = FilterStats()
    kept = [t for t in trajectories if keep(t, stats)]

    write_jsonl(out_path, kept)
    print(stats.report(), file=sys.stderr)
    print("drop_reasons:", dict(stats.drop_reasons), file=sys.stderr)
    print(f"wrote {len(kept)} → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
