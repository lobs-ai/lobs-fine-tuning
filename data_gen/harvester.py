"""
Harvest squad transcripts from disk → canonical Trajectory records.

Reads from any of:
- `~/.lobs/agents/{agentType}/sessions/{runId}.jsonl` — squad's per-agent
  session log. Currently summary-only, but if/when squad starts calling
  `transcript.writeTurn()` per turn, this harvester will pick that up
  automatically (we detect on `type` field).
- `~/.lobs/agents/{agentType}/sessions/{runId}.md` — markdown sidecar; we
  read it only as a fallback summary.
- `data/trajectories_raw/full/*.jsonl` — full Anthropic-format
  message dumps written by `data_gen/run_trajectories.py` (one trajectory
  per file, one JSON object per file written as a single line).
- `data/trajectories_raw/manual/*.json` — manually exported sessions.
- Squad gateway DB (`~/.lobs/lobs.db`) — when populated, joins
  `chat_messages` + `tool_calls` per session.

Filters at harvest time:
- only keep trajectories that used at least one tool from `ALLOWED_TOOLS`,
- drop trajectories whose entire tool surface is outside `ALLOWED_TOOLS`
  (we don't want to teach the student tools we vendored out),
- if `--strict-tools` is set, drop any trajectory that touched ANY tool
  not in `ALLOWED_TOOLS` (default: just rewrite those calls into a generic
  "unsupported_tool" placeholder so the assistant turn is still useful).

Usage:
    python -m data_gen.harvester --in ~/.lobs/agents --out data/trajectories_canonical/all.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from data_gen.squad_config import load_squad_config
from data_gen.trajectory import (
    AssistantTurn,
    ToolCall,
    ToolResult,
    ToolResultTurn,
    Trajectory,
    TrajectoryMeta,
    UserTurn,
    write_jsonl,
)
from tools.schemas import ALLOWED_TOOLS


# ── stats accumulator ────────────────────────────────────────────────────────


@dataclass
class HarvestStats:
    files_seen: int = 0
    summary_only_skipped: int = 0
    parse_errors: int = 0
    no_allowed_tools_skipped: int = 0
    strict_filter_dropped: int = 0
    kept: int = 0

    def report(self) -> str:
        return (
            f"files_seen={self.files_seen} kept={self.kept} "
            f"summary_only_skipped={self.summary_only_skipped} "
            f"no_allowed_tools_skipped={self.no_allowed_tools_skipped} "
            f"strict_filter_dropped={self.strict_filter_dropped} "
            f"parse_errors={self.parse_errors}"
        )


# ── source: per-agent session JSONL ──────────────────────────────────────────


def _load_session_jsonl(path: Path) -> Trajectory | None:
    """
    Parse a squad session JSONL.

    There are two layouts in the wild:
    1. summary-only — a single line `{"type":"summary",...}`. This is what
       the live agent loop writes today (as of squad commit 7daf6dfd).
    2. per-turn + summary — N TurnRecord lines followed by one summary line.
       This is what `SessionTranscript.writeTurn()` is *designed* to write,
       but the runner doesn't call it. If/when it does, we pick it up here.

    Even the summary-only case is worth keeping as a metadata stub — it
    shows what runs happened and which succeeded — but it has no tool calls
    so it'll be dropped by the no_allowed_tools filter.
    """
    try:
        with path.open() as f:
            lines = [json.loads(line) for line in f if line.strip()]
    except (json.JSONDecodeError, OSError):
        return None

    if not lines:
        return None

    summary = next((line for line in lines if line.get("type") == "summary"), None)
    turn_records = [line for line in lines if line.get("type") != "summary"]

    if summary is None:
        return None

    meta = TrajectoryMeta(
        run_id=summary.get("runId", path.stem),
        agent_type=summary.get("agentType", path.parent.parent.name),
        succeeded=bool(summary.get("succeeded")),
        total_turns=int(summary.get("totalTurns", 0)),
        stop_reason=str(summary.get("stopReason", "")),
        duration_seconds=int(summary.get("durationSeconds", 0)),
        timestamp=str(summary.get("timestamp", "")),
        source="full_jsonl" if turn_records else "summary",
        source_path=str(path),
        extra={
            "totalUsage": summary.get("totalUsage"),
            "error": summary.get("error"),
        },
    )

    # Summary-only path: we have nothing to train on, but we still emit a
    # Trajectory with an empty turn list so downstream stats are honest.
    if not turn_records:
        return Trajectory(meta=meta, turns=[])

    # Per-turn path: each TurnRecord captures `toolCalls` (the calls the
    # assistant made on that turn). We do NOT have the model's text or tool
    # results — those live only in the in-memory Session. So we synthesize
    # a minimal AssistantTurn per record and a ToolResultTurn placeholder.
    # This is best-effort; the FULL trajectory format below is preferred.
    turns: list[Any] = []
    for tr in turn_records:
        tool_calls = [
            ToolCall(
                id=f"{meta.run_id}_t{tr.get('turn', 0)}_{i}",
                name=str(call.get("name", "")),
                arguments=call.get("input", {}) or {},
            )
            for i, call in enumerate(tr.get("toolCalls", []))
        ]
        if tool_calls:
            turns.append(AssistantTurn(text="", tool_calls=tool_calls))
            # Synthetic empty tool-result turn so the trajectory shape is valid.
            turns.append(
                ToolResultTurn(
                    results=[
                        ToolResult(tool_use_id=tc.id, content="", is_error=False)
                        for tc in tool_calls
                    ]
                )
            )
    return Trajectory(meta=meta, turns=turns)


def iter_session_jsonl(root: Path) -> Iterator[Path]:
    for agent_dir in sorted(root.iterdir()):
        sessions = agent_dir / "sessions"
        if not sessions.is_dir():
            continue
        for jsonl in sorted(sessions.glob("*.jsonl")):
            yield jsonl


# ── source: full Anthropic-format trajectory dumps ───────────────────────────


def _load_full_anthropic(path: Path) -> Trajectory | None:
    """
    Load a full trajectory written by `data_gen/run_trajectories.py`.

    Schema (single JSON object, can be on multiple lines):
    {
      "meta": { runId, agentType, model, succeeded, totalTurns, stopReason,
                durationSeconds, timestamp, toolsOffered, systemPrompt },
      "messages": [ AnthropicMessage, ... ]
    }

    AnthropicMessage:
      {"role": "user", "content": "string"}
      {"role": "user", "content": [{"type":"tool_result", "tool_use_id":"…", "content":"…"}, ...]}
      {"role": "assistant", "content": [{"type":"text","text":"…"}, {"type":"tool_use","id":"…","name":"…","input":{...}}, ...]}
    """
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    meta_raw = data.get("meta", {}) or {}
    messages = data.get("messages", []) or []

    meta = TrajectoryMeta(
        run_id=meta_raw.get("runId", path.stem),
        agent_type=meta_raw.get("agentType", "unknown"),
        model=meta_raw.get("model", ""),
        succeeded=bool(meta_raw.get("succeeded", False)),
        total_turns=int(meta_raw.get("totalTurns", 0)),
        stop_reason=str(meta_raw.get("stopReason", "")),
        duration_seconds=int(meta_raw.get("durationSeconds", 0)),
        timestamp=str(meta_raw.get("timestamp", "")),
        source="full_jsonl",
        source_path=str(path),
        tools_offered=list(meta_raw.get("toolsOffered", [])),
        system_prompt=str(meta_raw.get("systemPrompt", "")),
        extra={k: v for k, v in meta_raw.items() if k not in {
            "runId", "agentType", "model", "succeeded", "totalTurns",
            "stopReason", "durationSeconds", "timestamp", "toolsOffered",
            "systemPrompt",
        }},
    )

    turns = _anthropic_messages_to_turns(messages)
    return Trajectory(meta=meta, turns=turns)


def _anthropic_messages_to_turns(messages: list[dict[str, Any]]) -> list[Any]:
    """
    Translate Anthropic's wire format messages into our canonical turns.

    The trickiest case is Anthropic's user-role message that carries
    `tool_result` blocks (sometimes mixed with a text block — squad's
    post-tool reminder). We split that into a ToolResultTurn (and discard
    the reminder text — the model regenerates it at training time).
    """
    turns: list[Any] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                turns.append(UserTurn(text=content))
                continue
            if not isinstance(content, list):
                continue
            tool_results: list[ToolResult] = []
            text_chunks: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_result":
                    raw_content = block.get("content")
                    if isinstance(raw_content, list):
                        # Anthropic supports an array of content blocks here;
                        # collapse to text for our flat format.
                        raw_content = "\n".join(
                            (b.get("text") or "") if isinstance(b, dict) else str(b)
                            for b in raw_content
                        )
                    tool_results.append(
                        ToolResult(
                            tool_use_id=str(block.get("tool_use_id", "")),
                            content=str(raw_content or ""),
                            is_error=bool(block.get("is_error", False)),
                        )
                    )
                elif btype == "text":
                    text_chunks.append(str(block.get("text", "")))
            if tool_results:
                turns.append(ToolResultTurn(results=tool_results))
            elif text_chunks:
                turns.append(UserTurn(text="\n".join(text_chunks)))
            continue

        if role == "assistant":
            if isinstance(content, str):
                turns.append(AssistantTurn(text=content))
                continue
            if not isinstance(content, list):
                continue
            text_chunks = []
            tool_calls: list[ToolCall] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_chunks.append(str(block.get("text", "")))
                elif btype == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=str(block.get("id", "")),
                            name=str(block.get("name", "")),
                            arguments=block.get("input", {}) or {},
                        )
                    )
            turns.append(AssistantTurn(text="\n".join(text_chunks), tool_calls=tool_calls))
            continue

        # ignore system / unknown roles — we capture system prompt via meta
    return turns


# ── source: gateway sqlite DB ────────────────────────────────────────────────


def harvest_gateway_db(db_path: Path) -> list[Trajectory]:
    """
    Pull trajectories from squad's gateway SQLite database.

    Looks for tables `chat_messages` (or `messages`) joined with
    `tool_calls`. The schema is gateway-side and may differ across squad
    versions, so this function is defensive and silently returns [] when
    expected tables are missing or empty.
    """
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Detect which message table this DB uses.
        names = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "chat_messages" in names:
            msg_query = "SELECT * FROM chat_messages ORDER BY session_key, created_at"
            session_col = "session_key"
        elif "messages" in names:
            msg_query = "SELECT * FROM messages ORDER BY session_id, created_at"
            session_col = "session_id"
        else:
            return []

        rows = list(conn.execute(msg_query))
        if not rows:
            return []

        # Group by session.
        by_session: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            by_session.setdefault(str(r[session_col]), []).append(r)

        out: list[Trajectory] = []
        for sid, messages in by_session.items():
            traj = _gateway_session_to_trajectory(sid, messages)
            if traj is not None:
                out.append(traj)
        return out
    finally:
        conn.close()


def _gateway_session_to_trajectory(
    session_id: str, rows: list[sqlite3.Row]
) -> Trajectory | None:
    """
    Each row is a chat message with a JSON `content` blob in Anthropic
    wire format. We feed them through the same converter as the
    full-trajectory dumps.
    """
    messages: list[dict[str, Any]] = []
    for r in rows:
        raw_content = r["content"]
        try:
            content_parsed = json.loads(raw_content) if raw_content else None
        except (TypeError, json.JSONDecodeError):
            content_parsed = raw_content
        messages.append(
            {
                "role": r["role"],
                "content": content_parsed if content_parsed is not None else raw_content,
            }
        )

    meta = TrajectoryMeta(
        run_id=session_id,
        agent_type="gateway",
        succeeded=False,  # unknown — gateway DB doesn't carry this
        total_turns=len([m for m in messages if m["role"] == "assistant"]),
        source="gateway_db",
        source_path=str(rows[0].keys()),
    )
    turns = _anthropic_messages_to_turns(messages)
    if not turns:
        return None
    return Trajectory(meta=meta, turns=turns)


# ── filtering at harvest time ────────────────────────────────────────────────


def _filter_for_allowed_tools(traj: Trajectory, *, strict: bool) -> Trajectory | None:
    """
    Decide whether to keep a trajectory based on which tools it touched.

    - If the trajectory used NO allowed tools at all, drop it: there's
      nothing for the student to learn.
    - If `strict`, drop any trajectory that used ANY tool outside the
      allowed set — we don't want the model imitating tools it can't call.
    - Otherwise (lenient), keep the trajectory but rewrite calls to
      out-of-set tools as a special `_unsupported` marker so the filter
      step can decide what to do.
    """
    used = traj.tool_names_used
    if used and not (used & ALLOWED_TOOLS):
        return None
    if not used:
        # No tool calls at all (e.g., summary-only or the model just
        # answered the user). Keep it — could be a single-turn baseline.
        return traj
    if strict and (used - ALLOWED_TOOLS):
        return None
    if not strict:
        for turn in traj.turns:
            if isinstance(turn, AssistantTurn):
                for tc in turn.tool_calls:
                    if tc.name not in ALLOWED_TOOLS:
                        tc.name = "_unsupported"
    return traj


# ── orchestrator ─────────────────────────────────────────────────────────────


def harvest(
    sessions_dir: Path | None,
    full_dir: Path | None,
    manual_dir: Path | None,
    db_path: Path | None,
    *,
    strict_tools: bool,
) -> tuple[list[Trajectory], HarvestStats]:
    stats = HarvestStats()
    out: list[Trajectory] = []

    def consider(traj: Trajectory | None) -> None:
        if traj is None:
            stats.parse_errors += 1
            return
        stats.files_seen += 1
        if not traj.turns:
            stats.summary_only_skipped += 1
            return
        kept = _filter_for_allowed_tools(traj, strict=strict_tools)
        if kept is None:
            if traj.tool_names_used and not (traj.tool_names_used & ALLOWED_TOOLS):
                stats.no_allowed_tools_skipped += 1
            else:
                stats.strict_filter_dropped += 1
            return
        stats.kept += 1
        out.append(kept)

    if sessions_dir is not None and sessions_dir.exists():
        for path in iter_session_jsonl(sessions_dir):
            consider(_load_session_jsonl(path))

    if full_dir is not None and full_dir.exists():
        for path in sorted(full_dir.glob("*.jsonl")):
            consider(_load_full_anthropic(path))
        for path in sorted(full_dir.glob("*.json")):
            consider(_load_full_anthropic(path))

    if manual_dir is not None and manual_dir.exists():
        for path in sorted(manual_dir.glob("*.json")):
            consider(_load_full_anthropic(path))

    if db_path is not None:
        for traj in harvest_gateway_db(db_path):
            consider(traj)

    return out, stats


# ── CLI ──────────────────────────────────────────────────────────────────────


def _expand(path: str | None) -> Path | None:
    if not path:
        return None
    return Path(os.path.expanduser(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest squad transcripts → canonical trajectories.")
    parser.add_argument(
        "--sessions-dir",
        default=os.environ.get("SQUAD_TRANSCRIPTS_DIR", "~/.lobs/agents"),
        help=(
            "Root containing {agentType}/sessions/*.jsonl per-agent transcripts. "
            "This is separate from SQUAD_HOME; squad's runner hard-codes its own path. "
            "Default: $SQUAD_TRANSCRIPTS_DIR or ~/.lobs/agents."
        ),
    )
    parser.add_argument(
        "--full-dir",
        default="data/trajectories_raw/full",
        help="Directory of full Anthropic-format trajectory dumps written by squad_runner.ts",
    )
    parser.add_argument(
        "--manual-dir",
        default="data/trajectories_raw/manual",
        help="Directory of manually exported sessions",
    )
    parser.add_argument(
        "--db-path",
        default="",
        help=(
            "Path to a squad gateway sqlite DB. If omitted we resolve it from "
            "$SQUAD_HOME/$SQUAD_NAME/squad.db via squad_config."
        ),
    )
    parser.add_argument(
        "--squad-name",
        default=os.environ.get("SQUAD_NAME", "default"),
        help="Which squad gateway to read the DB for (default: $SQUAD_NAME or 'default')",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for canonical trajectories JSONL",
    )
    parser.add_argument(
        "--strict-tools",
        action="store_true",
        help="Drop any trajectory that used a tool outside the allowed set",
    )
    args = parser.parse_args()

    db_path: Path | None
    if args.db_path:
        db_path = _expand(args.db_path)
    else:
        sq = load_squad_config()
        candidate = sq.gateway_db(args.squad_name)
        db_path = candidate if candidate.exists() else None

    trajectories, stats = harvest(
        sessions_dir=_expand(args.sessions_dir),
        full_dir=_expand(args.full_dir),
        manual_dir=_expand(args.manual_dir),
        db_path=db_path,
        strict_tools=args.strict_tools,
    )

    out_path = Path(args.out)
    write_jsonl(out_path, trajectories)
    print(stats.report(), file=sys.stderr)
    print(f"wrote {len(trajectories)} trajectories → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
