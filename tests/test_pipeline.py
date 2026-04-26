"""
End-to-end smoke test: build a synthetic trajectory, run it through the
harvester filter + format helpers, confirm everything wires up.

Skips the tokenizer-dependent steps when transformers isn't installed so
this can also run in lightweight envs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_gen.filter import _is_immediate_loop, _malformed_call_reason, keep, FilterStats
from data_gen.harvester import (
    _anthropic_messages_to_turns,
    _filter_for_allowed_tools,
    _load_full_anthropic,
    _load_session_jsonl,
)
from data_gen.secret_scrub import REDACTED_TOKEN, scrub_all, scrub_text, ScrubReport
from data_gen.trajectory import (
    AssistantTurn,
    ToolCall,
    ToolResult,
    ToolResultTurn,
    Trajectory,
    TrajectoryMeta,
    UserTurn,
    read_jsonl,
    write_jsonl,
)


def make_traj(*, with_loop: bool = False, malformed: bool = False) -> Trajectory:
    """Build a small but plausible squad trajectory in canonical form."""
    meta = TrajectoryMeta(
        run_id="test1",
        agent_type="squad",
        model="anthropic/claude-sonnet-4-5",
        succeeded=True,
        total_turns=2,
        stop_reason="end_turn",
        source="manual",
        tools_offered=["read", "edit", "exec"],
        extra={"difficulty": "easy"},
    )
    turns: list = [UserTurn(text="Add a docstring to the foo function in src/foo.py")]
    turns.append(
        AssistantTurn(
            text="I'll read the file first.",
            tool_calls=[
                ToolCall(id="t1", name="read", arguments={"file_path": "/tmp/foo.py"}),
            ],
        )
    )
    turns.append(
        ToolResultTurn(
            results=[ToolResult(tool_use_id="t1", content="def foo():\n    return 42\n")]
        )
    )
    second_call = ToolCall(
        id="t2",
        name="edit",
        arguments={
            "file_path": "/tmp/foo.py",
            "old_string": "def foo():",
            "new_string": 'def foo():\n    """Return the answer."""',
        },
    )
    if with_loop:
        # immediate loop: same tool, same args, twice in a row
        second_call_loop = ToolCall(id="t3", name="read", arguments={"file_path": "/tmp/foo.py"})
        turns[1].tool_calls.append(second_call_loop)
    if malformed:
        second_call.arguments = {}  # missing required `old_string`/`new_string` is in `edits` schema; use create_task instead
        second_call.name = "create_task"  # missing required subject/description
    turns.append(AssistantTurn(text="Done.", tool_calls=[second_call]))
    turns.append(ToolResultTurn(results=[ToolResult(tool_use_id="t2", content="ok")]))
    turns.append(AssistantTurn(text="The docstring has been added."))
    return Trajectory(meta=meta, turns=turns)


def test_canonical_round_trip(tmp_path: Path) -> None:
    traj = make_traj()
    out = tmp_path / "x.jsonl"
    write_jsonl(out, [traj])
    loaded = read_jsonl(out)
    assert len(loaded) == 1
    assert loaded[0].meta.run_id == "test1"
    assert len(loaded[0].turns) == len(traj.turns)
    assert loaded[0].assistant_turns[0].tool_calls[0].name == "read"


def test_anthropic_to_canonical() -> None:
    msgs = [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling read"},
                {"type": "tool_use", "id": "t1", "name": "read", "input": {"file_path": "/x"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file contents"},
                {"type": "text", "text": "post-tool reminder we discard"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    turns = _anthropic_messages_to_turns(msgs)
    assert len(turns) == 4
    assert isinstance(turns[0], UserTurn) and turns[0].text == "do the thing"
    assert isinstance(turns[1], AssistantTurn)
    assert turns[1].tool_calls[0].name == "read"
    assert isinstance(turns[2], ToolResultTurn)
    assert turns[2].results[0].content == "file contents"
    assert isinstance(turns[3], AssistantTurn) and turns[3].text == "done"


def test_filter_allowed_tools_drops_alien_only_traj() -> None:
    traj = make_traj()
    for at in [t for t in traj.turns if isinstance(t, AssistantTurn)]:
        for tc in at.tool_calls:
            tc.name = "spawn_subagent"  # not in our allowed set
    kept = _filter_for_allowed_tools(traj, strict=False)
    assert kept is None


def test_filter_allowed_tools_strict_drops_mixed_traj() -> None:
    traj = make_traj()
    # add an out-of-set call
    traj.turns[1].tool_calls.append(
        ToolCall(id="x", name="spawn_subagent", arguments={"task": "..."})
    )
    assert _filter_for_allowed_tools(traj, strict=True) is None
    # lenient mode renames it instead
    traj2 = make_traj()
    traj2.turns[1].tool_calls.append(
        ToolCall(id="x", name="web_search", arguments={"query": "?"})
    )
    out = _filter_for_allowed_tools(traj2, strict=False)
    assert out is not None
    names = [tc.name for at in out.assistant_turns for tc in at.tool_calls]
    assert "_unsupported" in names


def test_phase3_filter_keeps_clean_trajectory() -> None:
    traj = make_traj()
    stats = FilterStats()
    assert keep(traj, stats) is True
    assert stats.kept == 1


def test_phase3_filter_catches_immediate_loop() -> None:
    traj = make_traj(with_loop=True)
    assert _is_immediate_loop(traj) is True
    stats = FilterStats()
    assert keep(traj, stats) is False
    assert stats.immediate_loop == 1


def test_phase3_filter_catches_missing_required_field() -> None:
    traj = make_traj(malformed=True)
    reason = _malformed_call_reason(traj)
    assert reason is not None and reason.startswith("missing_required:create_task")


def test_phase3_filter_requires_terminal_text() -> None:
    traj = make_traj()
    # remove the final answer turn → trajectory ends mid-tool-loop
    traj.turns = traj.turns[:-1]
    stats = FilterStats()
    assert keep(traj, stats) is False
    assert stats.no_terminal_text == 1


def test_load_summary_only_jsonl(tmp_path: Path) -> None:
    """The on-disk shape we actually have today: a single summary line."""
    p = tmp_path / "agents/squad/sessions/abc123.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "type": "summary",
                "runId": "abc123",
                "agentType": "squad",
                "succeeded": True,
                "totalTurns": 0,
                "totalUsage": {"inputTokens": 1, "outputTokens": 1},
                "durationSeconds": 0,
                "stopReason": "end_turn",
                "timestamp": "2026-04-25T00:00:00Z",
            }
        )
        + "\n"
    )
    traj = _load_session_jsonl(p)
    assert traj is not None
    assert traj.meta.source == "summary"
    assert traj.turns == []  # nothing trainable, but parsed cleanly


def test_load_full_anthropic_dump(tmp_path: Path) -> None:
    msgs = [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "ls", "input": {"path": "/tmp"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "a\nb\n"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Found two files."}]},
    ]
    p = tmp_path / "abc.json"
    p.write_text(
        json.dumps(
            {
                "meta": {
                    "runId": "abc",
                    "agentType": "finetune-data",
                    "model": "anthropic/claude-sonnet-4-5",
                    "succeeded": True,
                    "totalTurns": 2,
                    "stopReason": "end_turn",
                    "toolsOffered": ["ls"],
                },
                "messages": msgs,
            }
        )
    )
    traj = _load_full_anthropic(p)
    assert traj is not None
    assert traj.meta.source == "full_jsonl"
    assert len(traj.assistant_turns) == 2
    assert traj.assistant_turns[0].tool_calls[0].name == "ls"


# ── secret scrubbing ────────────────────────────────────────────────────────


def test_scrub_anthropic_key() -> None:
    report = ScrubReport()
    out = scrub_text("here is my key sk-ant-abcdef0123456789ABCDEF and that's it", set(), report)
    assert "sk-ant-" not in out
    assert REDACTED_TOKEN.format(reason="anthropic_key") in out
    assert report.hits["anthropic_key"] == 1


def test_scrub_openai_key() -> None:
    report = ScrubReport()
    out = scrub_text("OPENAI_API_KEY=sk-proj-AbCdEf0123456789xxxxxxxxxxxx", set(), report)
    assert "sk-proj-" not in out
    assert "anthropic_key" not in report.hits


def test_scrub_pem_block() -> None:
    pem = (
        "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSi\n-----END PRIVATE KEY-----"
    )
    report = ScrubReport()
    out = scrub_text(f"key follows: {pem} done", set(), report)
    assert "MIIE" not in out
    assert report.hits["pem_private_key"] == 1


def test_scrub_live_secret_literal() -> None:
    report = ScrubReport()
    secret = "f3QkUMPrwlX_CHH6-8gLW2TBRRxmthZXQ96lqs0rLeA"
    out = scrub_text(f"got token={secret}", {secret}, report)
    assert secret not in out
    assert report.hits["live_secret"] == 1


def test_scrub_placeholder_unchanged() -> None:
    report = ScrubReport()
    out = scrub_text("<REDACTED:anthropic_key>", set(), report)
    assert out == "<REDACTED:anthropic_key>"
    assert sum(report.hits.values()) == 0


def test_scrub_broad_token_only_with_context() -> None:
    """A long URL-safe-base64 string is only redacted when the surrounding
    context contains the word 'token' / 'key' / similar — otherwise we'd
    shred random hashes."""
    report = ScrubReport()
    # No context — should pass through.
    benign = "abcdef0123456789ABCDEFabcdef0123456789ABCDEF"  # 44 chars
    out = scrub_text(f"git commit {benign}", set(), report)
    assert benign in out
    # With context — should be caught.
    report2 = ScrubReport()
    out2 = scrub_text(f"my token is {benign}", set(), report2)
    assert benign not in out2


def test_scrub_set_config_redacts_sensitive_path() -> None:
    traj = make_traj()
    traj.turns[1] = AssistantTurn(
        text="setting the key",
        tool_calls=[
            ToolCall(
                id="t1",
                name="set_config",
                arguments={"path": "llm.providers.anthropic.apiKey", "value": "sk-ant-xxxxxxxxxxxxxxxxxxxx"},
            )
        ],
    )
    out, report = scrub_all([traj], live_secrets=set())
    assert len(out) == 1
    redacted_args = out[0].assistant_turns[0].tool_calls[0].arguments
    assert redacted_args["value"] == REDACTED_TOKEN.format(reason="sensitive_config_path")
    assert report.hits["sensitive_config_path"] == 1


def test_scrub_drops_trajectory_with_private_key_block() -> None:
    traj = make_traj()
    traj.turns.insert(
        0,
        UserTurn(
            text=(
                "deploy this:\n-----BEGIN OPENSSH PRIVATE KEY-----\n"
                "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB\n"
                "-----END OPENSSH PRIVATE KEY-----"
            )
        ),
    )
    out, report = scrub_all([traj], live_secrets=set())
    assert out == []
    assert report.trajectories_dropped == 1
    assert report.drop_reasons["private_key_in_trajectory"] == 1


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_transformers(), reason="transformers not installed")
def test_format_round_trip() -> None:
    """The most critical test: assistant tokens decode to assistant text.

    Defaults to Qwen2.5-7B-Instruct because it's small to download and has
    the OpenAI-style tools+tool_calls chat template we depend on. Override
    via $TEST_MODEL to verify against any other instruct tokenizer.
    """
    import os
    from transformers import AutoTokenizer
    from data_gen.format_for_training import format_one, verify_example

    name = os.environ.get("TEST_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    traj = make_traj()
    ex = format_one(traj, tok, max_len=8192)
    assert ex is not None
    assert verify_example(ex, tok, traj) is None
