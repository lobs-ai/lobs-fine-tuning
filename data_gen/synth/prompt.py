"""
Build the prompt sent to Gemini so it produces a complete synthetic
trajectory in one shot.

We give Gemini the 12 vendored tool schemas and ask it to *simulate* an
agent solving a task using only those tools — emitting a full Anthropic-
format message log (user → assistant with tool_uses → user with
tool_results → … → final assistant text). The output is a single JSON
object that drops straight into `data/trajectories_raw/full/{run_id}.json`
and is read by the existing harvester (`_load_full_anthropic`).

Why one-shot generation instead of an agent loop with executed tools:
- Trivial: one LLM call per trajectory, no sandboxing, no error handling.
- Tool surface is guaranteed correct because we hand-feed the schemas and
  the model writes both calls and "results" against them.
- Tool results are fictional but plausible — fine for teaching tool-call
  *shape*, which is what spec §9.2 actually trains on. (We mask everything
  but assistant tokens anyway, so the fictional result text is never a
  prediction target.)

Categories steer variety. Add more freely; the prompt asks Gemini to honor
the requested category but otherwise pick a realistic task within it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tools.schemas import schemas_for


CATEGORIES: dict[str, str] = {
    "file_edit": (
        "Make a focused edit to one or two existing files (rename a function, "
        "fix a bug, add a docstring). Should require Read before Edit."
    ),
    "debugging": (
        "Track down why a small program misbehaves. Should use grep/exec/read "
        "to gather evidence, then edit to fix. 4–8 turns is typical."
    ),
    "search": (
        "Answer a question about the codebase (where is X used? which files "
        "import Y?). Mostly grep/ls/read, ending with a textual answer."
    ),
    "multi_file_refactor": (
        "Rename/restructure across 2–4 files. Read every touched file first, "
        "then edit each. Use create_task/update_task to track progress."
    ),
    "config_change": (
        "Adjust the gateway config. Use list_config_paths to discover, "
        "get_config to read, set_config to update. End with a summary."
    ),
    "task_management": (
        "Plan a piece of work by breaking it into create_task entries, then "
        "marking the first one in_progress. No file edits required."
    ),
    "shell": (
        "Run a small command via exec to verify behavior (tests, type-check, "
        "git status). May read a file or two for context. Short trajectory."
    ),
    "mixed": (
        "Realistic small task that spans multiple tool families: e.g. read a "
        "config, change behavior in code, run tests to verify."
    ),
}


SYSTEM_INSTRUCTION = """\
You are generating a synthetic agent trajectory for a fine-tuning dataset.

You will produce a complete worked example: a user task, the agent's full
sequence of reasoning and tool calls, simulated tool results, and a final
text answer. You are SIMULATING the agent and the environment both.

Hard rules:
1. The agent may ONLY use tools from the provided list. Do not invent tools.
   Do not call your own (Gemini-native) tools — output is plain text JSON.
2. Tool inputs must conform to each tool's input_schema (required fields,
   correct types). Read the schemas carefully.
3. Tool results you simulate must be realistic — file contents that look
   like real code, exec output that matches what the command would print,
   reasonable line numbers, etc.
4. The trajectory MUST terminate with an assistant turn that contains text
   only (no tool_use blocks). That is how the agent loop signals "done".
5. Keep trajectories between 2 and 10 assistant turns. Don't pad.
6. Do not repeat the same tool call with identical arguments back-to-back.
7. Read a file before editing it (the edit tool refuses without prior read).
8. Use absolute paths for file_path / path arguments.

Output: a single JSON object, no markdown fences, with this exact shape:

{
  "meta": {
    "runId": "<unique slug like 'synth-fix-login-001'>",
    "agentType": "synth",
    "model": "gemini",
    "succeeded": true,
    "totalTurns": <n_assistant_turns>,
    "stopReason": "end_turn",
    "toolsOffered": ["read", "edit", ...],
    "systemPrompt": "You are a squad agent. Use the available tools to complete the user's task. Read files before editing them, prefer targeted actions over broad sweeps, and stop once the task is done.",
    "category": "<the requested category>"
  },
  "messages": [
    {"role": "user", "content": "<the task prompt the user would type>"},
    {"role": "assistant", "content": [
      {"type": "text", "text": "<short plan/intent>"},
      {"type": "tool_use", "id": "tu_001", "name": "read", "input": {"file_path": "/abs/path"}}
    ]},
    {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "tu_001", "content": "<simulated file contents>"}
    ]},
    ...
    {"role": "assistant", "content": [{"type": "text", "text": "<final answer>"}]}
  ]
}

tool_use_id values must match between assistant tool_use and the corresponding
user tool_result. IDs should be stable strings like "tu_001", "tu_002", ...
"""


@dataclass
class SynthRequest:
    category: str
    seed: int  # used as a "diversity nudge" in the prompt; not a true PRNG seed
    run_id_hint: str

    def to_user_prompt(self) -> str:
        desc = CATEGORIES.get(self.category, CATEGORIES["mixed"])
        return (
            f"Generate ONE trajectory in the '{self.category}' category.\n"
            f"Category description: {desc}\n"
            f"Use 'synth-{self.run_id_hint}' as the runId.\n"
            f"Diversity nudge #{self.seed}: pick a task you have not generated before; "
            f"vary the codebase domain (web app / CLI / data pipeline / library / config)."
        )


def render_system_prompt() -> str:
    """Schema-laden system prompt fed to Gemini."""
    schemas = schemas_for()  # all 12
    schemas_json = json.dumps(schemas, indent=2)
    return (
        SYSTEM_INSTRUCTION
        + "\n\nAvailable tools (full schemas):\n"
        + schemas_json
    )


def render_full_prompt(req: SynthRequest) -> str:
    """
    Single combined prompt for `gemini -p` (which is one-shot, no separate
    system slot exposed in the binary). System instructions go first, then
    the user-style request.
    """
    return render_system_prompt() + "\n\n---\n\n" + req.to_user_prompt()


def expected_meta_keys() -> set[str]:
    return {
        "runId",
        "agentType",
        "model",
        "succeeded",
        "totalTurns",
        "stopReason",
    }


def validate_trajectory_shape(obj: dict[str, Any]) -> str | None:
    """
    Return None if the JSON shape is acceptable, else an error string.
    Doesn't enforce semantic correctness (real tools called with sane args
    is the filter step's job — same one we already use for live data).
    """
    if not isinstance(obj, dict):
        return f"top-level not an object (got {type(obj).__name__})"
    meta = obj.get("meta")
    if not isinstance(meta, dict):
        return "meta missing or not an object"
    missing = expected_meta_keys() - set(meta.keys())
    if missing:
        return f"meta missing keys: {sorted(missing)}"
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return "messages missing or empty"
    if msgs[0].get("role") != "user":
        return "first message is not role=user"
    last = msgs[-1]
    if last.get("role") != "assistant":
        return "last message is not role=assistant"
    last_content = last.get("content")
    # Final assistant turn must be text-only (no tool_use). Allow string OR
    # a list of {type:text} blocks.
    if isinstance(last_content, list):
        if any(b.get("type") == "tool_use" for b in last_content if isinstance(b, dict)):
            return "final assistant turn still contains tool_use blocks"
        if not any(b.get("type") == "text" and (b.get("text") or "").strip()
                   for b in last_content if isinstance(b, dict)):
            return "final assistant turn has no text"
    elif isinstance(last_content, str):
        if not last_content.strip():
            return "final assistant turn is empty"
    else:
        return f"final assistant content has odd type: {type(last_content).__name__}"
    return None
