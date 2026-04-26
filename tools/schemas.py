"""
Vendored tool schemas from squad.

Source: ~/other/lobs/squad @ commit 7daf6dfde0ac105d19d48908f38abd64817d3782
(the vendor sync hash referenced in squad's session-transcript.ts)

These are the EXACT tool definitions the teacher (Claude Sonnet via squad)
sees during data generation. They must also be the exact definitions the
student (fine-tuned Qwen via squad) sees at eval time. If squad updates
these schemas, this file must be re-vendored AND all training data must be
regenerated. Schema drift between data-gen and eval silently destroys the
fine-tune.

Schema format is "Anthropic-style" (`{name, description, input_schema}`),
which is what squad uses internally. The format_for_training step
re-renders these into whatever the target tokenizer's chat template
expects (Qwen 2.5 uses an OpenAI-flavored tools array natively).
"""

from __future__ import annotations

from typing import Any

ToolSchema = dict[str, Any]


READ_TOOL: ToolSchema = {
    "name": "read",
    "description": (
        "Reads a file from the local filesystem. Assume any user-provided path is worth checking. "
        "Use an absolute path via file_path when possible. By default it reads from the start of the file "
        "and returns line-numbered text. When you already know the area you need, use offset and limit "
        "for a targeted read instead of re-reading the whole file. This tool reads files only, not directories."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file to read"},
            "path": {"type": "string", "description": "Backward-compatible path field; file_path is preferred"},
            "offset": {"type": "number", "description": "Optional 1-based line number to start reading from"},
            "limit": {"type": "number", "description": "Optional maximum number of lines to read"},
            "full": {
                "type": "boolean",
                "description": "Return the entire file without truncation (fails for files > 200KB)",
            },
        },
        "required": [],
    },
}


EDIT_TOOL: ToolSchema = {
    "name": "edit",
    "description": (
        "Performs exact string replacements in files. You must use Read on the file before editing it. "
        "Preserve exact indentation and whitespace exactly as it appears in the file, excluding any "
        "line-number prefix from Read output. Use the smallest clearly unique old_string you can, "
        "usually only a few adjacent lines. The edit fails if old_string is ambiguous; provide more "
        "context or use replace_all when you intentionally want every instance updated."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file to edit"},
            "path": {"type": "string", "description": "Backward-compatible path field; file_path is preferred"},
            "old_string": {"type": "string", "description": "Exact text to find and replace (must match exactly)"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of enforcing uniqueness (default: false)",
            },
            "edits": {
                "type": "array",
                "description": (
                    "Batch of edits to apply in sequence. Use instead of old_string/new_string to make "
                    "several changes in one call."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": [],
    },
}


WRITE_TOOL: ToolSchema = {
    "name": "write",
    "description": (
        "Writes a file to the local filesystem, creating it if it doesn't exist or overwriting it if it does. "
        "Prefer Edit when the file already exists and you only need to change part of it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file to write"},
            "content": {"type": "string", "description": "Full file contents"},
        },
        "required": ["file_path", "content"],
    },
}


LS_TOOL: ToolSchema = {
    "name": "ls",
    "description": (
        "List entries in a directory. Returns names with type (file/dir) and size. Useful for "
        "exploring an unfamiliar project layout."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute directory path"},
        },
        "required": ["path"],
    },
}


GREP_TOOL: ToolSchema = {
    "name": "grep",
    "description": (
        "Search for a regex pattern across files. Returns matching lines with file path and line number. "
        "Prefer this over `exec grep` so the agent loop can cap output cleanly."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression to search for"},
            "path": {"type": "string", "description": "Directory or file to search (defaults to cwd)"},
            "include": {"type": "string", "description": "Glob filter for files to include, e.g. '*.py'"},
            "case_insensitive": {"type": "boolean"},
        },
        "required": ["pattern"],
    },
}


EXEC_TOOL: ToolSchema = {
    "name": "exec",
    "description": (
        "Execute a shell command in the current working directory or an optional workdir. "
        "Returns structured stdout, stderr, and exit status. Prefer dedicated tools like Read, "
        "Edit, Glob, and Grep when they fit the task instead of routing everything through Bash. "
        "Prefer targeted commands over huge output. Use timeout to limit execution time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "cmd": {"type": "string", "description": "Shell command to execute"},
            "command": {"type": "string", "description": "Backward-compatible command field; cmd is preferred"},
            "workdir": {"type": "string", "description": "Working directory (defaults to agent cwd)"},
            "timeout": {"type": "number", "description": "Timeout in seconds (default 30, max 300)"},
            "env": {
                "type": "object",
                "description": "Additional environment variables",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": [],
    },
}


CREATE_TASK_TOOL: ToolSchema = {
    "name": "create_task",
    "description": (
        "Add a new task to the session tree's shared task list. Returns the task id. "
        "Use this to break large work into trackable units, then update_task as you progress."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Short imperative title, e.g. 'Fix login redirect'"},
            "description": {"type": "string", "description": "Full detail of what needs to be done"},
            "activeForm": {
                "type": "string",
                "description": "Present-continuous form for the spinner, e.g. 'Fixing login redirect'",
            },
            "blockedBy": {"type": "array", "items": {"type": "string"}},
            "blocks": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object"},
        },
        "required": ["subject", "description"],
    },
}


UPDATE_TASK_TOOL: ToolSchema = {
    "name": "update_task",
    "description": (
        "Update a task: change status, subject, description, owner, or dependencies. "
        "Claim a task by setting owner (typically to yourself) and status to in_progress in the same call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "taskId": {"type": "string"},
            "subject": {"type": "string"},
            "description": {"type": "string"},
            "activeForm": {"type": "string"},
            "owner": {"type": ["string", "null"]},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "deleted"],
            },
            "addBlocks": {"type": "array", "items": {"type": "string"}},
            "addBlockedBy": {"type": "array", "items": {"type": "string"}},
            "removeBlocks": {"type": "array", "items": {"type": "string"}},
            "removeBlockedBy": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object"},
        },
        "required": ["taskId"],
    },
}


LIST_TASKS_TOOL: ToolSchema = {
    "name": "list_tasks",
    "description": (
        "List the current tasks in the session tree's shared task list (ordered by creation time)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "includeDeleted": {"type": "boolean"},
            "status": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "deleted"],
                },
            },
        },
    },
}


_PATH_GUIDANCE = (
    "Paths are dot-separated. Numeric segments address array indices. "
    "Examples: `llm.primary.model`, `llm.fallbacks.0.model`, "
    "`subagents.max_concurrent_global`, `chat.delivery.mode`, `auth.tokens.0.scopes`, "
    "`policy.approvals.require_for_tags`. "
    "Use list_config_paths to discover what's currently set."
)


GET_CONFIG_TOOL: ToolSchema = {
    "name": "get_config",
    "description": (
        f"Read a value from the gateway config. Omit `path` to get the full config tree. {_PATH_GUIDANCE}"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Dot-path into the config (empty for full tree)"},
        },
    },
}


SET_CONFIG_TOOL: ToolSchema = {
    "name": "set_config",
    "description": (
        "Write a value to the gateway config, persisting to config.json. The new config is validated "
        "through the same schema used at boot; invalid values are rejected without touching disk. "
        "Some changes (subagent pool limits, server port, provider keys, loaded plugins) only take effect "
        f"on restart — the return payload flags these. {_PATH_GUIDANCE}"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Dot-path into the config"},
            "value": {
                "description": "The new value. May be any JSON-compatible type (string, number, boolean, object, array).",
            },
        },
        "required": ["path", "value"],
    },
}


LIST_CONFIG_PATHS_TOOL: ToolSchema = {
    "name": "list_config_paths",
    "description": (
        "List every leaf dot-path currently present in the gateway config. "
        "Use this to discover what's configurable before calling set_config."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}


TOOL_SCHEMAS: dict[str, ToolSchema] = {
    t["name"]: t
    for t in [
        READ_TOOL,
        EDIT_TOOL,
        WRITE_TOOL,
        LS_TOOL,
        GREP_TOOL,
        EXEC_TOOL,
        CREATE_TASK_TOOL,
        UPDATE_TASK_TOOL,
        LIST_TASKS_TOOL,
        GET_CONFIG_TOOL,
        SET_CONFIG_TOOL,
        LIST_CONFIG_PATHS_TOOL,
    ]
}

ALLOWED_TOOLS: frozenset[str] = frozenset(TOOL_SCHEMAS.keys())


def get_schema(name: str) -> ToolSchema:
    if name not in TOOL_SCHEMAS:
        raise KeyError(f"unknown tool {name!r}; allowed: {sorted(ALLOWED_TOOLS)}")
    return TOOL_SCHEMAS[name]


def schemas_for(names: list[str] | None = None) -> list[ToolSchema]:
    """Return the canonical list of schemas, optionally filtered to `names`."""
    if names is None:
        return list(TOOL_SCHEMAS.values())
    return [TOOL_SCHEMAS[n] for n in names if n in TOOL_SCHEMAS]


def to_openai_tools(schemas: list[ToolSchema]) -> list[dict[str, Any]]:
    """
    Translate Anthropic-style {name, description, input_schema} into the
    OpenAI-style {type:"function", function:{name, description, parameters}}.
    Qwen 2.5's chat template expects this OpenAI flavor natively.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", help="print the schemas as JSON")
    parser.add_argument("--openai", action="store_true", help="print in OpenAI/Qwen format")
    args = parser.parse_args()

    schemas = list(TOOL_SCHEMAS.values())
    out = to_openai_tools(schemas) if args.openai else schemas
    print(json.dumps(out, indent=2))
