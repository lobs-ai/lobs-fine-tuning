# DECISIONS.md

Log of choices that deviate from `SPEC.md` or that future-me will want explained.

## Target system: squad

The fine-tune targets the **squad** agent system at `~/other/lobs/squad`. Squad gives us four things the spec assumed we'd build:

1. A working agent loop (`packages/runner/src/agent-loop.ts`) — multi-turn, parallel tool calls, loop detection, timeouts.
2. Tool definitions in JSON Schema, already compatible with Anthropic + OpenAI function-calling formats.
3. Multi-provider LLM client (`packages/llm`) — Anthropic, OpenAI, and OpenAI-compatible (vLLM, OpenRouter, Ollama, etc.). Same code path for the teacher (Claude) and the student (Qwen via vLLM).
4. Session transcript writer (`packages/runner/src/session-transcript.ts`).

Implication: Phase 0 ("define and freeze tools") becomes a vendoring exercise. Phase 2 ("build an agent loop") becomes "drive `runAgent()`". Phase 6's eval harness reuses the same code path.

## Tool subset (FROZEN)

We pick **10 tools** out of squad's ~15+ surface:

| name              | category | source                                   |
| ----------------- | -------- | ---------------------------------------- |
| `read`            | files    | `packages/tools/src/read.ts`             |
| `edit`            | files    | `packages/tools/src/edit.ts`             |
| `write`           | files    | `packages/tools/src/write.ts`            |
| `ls`              | files    | `packages/tools/src/ls.ts`               |
| `grep`            | search   | `packages/tools/src/grep.ts`             |
| `exec`            | shell    | `packages/tools/src/exec.ts`             |
| `create_task`     | tasks    | `packages/tools/src/tasks/tools.ts`      |
| `update_task`     | tasks    | `packages/tools/src/tasks/tools.ts`      |
| `list_tasks`      | tasks    | `packages/tools/src/tasks/tools.ts`      |
| `get_config`      | config   | `packages/tools/src/config/tools.ts`     |
| `set_config`      | config   | `packages/tools/src/config/tools.ts`     |
| `list_config_paths` | config | `packages/tools/src/config/tools.ts`     |

OK, that's actually 12. Reasoning:

- **Files (4):** the spec requires "tool that can fail gracefully" — `edit` fits, since it fails when `old_string` is ambiguous or the file wasn't read. `write` is included because real squad sessions include file creation.
- **search (1):** `grep` covers the bulk of "find a thing" queries; we drop `glob`, `find_files`, `code_search` to keep the surface tight (collapsible into grep for v1).
- **shell (1):** `exec` for verification, builds, tests. The spec's "fail gracefully" requirement is also satisfied here (non-zero exits return structured errors).
- **tasks (3):** include the squad-distinctive task tools, but drop `get_task` since `list_tasks` covers it for v1 and the model only needs to learn *one* read pattern.
- **config (3):** `get_config`, `set_config`, `list_config_paths`. Excluding `unset_config` keeps the write surface to a single mutating tool. Config tools are squad's "structured key-value store" tools — they exercise schema-validated arguments and dot-path navigation.

**Excluded from v1:**
- `spawn_subagent` — produces nested trajectories, blows up sequence length, save for v2.
- `ask_user` — interactive; can't be replayed from transcripts deterministically.
- `web_search`, `web_fetch` — non-deterministic; would force sandboxing the internet.
- `pptx_*`, `html_to_pdf` — niche; exec covers most of what they do.
- `unset_config`, `get_task`, `glob`, `find_files`, `code_search` — collapsible into kept tools.

**Finalize tool:** the spec asks for an explicit `submit_answer`. Squad's loop terminates on `stop_reason == "end_turn"` with a final assistant text block. We follow that — termination is "assistant produces text without a tool call." This matches squad's actual behavior, so the model learns the right termination pattern for the real system.

## Data sources (in priority order)

1. **Live squad runs.** When squad is invoked (gateway or direct `runAgent()`), the runner logs to `$SQUAD_TRANSCRIPTS_DIR/{agentType}/sessions/{runId}.jsonl` (defaults to `~/.lobs/agents`, the path squad's `session-transcript.ts` hard-codes today). Currently `writeTurn()` is defined but **never called by the agent loop** — only the summary is written. We're shipping a wrapper script (`data_gen/squad_runner.ts`) that calls `runAgent()` directly and dumps the full `Session` message history to a richer JSONL format.
2. **Gateway DB.** Squad's gateway home is `$SQUAD_HOME` (default `~/.squad/`). Each registered squad has its own DB at `$SQUAD_HOME/{name}/squad.db` with `chat_messages` + `tool_calls` tables (currently empty in this install). Harvester reads these once populated; the squad name is `--squad-name` (default `default`, also reads `$SQUAD_NAME`).
3. **Manually exported Anthropic-format JSON.** Drop a JSON file with `{messages: [...]}` into `data/trajectories_raw/manual/` and the harvester picks it up.

The squad repo path is resolved from `$SQUAD_HOME/config` (the env-style file `squad onboard` writes), with `$SQUAD_REPO` as override. Both Python (`data_gen.squad_config`) and TS (`squad_runner.ts`) honor this.

The harvester normalizes all three to a single canonical trajectory format (see `data_gen/harvester.py`).

## Schema vendoring

`tools/schemas.py` contains hand-ported JSON Schemas for the 12 chosen tools, copied from squad commit `7daf6dfd…` (the vendor sync hash from `session-transcript.ts`). If squad updates these schemas, we re-vendor, bump a version, and **regenerate all training data**. The vendored schemas are the source of truth for both the teacher's tool definitions during data gen and the student's tool definitions at eval.

We do not vendor the *implementations* — for data gen we run real squad. For training data conversion we only need the schemas (so the teacher message has the same JSON describing each tool that the eval harness will use).

## Teacher model

`anthropic/claude-sonnet-4-5` (squad's default). Squad handles the API call; we just pass the model string. Falls back automatically if it's down (squad's model-chain feature).

## Format pipeline

Four normalization layers between raw squad transcripts and Qwen training tensors:

```
raw transcript (summary JSONL | full JSONL | gateway DB | Anthropic JSON)
        │
        ▼  data_gen/harvester.py
canonical trajectory (our own simple shape — see Trajectory dataclass)
        │
        ▼  data_gen/secret_scrub.py  (ALWAYS run before training)
scrubbed canonical trajectory
        │
        ▼  data_gen/filter.py  (hard filters only in v1; soft LLM-judge filter is v2)
filtered trajectory
        │
        ▼  data_gen/format_for_training.py
Qwen-templated example with masked labels (train.jsonl)
```

## Secret scrubbing

We never want a secret in the model's weights. `data_gen/secret_scrub.py` runs three layers, in order:

1. **Live secrets seeded from local config.** Reads `$SQUAD_HOME/config` for `SQUAD_TOKEN`/etc. and the `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `WANDB_API_KEY` / `HF_TOKEN` / `GITHUB_TOKEN` env vars; replaces those exact strings wherever they appear. Catches the tail risk of the model echoing the user's actual token verbatim.
2. **Pattern catalog.** Anthropic / OpenAI / GitHub / AWS keys, Slack bot tokens, Bearer auth headers, JWTs, PEM private-key blocks, env-style `KEY=value` assignments. The fallback "long URL-safe base64 token" pattern only fires when the surrounding context contains words like `token`/`key`/`secret` — we don't shred random commit hashes.
3. **Squad-specific config-path awareness.** `set_config` calls whose `path` matches `auth.tokens.*`, `llm.providers.*.apiKey`, or anything containing `key`/`secret`/`token`/`password` get the `value` field replaced with `<REDACTED:sensitive_config_path>`. The call shape is preserved so the model still learns the right structure.

Anything matching an existing placeholder (`<REDACTED…>`, `${ENV_VAR}`, `xxx-xxxx`) is left alone.

A trajectory containing a PEM/SSH private-key block is **dropped entirely** rather than redacted — pattern scrubbing replaces the body, but the surrounding context is too suspect to keep.

Output is a JSON report (`--report`) listing the hit count per pattern and any drop reasons. Treat any non-zero `live_secret` hit as a finding to investigate — that's a real token that escaped into a transcript.

The canonical trajectory shape is deliberately *not* Anthropic's wire format — it's a flat list of typed turns (`UserTurn`, `AssistantTurn`, `ToolResultTurn`) that's trivial to translate into either Anthropic-style or Qwen-style messages. This keeps the harvester independent of which provider squad happened to use.

## Loss masking (the bug-prone bit)

Per spec §9.2:
- Mask system + user + tool-result turns
- Train on assistant turns including tool-call tokens

Implementation lives in `data_gen/format_for_training.py`. We compute the mask by templating each segment separately and tracking byte offsets, then translating offsets to token indices via the tokenizer's offset-mapping. There's a verification step that decodes the unmasked positions and asserts they equal the assistant turns rendered in isolation. **Do not start a long training run if that verification doesn't pass.**

## Soft filters: deferred

The spec §8 calls for an LLM-judge soft filter (3×5 ratings). We're skipping it in v1 because:
- It triples data-gen API spend.
- Real squad sessions are already filtered by the user not retrying — there's a built-in success signal.
- It's purely additive; we can add it later without invalidating earlier data.

If hard-filtered data is too noisy (per spec acceptance >75% pass = filters too lenient), turn the LLM judge on.

## Eval

Same code path: serve LoRA-merged Qwen via vLLM on `localhost:8000`, register it in squad's config as an OpenAI-compatible provider:

```json
{
  "llm": {
    "primary": { "model": "openai-compat/qwen-ft", "baseURL": "http://localhost:8000/v1" }
  }
}
```

…then run the held-out task set through `runAgent()`. Same tool registry, same prompts. Apples-to-apples vs. teacher and base.

## Open follow-ups (not blocking v1)

- Patch squad to actually call `transcript.writeTurn()` at every turn (10-line PR).
- Add `spawn_subagent` to the trained tool surface once nested trajectory flattening is figured out.
- Hook the gateway DB harvester up once a real install accumulates traffic.
- Mix in some general-purpose chat data to combat catastrophic forgetting (only if the report calls it out as a problem).
