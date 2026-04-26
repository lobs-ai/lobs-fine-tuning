"""
Phase 4 — render canonical trajectories into training tensors for any
chat-templated HF causal LM.

Two output kinds:
- text-only inspection JSONL (--text-out): one record per trajectory with
  the rendered prompt + the assistant turns rendered separately. Use this
  to eyeball the chat template result before tokenizing thousands.
- tokenized JSONL (--train-out / --val-out): one record per trajectory
  with `{input_ids, attention_mask, labels}` where labels=-100 on every
  token the model should NOT learn to predict (system, user, tool_result).

Model-agnostic. Pass any HF tokenizer id whose chat template supports the
OpenAI-style `tools=` argument and the `role: "tool"` reply shape — that
includes Qwen2/Qwen2.5/Qwen3, Llama 3.x Instruct, Mistral Instruct, etc.

Loss masking strategy
---------------------
Spec §9.2: train on assistant turns only, including their tool_call tokens.
Mask everything else. Implementation:

1. Build the *full* messages list (system + user + assistant + tool_result + …).
2. Render the full conversation with `apply_chat_template(..., tokenize=False)`
   to get both rendered text and (separately) the token sequence.
3. For each assistant turn, locate its rendered segment by templating the
   prefix-up-to-and-including each assistant turn and the prefix-just-before,
   then taking the substring delta. Translate the byte span to a token span
   using the tokenizer's `offset_mapping` when available (fast tokenizers),
   else fall back to re-tokenizing prefixes.
4. Set labels = input_ids inside assistant spans, -100 elsewhere.

The verification step decodes the unmasked positions and asserts the result
contains the assistant text. If that check fails, do not start training.

Run:
    python -m data_gen.format_for_training \
      --in data/trajectories_filtered/all.jsonl \
      --train-out data/train.jsonl \
      --val-out data/val.jsonl \
      --tokenizer "$MODEL" \
      --max-len 8192 \
      --val-frac 0.05 \
      --verify
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_gen.trajectory import (
    AssistantTurn,
    ToolCall,
    ToolResultTurn,
    Trajectory,
    UserTurn,
    read_jsonl,
)
from tools.schemas import schemas_for, to_openai_tools

# Lazy-import transformers so the rest of this module is testable without it.
def _load_tokenizer(name: str):  # noqa: ANN202
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return tok


# ── render canonical → openai/qwen-style messages ───────────────────────────


def trajectory_to_messages(traj: Trajectory) -> list[dict[str, Any]]:
    """
    Convert a Trajectory to the OpenAI-style message list that any modern
    instruct chat template consumes: assistant turns may carry `tool_calls`,
    tool replies are separate `role:"tool"` messages.

    System prompt: if the trajectory recorded one, use it; otherwise build
    a minimal default that matches squad's runtime default (so the student
    learns to attend to the tool docs in the system message, not memorise
    a particular prompt).
    """
    system = traj.meta.system_prompt.strip()
    if not system:
        system = (
            f"You are a {traj.meta.agent_type or 'squad'} agent. Use the available tools "
            "to complete the user's task. Read files before editing them, prefer targeted "
            "actions over broad sweeps, and stop once the task is done."
        )

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    for turn in traj.turns:
        if isinstance(turn, UserTurn):
            messages.append({"role": "user", "content": turn.text})
        elif isinstance(turn, AssistantTurn):
            msg: dict[str, Any] = {"role": "assistant", "content": turn.text}
            if turn.tool_calls:
                msg["tool_calls"] = [_tool_call_to_openai(tc) for tc in turn.tool_calls]
            messages.append(msg)
        elif isinstance(turn, ToolResultTurn):
            for r in turn.results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": r.tool_use_id,
                        "content": r.content if not r.is_error else f"[error] {r.content}",
                    }
                )
    return messages


def _tool_call_to_openai(tc: ToolCall) -> dict[str, Any]:
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
        },
    }


def trajectory_tools(traj: Trajectory) -> list[dict[str, Any]]:
    """Tool definitions to inject into the chat template."""
    names = traj.meta.tools_offered or sorted(
        {tc.name for tc in traj.tool_calls} or set()
    )
    return to_openai_tools(schemas_for(names) if names else schemas_for())


# ── loss masking ────────────────────────────────────────────────────────────


@dataclass
class FormattedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    rendered: str
    n_assistant_tokens: int
    n_total_tokens: int


def _find_segment_spans(rendered: str, segments: list[str]) -> list[tuple[int, int]]:
    """
    For each segment string, find its (start, end) byte offset inside
    `rendered`. Searches left-to-right; each match advances the cursor.
    Raises if a segment cannot be found.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, seg in enumerate(segments):
        # Some templates emit assistant content *after* a role marker
        # (e.g., "<|im_start|>assistant\n"). The seg we render in isolation
        # may include that marker too. We search for the seg verbatim;
        # callers should make sure both renders use the same template.
        idx = rendered.find(seg, cursor)
        if idx < 0:
            raise ValueError(
                f"assistant segment {i} not found in rendered conversation; "
                f"this means the chat template did not round-trip cleanly. "
                f"Segment starts: {seg[:80]!r}"
            )
        spans.append((idx, idx + len(seg)))
        cursor = idx + len(seg)
    return spans


def _byte_spans_to_token_spans(
    tokenizer: Any,
    text: str,
    byte_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Translate (byte_start, byte_end) → (token_start, token_end) using the
    tokenizer's offset_mapping if available, else by re-tokenizing prefixes.
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets: list[tuple[int, int]] = enc.get("offset_mapping") or []
    if offsets:
        out: list[tuple[int, int]] = []
        for bs, be in byte_spans:
            t_start = next(
                (i for i, (a, b) in enumerate(offsets) if b > bs),
                len(offsets),
            )
            t_end = next(
                (i for i, (a, b) in enumerate(offsets) if a >= be),
                len(offsets),
            )
            out.append((t_start, t_end))
        return out

    # Fallback: tokenize prefixes. Slower but works for any tokenizer.
    out = []
    for bs, be in byte_spans:
        t_start = len(
            tokenizer(text[:bs], add_special_tokens=False)["input_ids"]
        )
        t_end = len(
            tokenizer(text[:be], add_special_tokens=False)["input_ids"]
        )
        out.append((t_start, t_end))
    return out


def format_one(
    traj: Trajectory,
    tokenizer: Any,
    *,
    max_len: int,
) -> FormattedExample | None:
    messages = trajectory_to_messages(traj)
    tools = trajectory_tools(traj)

    # 1. render full conversation as text (for span finding) and as tokens.
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer(
        rendered, add_special_tokens=False, return_attention_mask=False
    )["input_ids"]

    if len(full_ids) > max_len:
        return None  # spec §4: drop sequences over max_len after templating

    # 2. render each assistant turn IN ITS POSITION inside the full
    #    conversation. We do this by templating the prefix-up-to-and-including
    #    each assistant turn and the prefix-just-before, then taking the
    #    substring delta. This is more robust than templating the assistant
    #    turn alone (which might insert different role markers in isolation).
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    segments: list[str] = []
    for idx in assistant_indices:
        before = tokenizer.apply_chat_template(
            messages[:idx], tools=tools, tokenize=False, add_generation_prompt=False
        )
        through = tokenizer.apply_chat_template(
            messages[: idx + 1], tools=tools, tokenize=False, add_generation_prompt=False
        )
        if not through.startswith(before):
            return None  # template isn't deterministic for prefixes; bail
        seg = through[len(before):]
        if not seg.strip():
            continue
        segments.append(seg)

    if not segments:
        return None  # nothing to learn from

    byte_spans = _find_segment_spans(rendered, segments)
    token_spans = _byte_spans_to_token_spans(tokenizer, rendered, byte_spans)

    # 3. build labels = input_ids on assistant spans, -100 elsewhere
    labels = [-100] * len(full_ids)
    n_assistant_tokens = 0
    for ts, te in token_spans:
        ts = max(0, min(ts, len(full_ids)))
        te = max(0, min(te, len(full_ids)))
        for i in range(ts, te):
            labels[i] = full_ids[i]
            n_assistant_tokens += 1

    if n_assistant_tokens == 0:
        return None

    return FormattedExample(
        input_ids=full_ids,
        attention_mask=[1] * len(full_ids),
        labels=labels,
        rendered=rendered,
        n_assistant_tokens=n_assistant_tokens,
        n_total_tokens=len(full_ids),
    )


# ── verification ────────────────────────────────────────────────────────────


def verify_example(ex: FormattedExample, tokenizer: Any, traj: Trajectory) -> str | None:
    """
    Decode the unmasked label positions and check that the result contains
    the assistant text from the trajectory. Returns an error message if
    the check fails, or None if all good.
    """
    unmasked_ids = [tid for tid, lab in zip(ex.input_ids, ex.labels) if lab != -100]
    decoded = tokenizer.decode(unmasked_ids, skip_special_tokens=False)

    for at in traj.assistant_turns:
        text = at.text.strip()
        if not text:
            continue
        # Look for a meaningful chunk; not full-string equality because the
        # chat template wraps assistant content with role markers that will
        # appear in `decoded` too.
        chunk = text[: min(len(text), 64)]
        if chunk and chunk not in decoded:
            return (
                f"assistant text not found in unmasked decode. "
                f"Looked for {chunk!r}, decoded starts: {decoded[:200]!r}"
            )
    return None


# ── stratified split ───────────────────────────────────────────────────────


def split_train_val(
    items: list[Any],
    val_frac: float,
    seed: int,
    stratify: list[str] | None = None,
) -> tuple[list[Any], list[Any]]:
    rng = random.Random(seed)
    if stratify is None or len(set(stratify)) <= 1:
        idx = list(range(len(items)))
        rng.shuffle(idx)
        cut = int(len(idx) * val_frac)
        val = [items[i] for i in idx[:cut]]
        train = [items[i] for i in idx[cut:]]
        return train, val

    by_strata: dict[str, list[int]] = {}
    for i, s in enumerate(stratify):
        by_strata.setdefault(s, []).append(i)
    val_idx: list[int] = []
    for stratum, idxs in by_strata.items():
        rng.shuffle(idxs)
        cut = max(1, int(len(idxs) * val_frac)) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:cut])
    val_set = set(val_idx)
    val = [items[i] for i in val_idx]
    train = [items[i] for i in range(len(items)) if i not in val_set]
    return train, val


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> int:
    import os

    parser = argparse.ArgumentParser(description="Phase 4 — render trajectories into HF-tokenized training JSONL.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--text-out", help="optional human-readable JSONL for inspection")
    parser.add_argument(
        "--messages-train-out",
        help="optional ChatML/OpenAI messages JSONL for upload to Unsloth Studio "
        "(or any external trainer that consumes raw messages). One record per "
        "trajectory: {messages: [...], tools: [...]}. Untokenized; no loss masking applied.",
    )
    parser.add_argument(
        "--messages-val-out",
        help="optional val split for --messages-train-out",
    )
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("MODEL", ""),
        help="HF tokenizer id (default: $MODEL). Required if $MODEL is unset.",
    )
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run round-trip verification on a sample of examples (recommended)",
    )
    parser.add_argument(
        "--verify-n",
        type=int,
        default=5,
        help="how many examples to verify when --verify is set",
    )
    args = parser.parse_args()

    if not args.tokenizer:
        print(
            "no tokenizer specified — pass --tokenizer or set $MODEL",
            file=sys.stderr,
        )
        return 2

    trajectories = read_jsonl(Path(args.in_path))
    if not trajectories:
        print("no input trajectories", file=sys.stderr)
        return 2

    tokenizer = _load_tokenizer(args.tokenizer)

    formatted: list[tuple[Trajectory, FormattedExample]] = []
    drops: Counter[str] = Counter()
    for traj in trajectories:
        try:
            ex = format_one(traj, tokenizer, max_len=args.max_len)
        except ValueError as err:
            drops[f"format_error:{type(err).__name__}"] += 1
            continue
        if ex is None:
            drops["over_max_len_or_empty"] += 1
            continue
        formatted.append((traj, ex))

    if args.verify and formatted:
        rng = random.Random(args.seed)
        sample = rng.sample(formatted, k=min(args.verify_n, len(formatted)))
        for traj, ex in sample:
            err = verify_example(ex, tokenizer, traj)
            if err:
                print(f"VERIFICATION FAILED for {traj.meta.run_id}: {err}", file=sys.stderr)
                return 3
        print(f"verification passed on {len(sample)}/{len(sample)} sampled examples", file=sys.stderr)

    examples = [ex for _, ex in formatted]
    trajs = [t for t, _ in formatted]
    strata = [t.meta.extra.get("difficulty", "unknown") if t.meta.extra else "unknown" for t in trajs]

    # Split once on the trajectory list itself so the messages and tokenized
    # outputs always agree on which trajectories went to which split.
    indices = list(range(len(formatted)))
    train_idx, val_idx = split_train_val(indices, args.val_frac, args.seed, stratify=strata)

    train_ex = [examples[i] for i in train_idx]
    val_ex = [examples[i] for i in val_idx]
    _write_tokenized(Path(args.train_out), train_ex)
    _write_tokenized(Path(args.val_out), val_ex)

    if args.messages_train_out:
        train_trajs = [trajs[i] for i in train_idx]
        val_trajs = [trajs[i] for i in val_idx]
        _write_messages(Path(args.messages_train_out), train_trajs)
        if args.messages_val_out:
            _write_messages(Path(args.messages_val_out), val_trajs)

    if args.text_out:
        _write_text(Path(args.text_out), formatted)

    total_tokens = sum(ex.n_total_tokens for ex in examples)
    asst_tokens = sum(ex.n_assistant_tokens for ex in examples)
    print(
        f"formatted {len(examples)} (train={len(train_ex)} val={len(val_ex)}); "
        f"total_tokens={total_tokens} assistant_tokens={asst_tokens} "
        f"drops={dict(drops)}",
        file=sys.stderr,
    )
    return 0


def _write_tokenized(path: Path, examples: list[FormattedExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "labels": ex.labels,
                    }
                )
                + "\n"
            )


def _write_messages(path: Path, trajectories: list[Trajectory]) -> None:
    """
    ChatML/OpenAI messages JSONL — what Unsloth Studio (and any HF chat-
    template-aware trainer) accepts directly. One record per trajectory:

        {"messages": [{"role": "system", ...}, ...],
         "tools": [{"type": "function", "function": {...}}, ...],
         "meta": {"run_id": "...", "category": "..."}}

    No tokenization, no loss masking — Studio applies its own chat template
    and (when `train_on_responses_only` is enabled) masks user/system/tool
    turns. Our spec §9.2 — "train on assistant turns only including tool
    calls" — is exactly what train_on_responses_only does for these
    templates, so the resulting masks match.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for traj in trajectories:
            messages = trajectory_to_messages(traj)
            tools = trajectory_tools(traj)
            record = {
                "messages": messages,
                "tools": tools,
                "meta": {
                    "run_id": traj.meta.run_id,
                    "category": (traj.meta.extra or {}).get("category", ""),
                    "agent_type": traj.meta.agent_type,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_text(path: Path, items: list[tuple[Trajectory, FormattedExample]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for traj, ex in items:
            f.write(
                json.dumps(
                    {
                        "run_id": traj.meta.run_id,
                        "rendered": ex.rendered,
                        "n_assistant_tokens": ex.n_assistant_tokens,
                        "n_total_tokens": ex.n_total_tokens,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    raise SystemExit(main())
