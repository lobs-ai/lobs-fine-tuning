"""
Held-out tool-call smoke eval.

Run a model (base, or base+LoRA, or merged weights) on prompts pulled
from a messages JSONL (typically `data/val.messages.jsonl`). For each
held-out trajectory, truncate the conversation right before the first
assistant turn that contained tool_calls, generate the next turn, and
score the output.

Metrics, all reported as percentages:
  any_tool_call   — generated text contained at least one parseable tool_call
  allowed_tool    — that tool name was in our 12-tool allowlist
  required_fields — call had every field marked `required` in the schema
  exact_tool_match — call's tool name matched the held-out expected name

`exact_tool_match` is the strictest: it asks "did the model converge on the
*same* decision the teacher made?". The first three are looser shape checks.
The base-vs-adapter delta on these is the headline number you care about.

Run:
    # base model only (baseline)
    python -m eval.smoke --model Qwen/Qwen2.5-7B-Instruct
    # base + LoRA adapter
    python -m eval.smoke --model Qwen/Qwen2.5-7B-Instruct --adapter-dir artifacts/adapter
    # 20 prompts, save full report
    python -m eval.smoke --model ... --adapter-dir ... --n 20 --report-out artifacts/eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tools.schemas import ALLOWED_TOOLS, TOOL_SCHEMAS, schemas_for, to_openai_tools


# ── dataset prompts ─────────────────────────────────────────────────────────


@dataclass
class HeldOutPrompt:
    run_id: str
    category: str
    messages_prefix: list[dict[str, Any]]  # everything up to (not including) the target turn
    expected_tool_calls: list[dict[str, Any]]  # the ground-truth tool_calls of that turn
    tools: list[dict[str, Any]]


def _load_held_out(path: Path, n: int | None) -> list[HeldOutPrompt]:
    """
    Walk val.messages.jsonl. For each record, find the first assistant turn
    whose `tool_calls` field is non-empty (or whose content list contains a
    tool_use block — depends on the writer). Truncate the messages at that
    point and return the prefix + the expected calls.
    """
    out: list[HeldOutPrompt] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec.get("messages", [])
            tools = rec.get("tools") or to_openai_tools(schemas_for())
            meta = rec.get("meta") or {}

            cut_at = None
            expected_calls: list[dict[str, Any]] = []
            for i, m in enumerate(messages):
                if m.get("role") != "assistant":
                    continue
                calls = m.get("tool_calls") or []
                if calls:
                    cut_at = i
                    expected_calls = calls
                    break
            if cut_at is None:
                # No tool-calling turn in this trajectory; can't smoke-test it.
                continue

            out.append(
                HeldOutPrompt(
                    run_id=meta.get("run_id", "?"),
                    category=meta.get("category", "?"),
                    messages_prefix=messages[:cut_at],
                    expected_tool_calls=expected_calls,
                    tools=tools,
                )
            )
            if n is not None and len(out) >= n:
                break
    return out


# ── tool-call parsing (model-output → list of {name, arguments}) ────────────


_QWEN_TC_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_LLAMA_TC_RE = re.compile(r"<\|python_tag\|>\s*(\{.*?\})(?:\s*<\|eom_id\|>)?", re.DOTALL)


def _safe_json(s: str) -> dict[str, Any] | None:
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        return None


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """
    Best-effort extraction of OpenAI-style {name, arguments} dicts from
    a model's free-form generated text. Handles Qwen's <tool_call>…</tool_call>,
    Llama's <|python_tag|>…, and plain {"name": "...", "arguments": {...}}
    JSON blocks. Falls back to scanning JSON-looking braces.
    """
    out: list[dict[str, Any]] = []

    for m in _QWEN_TC_RE.finditer(text):
        obj = _safe_json(m.group(1))
        if obj and "name" in obj:
            out.append({"name": obj["name"], "arguments": obj.get("arguments") or obj.get("parameters") or {}})
    if out:
        return out

    for m in _LLAMA_TC_RE.finditer(text):
        obj = _safe_json(m.group(1))
        if obj and "name" in obj:
            out.append({"name": obj["name"], "arguments": obj.get("arguments") or obj.get("parameters") or {}})
    if out:
        return out

    # Generic fallback: find every top-level {…} block, json.loads, keep ones
    # that look like tool calls.
    for chunk in _iter_balanced_braces(text):
        obj = _safe_json(chunk)
        if obj and "name" in obj and obj["name"] in ALLOWED_TOOLS:
            out.append({"name": obj["name"], "arguments": obj.get("arguments") or obj.get("parameters") or {}})
    return out


def _iter_balanced_braces(text: str) -> list[str]:
    """Scan `text` left-to-right and yield every balanced {…} block."""
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
    return out


# ── scoring ─────────────────────────────────────────────────────────────────


@dataclass
class CallScore:
    any_tool_call: bool = False
    allowed_tool: bool = False
    required_fields: bool = False
    exact_tool_match: bool = False


@dataclass
class PromptResult:
    run_id: str
    category: str
    expected_tools: list[str]
    generated_text: str
    parsed_calls: list[dict[str, Any]]
    score: CallScore = field(default_factory=CallScore)


def _score_one(parsed: list[dict[str, Any]], expected: list[dict[str, Any]]) -> CallScore:
    s = CallScore()
    if not parsed:
        return s
    s.any_tool_call = True
    first = parsed[0]
    name = first.get("name", "")
    args = first.get("arguments") or {}
    if not isinstance(args, dict):
        # Some templates emit arguments as a JSON-encoded string.
        if isinstance(args, str):
            parsed_args = _safe_json(args)
            args = parsed_args or {}
        else:
            args = {}
    s.allowed_tool = name in ALLOWED_TOOLS
    if s.allowed_tool:
        schema = TOOL_SCHEMAS[name]
        required = schema.get("input_schema", {}).get("required", []) or []
        s.required_fields = all(r in args for r in required)
    expected_first = expected[0] if expected else None
    if expected_first:
        ename = expected_first.get("function", {}).get("name") or expected_first.get("name") or ""
        s.exact_tool_match = (name == ename)
    return s


@dataclass
class EvalReport:
    label: str
    n: int = 0
    counts: dict[str, int] = field(default_factory=lambda: {
        "any_tool_call": 0,
        "allowed_tool": 0,
        "required_fields": 0,
        "exact_tool_match": 0,
    })
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)
    per_prompt: list[PromptResult] = field(default_factory=list)

    def absorb(self, pr: PromptResult) -> None:
        self.n += 1
        self.per_prompt.append(pr)
        for k in self.counts:
            if getattr(pr.score, k):
                self.counts[k] += 1
        cat = pr.category
        cb = self.by_category.setdefault(cat, dict(self.counts.fromkeys(self.counts, 0)))
        cb["n"] = cb.get("n", 0) + 1
        for k in ("any_tool_call", "allowed_tool", "required_fields", "exact_tool_match"):
            if getattr(pr.score, k):
                cb[k] = cb.get(k, 0) + 1

    def pct(self, key: str) -> float:
        return 100.0 * self.counts[key] / self.n if self.n else 0.0


# ── model loading + generation ──────────────────────────────────────────────


def _load_model(
    model_name: str,
    adapter_dir: str | None,
    *,
    max_seq_length: int,
    load_in_4bit: bool,
):  # noqa: ANN202 — heavy types
    """
    Load model + tokenizer. If adapter_dir is set, layer the adapter on top
    of the base model. Uses Unsloth's FastLanguageModel for speed; falls
    back to plain HF + PEFT if Unsloth isn't installed.
    """
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except ImportError:
        # Plain HF path — slower but no Unsloth dep needed.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
        return model, tokenizer


def _generate(model, tokenizer, prompt: HeldOutPrompt, max_new_tokens: int) -> str:
    import torch

    text = tokenizer.apply_chat_template(
        prompt.messages_prefix,
        tools=prompt.tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


# ── orchestration ───────────────────────────────────────────────────────────


def run_eval(
    label: str,
    model_name: str,
    adapter_dir: str | None,
    prompts: list[HeldOutPrompt],
    *,
    max_seq_length: int,
    max_new_tokens: int,
    load_in_4bit: bool,
    verbose: bool,
) -> EvalReport:
    print(f"\n▸ loading [{label}] model={model_name} adapter={adapter_dir}", file=sys.stderr)
    model, tokenizer = _load_model(
        model_name,
        adapter_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    report = EvalReport(label=label)
    for i, prompt in enumerate(prompts):
        try:
            generated = _generate(model, tokenizer, prompt, max_new_tokens)
        except Exception as e:  # noqa: BLE001
            generated = f"[generation error: {e}]"
        parsed = parse_tool_calls(generated)
        score = _score_one(parsed, prompt.expected_tool_calls)
        expected_names = [
            (c.get("function", {}).get("name") or c.get("name") or "")
            for c in prompt.expected_tool_calls
        ]
        pr = PromptResult(
            run_id=prompt.run_id,
            category=prompt.category,
            expected_tools=expected_names,
            generated_text=generated,
            parsed_calls=parsed,
            score=score,
        )
        report.absorb(pr)
        if verbose:
            tag = "✓" if score.exact_tool_match else ("·" if score.allowed_tool else "✗")
            print(
                f"  {tag} [{i+1}/{len(prompts)}] {prompt.category:18s} "
                f"want={','.join(expected_names) or '?':12s} "
                f"got={parsed[0]['name'] if parsed else '<no call>':12s}",
                file=sys.stderr,
            )

    # Free the model so a second run can load another one.
    try:
        del model
        import gc, torch  # noqa: I001

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return report


def _print_summary(reports: list[EvalReport]) -> None:
    keys = ["any_tool_call", "allowed_tool", "required_fields", "exact_tool_match"]
    pad = max(len(r.label) for r in reports) + 2
    header = f"{'metric':22s}" + "".join(f"{r.label:>{pad}}" for r in reports)
    print(header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:22s}"
        for r in reports:
            row += f"{r.pct(k):>{pad-1}.1f}%"
        print(row)
    print()
    if len(reports) >= 2:
        b, a = reports[0], reports[1]
        print(f"Δ ({a.label} − {b.label}):")
        for k in keys:
            print(f"  {k:22s} {a.pct(k) - b.pct(k):+.1f} pp")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Held-out tool-call smoke eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m eval.smoke --model Qwen/Qwen2.5-7B-Instruct\n"
            "  python -m eval.smoke --model Qwen/Qwen2.5-7B-Instruct --adapter-dir artifacts/adapter\n"
            "  python -m eval.smoke --model ... --adapter-dir ... --n 50 --report-out artifacts/eval.json\n"
        ),
    )
    parser.add_argument("--model", default=os.environ.get("MODEL", ""), help="HF base model id (default: $MODEL)")
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="LoRA adapter directory (omit for base-only eval). When set, both base and adapter are evaluated and diffed.",
    )
    parser.add_argument("--prompts", default="data/val.messages.jsonl", help="messages JSONL of held-out prompts")
    parser.add_argument("--n", type=int, default=20, help="how many prompts to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--no-4bit", action="store_true", help="load full precision (more VRAM, faster on H100)")
    parser.add_argument("--report-out", default=None, help="write the full per-prompt report to this JSON file")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="run only the base model (skip the adapter pass even if --adapter-dir is set)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.model:
        print("no --model — pass --model <hf-id> or set $MODEL", file=sys.stderr)
        return 2

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"no prompts at {prompts_path} — run scripts/build_dataset.sh first", file=sys.stderr)
        return 2
    prompts = _load_held_out(prompts_path, args.n)
    if not prompts:
        print(f"no prompts in {prompts_path} contained a tool-calling turn", file=sys.stderr)
        return 2
    print(f"▸ loaded {len(prompts)} held-out prompts from {prompts_path}", file=sys.stderr)

    reports: list[EvalReport] = []
    base_label = f"base ({args.model.split('/')[-1]})"
    reports.append(
        run_eval(
            base_label,
            args.model,
            None,
            prompts,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=not args.no_4bit,
            verbose=args.verbose,
        )
    )
    if args.adapter_dir and not args.base_only:
        reports.append(
            run_eval(
                "adapter",
                args.model,
                args.adapter_dir,
                prompts,
                max_seq_length=args.max_seq_length,
                max_new_tokens=args.max_new_tokens,
                load_in_4bit=not args.no_4bit,
                verbose=args.verbose,
            )
        )

    print()
    _print_summary(reports)

    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                [
                    {
                        "label": r.label,
                        "n": r.n,
                        "counts": r.counts,
                        "by_category": r.by_category,
                        "per_prompt": [asdict(p) for p in r.per_prompt],
                    }
                    for r in reports
                ],
                indent=2,
            )
        )
        print(f"wrote report → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
