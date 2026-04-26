"""
Synthesize one trajectory per invocation by shelling out to the Gemini CLI.

Pipeline per invocation:
1. Build the prompt (schemas + category-specific request) via `prompt.py`.
2. Run `gemini -p <prompt>` (configurable; see GEMINI_BIN / GEMINI_MODEL).
3. Strip any markdown fences from stdout, json.loads it.
4. Validate the shape (validate_trajectory_shape).
5. Write to {out_dir}/{run_id}.json — picked up by data_gen.harvester
   exactly like a real squad full-trajectory dump.

Failure modes (non-zero exit, malformed JSON, schema mismatch) print to
stderr and write the raw stdout to {out_dir}/.failed/{run_id}.txt for
debugging. We never write a partial / invalid trajectory into the harvest
input directory.

CLI:
    python -m data_gen.synth                                # one trajectory, default category mix
    python -m data_gen.synth --category file_edit
    python -m data_gen.synth --n 20 --concurrency 4
    python -m data_gen.synth --gemini-bin /opt/gemini --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_gen.synth.prompt import (
    CATEGORIES,
    SynthRequest,
    render_full_prompt,
    validate_trajectory_shape,
)


_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n(?P<body>.*?)\n```\s*$", re.DOTALL)


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        return m.group("body").strip()
    # Sometimes the model emits a fenced block somewhere in a longer reply.
    m = re.search(r"```(?:json|JSON)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _slug(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


@dataclass
class GeminiConfig:
    bin: str
    model: str | None
    extra_args: list[str]
    timeout: int

    def build_argv(self, prompt: str) -> list[str]:
        argv = [self.bin]
        if self.model:
            argv += ["-m", self.model]
        argv += list(self.extra_args)
        argv += ["-p", prompt]
        return argv


def _run_gemini(cfg: GeminiConfig, prompt: str) -> tuple[int, str, str]:
    """Run the gemini CLI, returning (exit_code, stdout, stderr)."""
    argv = cfg.build_argv(prompt)
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=cfg.timeout,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", f"gemini binary not found: {cfg.bin!r}. Set --gemini-bin or install it."
    except subprocess.TimeoutExpired as e:
        partial = e.stdout
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        return 124, partial or "", f"gemini timed out after {cfg.timeout}s"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _ensure_run_id(obj: dict[str, Any], fallback: str) -> str:
    meta = obj.get("meta")
    if isinstance(meta, dict):
        rid = str(meta.get("runId") or "").strip()
        if rid:
            return rid
        meta["runId"] = fallback
        return fallback
    return fallback


def _write_trajectory(out_dir: Path, obj: dict[str, Any], run_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    return out_path


def _write_failure(out_dir: Path, raw: str, run_id: str, reason: str) -> Path:
    fail_dir = out_dir / ".failed"
    fail_dir.mkdir(parents=True, exist_ok=True)
    p = fail_dir / f"{run_id}.txt"
    p.write_text(f"REASON: {reason}\n\n---\n\n{raw}")
    return p


@dataclass
class GenStats:
    requested: int = 0
    succeeded: int = 0
    failed: int = 0
    by_category: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.by_category is None:
            self.by_category = {}

    def report(self) -> str:
        return (
            f"requested={self.requested} succeeded={self.succeeded} "
            f"failed={self.failed} by_category={self.by_category}"
        )


def generate_one(
    cfg: GeminiConfig,
    out_dir: Path,
    category: str,
    seed: int,
    *,
    verbose: bool = False,
) -> tuple[bool, str]:
    run_id = f"{category}-{_slug()}"
    req = SynthRequest(category=category, seed=seed, run_id_hint=run_id)
    prompt = render_full_prompt(req)

    if verbose:
        print(f"  [{run_id}] running gemini ({len(prompt)} chars prompt)…", file=sys.stderr)

    code, stdout, stderr = _run_gemini(cfg, prompt)
    if code != 0:
        _write_failure(out_dir, f"STDERR:\n{stderr}\n\nSTDOUT:\n{stdout}", run_id, f"exit={code}")
        return False, f"gemini exit={code}: {stderr.strip()[:200]}"

    body = _strip_markdown_fences(stdout)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        _write_failure(out_dir, stdout, run_id, f"json_parse: {e}")
        return False, f"json parse failed: {e}"

    err = validate_trajectory_shape(obj)
    if err:
        _write_failure(out_dir, stdout, run_id, f"shape: {err}")
        return False, f"shape invalid: {err}"

    final_run_id = _ensure_run_id(obj, run_id)
    if isinstance(obj.get("meta"), dict):
        obj["meta"].setdefault("category", category)
        obj["meta"].setdefault("synth_seed", seed)

    path = _write_trajectory(out_dir, obj, final_run_id)
    return True, str(path)


def _category_plan(n: int, category: str | None, mix: list[str] | None) -> list[str]:
    if category:
        if category not in CATEGORIES:
            raise SystemExit(
                f"unknown category {category!r}. Known: {sorted(CATEGORIES)}"
            )
        return [category] * n
    cats = mix or list(CATEGORIES.keys())
    out: list[str] = []
    for i in range(n):
        out.append(cats[i % len(cats)])
    random.shuffle(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize tool-calling trajectories via the Gemini CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m data_gen.synth                  # 1 trajectory, mixed category\n"
            "  python -m data_gen.synth --n 20\n"
            "  python -m data_gen.synth --n 50 --concurrency 4\n"
            "  python -m data_gen.synth --category debugging --n 10\n"
        ),
    )
    parser.add_argument("--n", type=int, default=1, help="how many trajectories to generate")
    parser.add_argument(
        "--category",
        default=None,
        help=f"single category to focus on (default: rotate over all). known: {sorted(CATEGORIES)}",
    )
    parser.add_argument(
        "--mix",
        nargs="+",
        default=None,
        help="explicit subset of categories to round-robin (overrides --category)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/trajectories_raw/full",
        help="where to write {run_id}.json (default: data/trajectories_raw/full — picked up by harvester)",
    )
    parser.add_argument(
        "--gemini-bin",
        default=os.environ.get("GEMINI_BIN", "gemini"),
        help="gemini CLI binary (default: $GEMINI_BIN or 'gemini' on PATH)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", ""),
        help="model id to pass to gemini -m (default: $GEMINI_MODEL, or whatever gemini picks)",
    )
    parser.add_argument(
        "--gemini-arg",
        action="append",
        default=[],
        help="extra arg to forward to gemini (repeatable; e.g. --gemini-arg=--yolo)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("GEMINI_TIMEOUT", "180")),
        help="per-call gemini timeout in seconds (default: 180)",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="parallel gemini calls")
    parser.add_argument("--seed", type=int, default=int(time.time()), help="rng seed (controls diversity nudges)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    cfg = GeminiConfig(
        bin=args.gemini_bin,
        model=args.model or None,
        extra_args=list(args.gemini_arg or []),
        timeout=args.timeout,
    )

    if shutil.which(cfg.bin) is None and not Path(cfg.bin).exists():
        print(
            f"gemini binary not found: {cfg.bin!r}. Install it (npm i -g @google/gemini-cli) "
            f"or pass --gemini-bin / set $GEMINI_BIN.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir)
    plan = _category_plan(args.n, args.category, args.mix)
    stats = GenStats(requested=args.n)

    def task(i: int, cat: str) -> tuple[int, str, bool, str]:
        ok, info = generate_one(cfg, out_dir, cat, seed=args.seed + i, verbose=args.verbose)
        return i, cat, ok, info

    if args.concurrency <= 1:
        results = [task(i, cat) for i, cat in enumerate(plan)]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(task, i, cat) for i, cat in enumerate(plan)]
            for fut in as_completed(futures):
                results.append(fut.result())

    for _, cat, ok, info in results:
        stats.by_category.setdefault(cat, 0)
        if ok:
            stats.succeeded += 1
            stats.by_category[cat] += 1
            print(f"  ✓ {cat}: {info}", file=sys.stderr)
        else:
            stats.failed += 1
            print(f"  ✗ {cat}: {info}", file=sys.stderr)

    print(stats.report(), file=sys.stderr)
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
