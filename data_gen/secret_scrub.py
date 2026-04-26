"""
Strip / redact secrets from canonical trajectories before training.

We never want a secret token in the model's weights. Three layers of
protection, in order of trust:

1. **Allowlist of placeholders.** If a value already looks like a
   placeholder (`<REDACTED>`, `${ENV_VAR}`, `xxx-xxxx-xxxx`), leave it
   alone — it's been pre-scrubbed upstream.
2. **Known-keys.** Squad's config tools (`set_config`, `get_config`) have
   a small set of dot-paths that are always sensitive
   (`auth.tokens.*`, `llm.providers.*.apiKey`, etc). Tool-call arguments
   and tool-results that touch these paths get the value field replaced
   with `<REDACTED:reason>`.
3. **Pattern scan over every text-like field.** Catch leaked
   API keys (sk-…, claude-, ghp_…), bearer tokens, AWS access keys,
   private-key blocks, JWTs, .env-style assignments, and the *literal*
   secret values found in the user's local `~/.squad/config` (so we don't
   accidentally bake the user's actual SQUAD_TOKEN into a training corpus).

Trajectories where scrubbing changes a tool-call's *argument structure*
(e.g., a key entirely vanished after redaction making the call invalid)
are dropped — we don't want the model learning malformed calls.

Usage:
    python -m data_gen.secret_scrub \
      --in data/trajectories_canonical/all.jsonl \
      --out data/trajectories_canonical/all.scrubbed.jsonl \
      --report data/_scrub_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_gen.squad_config import load_squad_config
from data_gen.trajectory import (
    AssistantTurn,
    ToolCall,
    ToolResultTurn,
    Trajectory,
    UserTurn,
    read_jsonl,
    write_jsonl,
)

REDACTED_TOKEN = "<REDACTED:{reason}>"


# ── pattern catalog ─────────────────────────────────────────────────────────
#
# Each entry is (reason, regex). Reasons feed both the redaction marker and
# the report so we can audit what got hit. Patterns are intentionally
# conservative — a few false positives are acceptable, false negatives are
# not.

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Anthropic
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    # OpenAI (project + classic)
    ("openai_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    # GitHub PATs / fine-grained / oauth / app
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}\b")),
    # AWS access key id
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # AWS secret access key (heuristic: 40 char base64-ish near "aws_secret")
    (
        "aws_secret_assignment",
        re.compile(r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key\s*[=:]\s*['\"]?[A-Za-z0-9/+]{40}['\"]?"),
    ),
    # Slack bot tokens
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    # Generic Bearer header
    ("bearer_header", re.compile(r"(?i)\bauthorization:\s*bearer\s+[A-Za-z0-9._\-]{16,}")),
    # JWT (header.payload.signature)
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    # PEM private key blocks
    (
        "pem_private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]+?-----END [^-]+-----"),
    ),
    # SSH private key one-liner remnants
    ("ssh_private_keyblob", re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----[A-Za-z0-9+/=\s]+")),
    # .env-style secret assignment (KEY=VAL where KEY contains TOKEN/SECRET/KEY/PASSWORD)
    (
        "env_secret_assignment",
        re.compile(
            r"(?im)^[\t ]*(?:export\s+)?[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD|PASSWD|API_KEY)[A-Z0-9_]*\s*=\s*['\"]?[A-Za-z0-9._\-/+=]{8,}['\"]?"
        ),
    ),
    # Squad-token-shape fallback (URL-safe base64, 32–64 chars, looks token-y)
    ("urlsafe_b64_token", re.compile(r"\b[A-Za-z0-9_\-]{32,64}\b")),
]


# Anything matching these fragments is presumed already-redacted; leave alone.
PLACEHOLDER_PATTERNS = [
    re.compile(r"^<REDACTED"),
    re.compile(r"^\$\{[A-Z][A-Z0-9_]*\}$"),
    re.compile(r"^xxx[\-x]+$", re.IGNORECASE),
    re.compile(r"^\*+$"),
]


# ── known sensitive config paths (squad-specific) ───────────────────────────


SENSITIVE_CONFIG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^auth\.tokens($|\.)"),
    re.compile(r"^llm\.providers\.[^.]+\.apiKey$"),
    re.compile(r"^llm\.providers\.[^.]+\.authToken$"),
    re.compile(r"(?i)\b(api[_\-]?key|apikey|secret|token|password|passwd)\b"),
]


def _is_sensitive_config_path(path: str) -> bool:
    return any(p.search(path) for p in SENSITIVE_CONFIG_PATTERNS)


# ── live-secret seeding from the user's own squad config ────────────────────


def _live_secret_literals() -> set[str]:
    """
    Pull the user's actual secrets out of $SQUAD_HOME/config and from
    standard env vars so we can scrub them by exact match. This catches
    cases where the model echoed a real token verbatim — the regexes
    above might miss a custom-shape one.
    """
    seeds: set[str] = set()
    sq = load_squad_config()
    for k, v in sq.extras.items():
        if not v:
            continue
        if any(needle in k.upper() for needle in ("TOKEN", "SECRET", "KEY", "PASSWORD")):
            seeds.add(v)
    for env_key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "WANDB_API_KEY",
        "HF_TOKEN",
        "GITHUB_TOKEN",
        "SQUAD_TOKEN",
    ):
        v = os.environ.get(env_key)
        if v:
            seeds.add(v)
    # Drop too-short / non-secret-shaped strings to avoid scrubbing common words.
    return {s for s in seeds if len(s) >= 12}


# ── scrubbing primitives ────────────────────────────────────────────────────


@dataclass
class ScrubReport:
    trajectories_seen: int = 0
    trajectories_modified: int = 0
    trajectories_dropped: int = 0
    hits: Counter[str] = field(default_factory=Counter)
    drop_reasons: Counter[str] = field(default_factory=Counter)

    def summary(self) -> dict[str, Any]:
        return {
            "trajectories_seen": self.trajectories_seen,
            "trajectories_modified": self.trajectories_modified,
            "trajectories_dropped": self.trajectories_dropped,
            "hits": dict(self.hits),
            "drop_reasons": dict(self.drop_reasons),
        }


def _is_placeholder(value: str) -> bool:
    return any(p.search(value) for p in PLACEHOLDER_PATTERNS)


def scrub_text(text: str, live_secrets: set[str], report: ScrubReport) -> str:
    """Run pattern + literal scrubbing over a single text field."""
    if not text or _is_placeholder(text.strip()):
        return text
    out = text

    # Literal exact-match secrets first — strongest signal.
    for secret in live_secrets:
        if secret and secret in out:
            out = out.replace(secret, REDACTED_TOKEN.format(reason="live_secret"))
            report.hits["live_secret"] += 1

    for reason, pattern in PATTERNS:
        # Skip the broad fallback if a tighter pattern already redacted everything.
        replacement = REDACTED_TOKEN.format(reason=reason)

        def _sub(m: re.Match[str]) -> str:
            report.hits[reason] += 1
            return replacement

        new_out = pattern.sub(_sub, out)
        if reason == "urlsafe_b64_token" and new_out != out:
            # Only accept the broad token catch when the surrounding context
            # looks token-y; otherwise we'd shred random base64 strings.
            if not _looks_token_context(out, pattern):
                continue
        out = new_out
    return out


_TOKEN_CONTEXT_HINTS = re.compile(
    r"(?i)\b(token|key|secret|bearer|authorization|api[_\-]?key|apikey)\b"
)


def _looks_token_context(text: str, pattern: re.Pattern[str]) -> bool:
    for m in pattern.finditer(text):
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        if _TOKEN_CONTEXT_HINTS.search(text[start:end]):
            return True
    return False


def scrub_value(value: Any, live_secrets: set[str], report: ScrubReport) -> Any:
    """Recursively scrub a JSON-like structure (dict / list / str / scalar)."""
    if isinstance(value, str):
        return scrub_text(value, live_secrets, report)
    if isinstance(value, list):
        return [scrub_value(v, live_secrets, report) for v in value]
    if isinstance(value, dict):
        return {k: scrub_value(v, live_secrets, report) for k, v in value.items()}
    return value


def scrub_tool_call_args(call: ToolCall, live_secrets: set[str], report: ScrubReport) -> bool:
    """
    Scrub a single tool call's arguments in-place. Returns False if the
    structure was mutated in a way that invalidates the call (caller drops
    the trajectory). Currently we only invalidate when set_config is given
    a sensitive path — we redact the value but keep the call shape, so
    this always returns True today; the hook is here for future hard-drops.
    """
    args = call.arguments or {}

    if call.name in ("set_config", "get_config", "list_config_paths"):
        path = args.get("path")
        if isinstance(path, str) and _is_sensitive_config_path(path):
            if "value" in args:
                args["value"] = REDACTED_TOKEN.format(reason="sensitive_config_path")
                report.hits["sensitive_config_path"] += 1

    call.arguments = scrub_value(args, live_secrets, report)
    return True


def scrub_trajectory(traj: Trajectory, live_secrets: set[str], report: ScrubReport) -> Trajectory | None:
    """
    Walk every text-bearing field. Returns the scrubbed trajectory, or
    None if it's now unsafe to keep (kept here for future invalidation
    rules — e.g., if a system prompt hits a private-key block we should
    drop the whole trajectory rather than redact).
    """
    original = json.dumps(traj.to_dict(), sort_keys=True)

    if traj.meta.system_prompt:
        traj.meta.system_prompt = scrub_text(traj.meta.system_prompt, live_secrets, report)

    for turn in traj.turns:
        if isinstance(turn, UserTurn):
            turn.text = scrub_text(turn.text, live_secrets, report)
        elif isinstance(turn, AssistantTurn):
            turn.text = scrub_text(turn.text, live_secrets, report)
            for call in turn.tool_calls:
                if not scrub_tool_call_args(call, live_secrets, report):
                    return None
        elif isinstance(turn, ToolResultTurn):
            for r in turn.results:
                r.content = scrub_text(r.content, live_secrets, report)

    if "BEGIN PRIVATE KEY" in original or "BEGIN OPENSSH PRIVATE KEY" in original:
        # A private key block leaked into the trajectory. Pattern scrubbing
        # will have replaced the body, but the surrounding context is still
        # suspect — drop entirely. Cheaper than auditing.
        report.drop_reasons["private_key_in_trajectory"] += 1
        return None

    if json.dumps(traj.to_dict(), sort_keys=True) != original:
        report.trajectories_modified += 1
    return traj


# ── orchestrator ────────────────────────────────────────────────────────────


def scrub_all(
    trajectories: list[Trajectory],
    live_secrets: set[str] | None = None,
) -> tuple[list[Trajectory], ScrubReport]:
    if live_secrets is None:
        live_secrets = _live_secret_literals()
    report = ScrubReport()
    out: list[Trajectory] = []
    for traj in trajectories:
        report.trajectories_seen += 1
        kept = scrub_trajectory(traj, live_secrets, report)
        if kept is None:
            report.trajectories_dropped += 1
            continue
        out.append(kept)
    return out, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Redact secrets from canonical trajectories.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--report", dest="report_path", help="optional JSON report path")
    parser.add_argument(
        "--no-live-secrets",
        action="store_true",
        help="Skip seeding from $SQUAD_HOME/config + env vars (regex-only).",
    )
    parser.add_argument(
        "--extra-secret",
        action="append",
        default=[],
        help="Add a literal string to scrub. Can be passed multiple times.",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    trajectories = read_jsonl(in_path)
    live = set() if args.no_live_secrets else _live_secret_literals()
    live.update(s for s in args.extra_secret if len(s) >= 8)

    scrubbed, report = scrub_all(trajectories, live)
    write_jsonl(out_path, scrubbed)

    summary = report.summary()
    print(json.dumps(summary, indent=2), file=sys.stderr)
    if args.report_path:
        Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_path).write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(scrubbed)} → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
