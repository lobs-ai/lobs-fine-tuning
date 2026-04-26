#!/usr/bin/env bash
#
# build_dataset.sh — end-to-end data pipeline.
#
# Walks every stage from raw squad transcripts to a tokenized Qwen training
# set, with explicit params for every knob. Defaults are chosen so the
# happy path is just `./scripts/build_dataset.sh`.
#
# Stages:
#   1. harvest  — pull transcripts from sessions dir + gateway DB + manual dumps
#   2. scrub    — redact secrets BEFORE filtering (do not skip)
#   3. filter   — drop malformed / looping / unterminated trajectories
#   4. format   — apply Qwen chat template + loss masking → train/val JSONL
#
# Usage:
#   ./scripts/build_dataset.sh                            # all defaults
#   ./scripts/build_dataset.sh --skip-format              # stop before tokenizer
#   SESSIONS_DIR=~/other/sessions ./scripts/build_dataset.sh
#
# Every flag is also settable via env var (the env var name is the same
# as the long flag in UPPER_SNAKE).

set -euo pipefail

# ── repo root ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── defaults ─────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

# Sources
SESSIONS_DIR="${SESSIONS_DIR:-${SQUAD_TRANSCRIPTS_DIR:-$HOME/.lobs/agents}}"
FULL_DIR="${FULL_DIR:-data/trajectories_raw/full}"
MANUAL_DIR="${MANUAL_DIR:-data/trajectories_raw/manual}"
SQUAD_NAME="${SQUAD_NAME:-default}"
DB_PATH="${DB_PATH:-}"  # empty → resolve from $SQUAD_HOME/$SQUAD_NAME/squad.db

# Stage outputs
CANONICAL_PATH="${CANONICAL_PATH:-data/trajectories_canonical/all.jsonl}"
SCRUBBED_PATH="${SCRUBBED_PATH:-data/trajectories_canonical/all.scrubbed.jsonl}"
SCRUB_REPORT="${SCRUB_REPORT:-data/_scrub_report.json}"
FILTERED_PATH="${FILTERED_PATH:-data/trajectories_filtered/all.jsonl}"
TRAIN_PATH="${TRAIN_PATH:-data/train.jsonl}"
VAL_PATH="${VAL_PATH:-data/val.jsonl}"
TEXT_INSPECT_PATH="${TEXT_INSPECT_PATH:-data/_inspect.jsonl}"

# Knobs
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-7B-Instruct}"
MAX_LEN="${MAX_LEN:-8192}"
VAL_FRAC="${VAL_FRAC:-0.05}"
SEED="${SEED:-42}"
MAX_TURNS="${MAX_TURNS:-15}"
STRICT_TOOLS="${STRICT_TOOLS:-0}"   # 1 = drop any traj that touched non-allowed tools
NO_LIVE_SECRETS="${NO_LIVE_SECRETS:-0}"  # 1 = regex-only scrub, skip $SQUAD_HOME seeds
EXTRA_SECRETS=()                    # filled by repeated --extra-secret=...

# Stage skips
SKIP_HARVEST=0
SKIP_SCRUB=0
SKIP_FILTER=0
SKIP_FORMAT=0
VERIFY_FORMAT=1

usage() {
    cat <<EOF
Usage: $0 [flags]

Sources (where transcripts come from):
  --sessions-dir PATH     per-agent jsonl root         [\$SESSIONS_DIR=$SESSIONS_DIR]
  --full-dir PATH         full Anthropic-format dumps  [\$FULL_DIR=$FULL_DIR]
  --manual-dir PATH       hand-exported sessions       [\$MANUAL_DIR=$MANUAL_DIR]
  --squad-name NAME       which gateway DB to read     [\$SQUAD_NAME=$SQUAD_NAME]
  --db-path PATH          override gateway DB lookup   [\$DB_PATH (auto)]

Stage outputs:
  --canonical PATH        harvest output               [\$CANONICAL_PATH=$CANONICAL_PATH]
  --scrubbed PATH         scrub output                 [\$SCRUBBED_PATH=$SCRUBBED_PATH]
  --scrub-report PATH     scrub stats JSON             [\$SCRUB_REPORT=$SCRUB_REPORT]
  --filtered PATH         filter output                [\$FILTERED_PATH=$FILTERED_PATH]
  --train PATH            train tokenized JSONL        [\$TRAIN_PATH=$TRAIN_PATH]
  --val PATH              val tokenized JSONL          [\$VAL_PATH=$VAL_PATH]
  --inspect PATH          human-readable rendered text [\$TEXT_INSPECT_PATH=$TEXT_INSPECT_PATH]

Knobs:
  --tokenizer NAME        HF tokenizer id              [\$TOKENIZER=$TOKENIZER]
  --max-len N             drop sequences > N tokens    [\$MAX_LEN=$MAX_LEN]
  --val-frac F            fraction held back           [\$VAL_FRAC=$VAL_FRAC]
  --seed N                rng seed                     [\$SEED=$SEED]
  --max-turns N           filter §8 turn cap           [\$MAX_TURNS=$MAX_TURNS]
  --strict-tools          drop any traj touching out-of-set tools
  --no-live-secrets       regex-only scrub (skip \$SQUAD_HOME seeds)
  --extra-secret STR      add literal string to redact (repeatable)
  --no-verify             skip the format round-trip check (NOT recommended)

Stage skips:
  --skip-harvest --skip-scrub --skip-filter --skip-format

Run with --help to see this. Pass --dry-run to print commands without executing.
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sessions-dir)   SESSIONS_DIR="$2"; shift 2 ;;
        --full-dir)       FULL_DIR="$2"; shift 2 ;;
        --manual-dir)     MANUAL_DIR="$2"; shift 2 ;;
        --squad-name)     SQUAD_NAME="$2"; shift 2 ;;
        --db-path)        DB_PATH="$2"; shift 2 ;;
        --canonical)      CANONICAL_PATH="$2"; shift 2 ;;
        --scrubbed)       SCRUBBED_PATH="$2"; shift 2 ;;
        --scrub-report)   SCRUB_REPORT="$2"; shift 2 ;;
        --filtered)       FILTERED_PATH="$2"; shift 2 ;;
        --train)          TRAIN_PATH="$2"; shift 2 ;;
        --val)            VAL_PATH="$2"; shift 2 ;;
        --inspect)        TEXT_INSPECT_PATH="$2"; shift 2 ;;
        --tokenizer)      TOKENIZER="$2"; shift 2 ;;
        --max-len)        MAX_LEN="$2"; shift 2 ;;
        --val-frac)       VAL_FRAC="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --max-turns)      MAX_TURNS="$2"; shift 2 ;;
        --strict-tools)   STRICT_TOOLS=1; shift ;;
        --no-live-secrets) NO_LIVE_SECRETS=1; shift ;;
        --extra-secret)   EXTRA_SECRETS+=("$2"); shift 2 ;;
        --no-verify)      VERIFY_FORMAT=0; shift ;;
        --skip-harvest)   SKIP_HARVEST=1; shift ;;
        --skip-scrub)     SKIP_SCRUB=1; shift ;;
        --skip-filter)    SKIP_FILTER=1; shift ;;
        --skip-format)    SKIP_FORMAT=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

# ── helpers ──────────────────────────────────────────────────────────────────
say()    { printf '\033[1;36m▸ %s\033[0m\n' "$*" >&2; }
hr()     { printf '\033[2m─────────────────────────────────────────────\033[0m\n' >&2; }
run() {
    say "$ $*"
    if [[ "$DRY_RUN" == "0" ]]; then
        "$@"
    fi
}

# ── preflight ────────────────────────────────────────────────────────────────
if [[ ! -x "$PYTHON" ]]; then
    echo "PYTHON='$PYTHON' is not executable. Set PYTHON= or create a venv:" >&2
    echo "  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
    exit 2
fi

mkdir -p \
    "$(dirname "$CANONICAL_PATH")" \
    "$(dirname "$SCRUBBED_PATH")" \
    "$(dirname "$FILTERED_PATH")" \
    "$(dirname "$TRAIN_PATH")" \
    "$(dirname "$VAL_PATH")"

hr
say "build_dataset.sh — pipeline starting"
say "  python:        $PYTHON"
say "  sessions-dir:  $SESSIONS_DIR"
say "  full-dir:      $FULL_DIR"
say "  manual-dir:    $MANUAL_DIR"
say "  squad-name:    $SQUAD_NAME"
say "  db-path:       ${DB_PATH:-<auto from \$SQUAD_HOME>}"
say "  tokenizer:     $TOKENIZER (max-len $MAX_LEN, val-frac $VAL_FRAC)"
hr

# ── stage 1: harvest ─────────────────────────────────────────────────────────
if [[ "$SKIP_HARVEST" == "0" ]]; then
    args=(
        --sessions-dir "$SESSIONS_DIR"
        --full-dir "$FULL_DIR"
        --manual-dir "$MANUAL_DIR"
        --squad-name "$SQUAD_NAME"
        --out "$CANONICAL_PATH"
    )
    [[ -n "$DB_PATH" ]]           && args+=(--db-path "$DB_PATH")
    [[ "$STRICT_TOOLS" == "1" ]]  && args+=(--strict-tools)
    run "$PYTHON" -m data_gen.harvester "${args[@]}"
else
    say "[skip] harvest"
fi

# ── stage 2: scrub ───────────────────────────────────────────────────────────
if [[ "$SKIP_SCRUB" == "0" ]]; then
    args=(
        --in "$CANONICAL_PATH"
        --out "$SCRUBBED_PATH"
        --report "$SCRUB_REPORT"
    )
    [[ "$NO_LIVE_SECRETS" == "1" ]] && args+=(--no-live-secrets)
    for s in "${EXTRA_SECRETS[@]:-}"; do
        [[ -n "$s" ]] && args+=(--extra-secret "$s")
    done
    run "$PYTHON" -m data_gen.secret_scrub "${args[@]}"
    say "scrub report: $SCRUB_REPORT"
else
    say "[skip] scrub — DO NOT skip on real data"
    SCRUBBED_PATH="$CANONICAL_PATH"
fi

# ── stage 3: filter ──────────────────────────────────────────────────────────
if [[ "$SKIP_FILTER" == "0" ]]; then
    run "$PYTHON" -m data_gen.filter \
        --in "$SCRUBBED_PATH" \
        --out "$FILTERED_PATH" \
        --max-turns "$MAX_TURNS"
else
    say "[skip] filter"
    FILTERED_PATH="$SCRUBBED_PATH"
fi

# ── stage 4: format ──────────────────────────────────────────────────────────
if [[ "$SKIP_FORMAT" == "0" ]]; then
    args=(
        --in "$FILTERED_PATH"
        --train-out "$TRAIN_PATH"
        --val-out "$VAL_PATH"
        --text-out "$TEXT_INSPECT_PATH"
        --tokenizer "$TOKENIZER"
        --max-len "$MAX_LEN"
        --val-frac "$VAL_FRAC"
        --seed "$SEED"
    )
    [[ "$VERIFY_FORMAT" == "1" ]] && args+=(--verify)
    run "$PYTHON" -m data_gen.format_for_training "${args[@]}"
else
    say "[skip] format"
fi

hr
say "done."
say "  scrubbed:  $SCRUBBED_PATH"
say "  filtered:  $FILTERED_PATH"
[[ "$SKIP_FORMAT" == "0" ]] && say "  train:     $TRAIN_PATH"
[[ "$SKIP_FORMAT" == "0" ]] && say "  val:       $VAL_PATH"
[[ "$SKIP_FORMAT" == "0" ]] && say "  inspect:   $TEXT_INSPECT_PATH"
hr
