#!/usr/bin/env bash
#
# generate_data.sh — bulk synthesize trajectories via the Gemini CLI, then
# (optionally) chain straight into build_dataset.sh so they're ready to train on.
#
# Each invocation produces N trajectories under data/trajectories_raw/full/,
# which is exactly where the existing harvester looks. After generation,
# build_dataset.sh runs the harvest → scrub → filter → format pipeline,
# leaving you with data/train.jsonl ready for `scripts/train.sh`.
#
# Usage:
#   ./scripts/generate_data.sh                            # 50 trajectories, mixed
#   ./scripts/generate_data.sh --n 200 --concurrency 6
#   ./scripts/generate_data.sh --category debugging --n 30
#   ./scripts/generate_data.sh --skip-build               # generate only
#   ./scripts/generate_data.sh --gemini-bin /opt/gemini --model gemini-2.5-pro
#
# Requirements:
#   - gemini CLI installed and authenticated (npm i -g @google/gemini-cli, then `gemini` once)
#   - $GEMINI_API_KEY in .env if your gemini setup requires it

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
N="${N:-50}"
CATEGORY=""
MIX=()
OUT_DIR="${OUT_DIR:-data/trajectories_raw/full}"
GEMINI_BIN="${GEMINI_BIN:-gemini}"
GEMINI_MODEL="${GEMINI_MODEL:-}"
GEMINI_ARGS=()
TIMEOUT="${GEMINI_TIMEOUT:-180}"
CONCURRENCY="${CONCURRENCY:-4}"
SEED="${SEED:-$(date +%s)}"
SKIP_BUILD=0
DRY_RUN=0

usage() {
    cat <<EOF
Usage: $0 [flags]

Generation:
  --n N                  how many trajectories to generate    [\$N=$N]
  --category NAME        focus on a single category (else mix all known)
  --mix CAT [CAT ...]    explicit categories to rotate over
  --out-dir PATH         output dir                            [\$OUT_DIR=$OUT_DIR]
  --concurrency N        parallel gemini calls                 [\$CONCURRENCY=$CONCURRENCY]
  --seed N               rng seed for diversity nudges

Gemini:
  --gemini-bin PATH      gemini binary                         [\$GEMINI_BIN=$GEMINI_BIN]
  --model NAME           gemini model (-m flag)                [\$GEMINI_MODEL=${GEMINI_MODEL:-<auto>}]
  --gemini-arg STR       extra arg forwarded to gemini (repeatable)
  --timeout N            per-call timeout in seconds           [\$GEMINI_TIMEOUT=$TIMEOUT]

Pipeline:
  --skip-build           skip the build_dataset.sh step (generate only)
  --dry-run              print the generation command and exit
  -h | --help            show this help

Examples:
  $0 --n 100
  $0 --n 30 --category debugging
  $0 --mix file_edit debugging multi_file_refactor --n 60
  $0 --n 200 --concurrency 8 --model gemini-2.5-pro
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)            N="$2"; shift 2 ;;
        --category)     CATEGORY="$2"; shift 2 ;;
        --mix)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MIX+=("$1"); shift
            done
            ;;
        --out-dir)      OUT_DIR="$2"; shift 2 ;;
        --gemini-bin)   GEMINI_BIN="$2"; shift 2 ;;
        --model)        GEMINI_MODEL="$2"; shift 2 ;;
        --gemini-arg)   GEMINI_ARGS+=("$2"); shift 2 ;;
        --timeout)      TIMEOUT="$2"; shift 2 ;;
        --concurrency)  CONCURRENCY="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --skip-build)   SKIP_BUILD=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

say() { printf '\033[1;36m▸ %s\033[0m\n' "$*" >&2; }
hr()  { printf '\033[2m─────────────────────────────────────────────\033[0m\n' >&2; }

if [[ ! -x "$PYTHON" ]]; then
    echo "PYTHON='$PYTHON' is not executable. Set PYTHON= or create a venv." >&2
    exit 2
fi

mkdir -p "$OUT_DIR"

# Build generation argv
gen_args=(--n "$N" --out-dir "$OUT_DIR" --concurrency "$CONCURRENCY" --seed "$SEED" --timeout "$TIMEOUT" --gemini-bin "$GEMINI_BIN")
[[ -n "$GEMINI_MODEL" ]] && gen_args+=(--model "$GEMINI_MODEL")
[[ -n "$CATEGORY" ]]      && gen_args+=(--category "$CATEGORY")
if [[ ${#MIX[@]} -gt 0 ]]; then
    gen_args+=(--mix "${MIX[@]}")
fi
for ga in "${GEMINI_ARGS[@]:-}"; do
    [[ -n "$ga" ]] && gen_args+=(--gemini-arg "$ga")
done

hr
say "generating $N trajectories → $OUT_DIR"
say "  gemini:      $GEMINI_BIN${GEMINI_MODEL:+ ($GEMINI_MODEL)}"
say "  concurrency: $CONCURRENCY  timeout=${TIMEOUT}s  seed=$SEED"
[[ -n "$CATEGORY" ]]    && say "  category:    $CATEGORY"
[[ ${#MIX[@]} -gt 0 ]]  && say "  mix:         ${MIX[*]}"
hr

cmd=("$PYTHON" -m data_gen.synth "${gen_args[@]}")
if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}" >&2; echo >&2
    exit 0
fi

"${cmd[@]}"

hr
if [[ "$SKIP_BUILD" == "0" ]]; then
    say "running build_dataset.sh on the new trajectories"
    ./scripts/build_dataset.sh
else
    say "[skip] build_dataset.sh — run it yourself when ready"
fi
hr
say "done."
