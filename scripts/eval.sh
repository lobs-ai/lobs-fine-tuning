#!/usr/bin/env bash
#
# eval.sh — held-out tool-call smoke eval. Runs both base and adapter and
# prints a side-by-side diff. The headline number is `exact_tool_match`:
# how often the trained model picks the same tool the teacher picked.
#
# Usage:
#   ./scripts/eval.sh                                  # uses $MODEL + artifacts/adapter
#   ./scripts/eval.sh --model Qwen/Qwen3-8B-Instruct
#   ./scripts/eval.sh --adapter-dir artifacts/run42/adapter --n 50
#   ./scripts/eval.sh --base-only                      # skip adapter (sanity check)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-artifacts/adapter}"
PROMPTS="${PROMPTS:-data/val.messages.jsonl}"
N="${N:-20}"
REPORT_OUT="${REPORT_OUT:-artifacts/eval_smoke.json}"
EXTRA=()

usage() {
    cat <<EOF
Usage: $0 [flags]

Common:
  --model NAME           HF base model id              [\$MODEL=$MODEL]
  --adapter-dir PATH     LoRA adapter dir              [\$ADAPTER_DIR=$ADAPTER_DIR]
  --prompts PATH         messages JSONL                [\$PROMPTS=$PROMPTS]
  --n N                  prompts to evaluate           [\$N=$N]
  --report-out PATH      JSON output                   [\$REPORT_OUT=$REPORT_OUT]
  --base-only            skip adapter pass
  --no-4bit              full precision (more VRAM, faster on H100)
  -h | --help            show this help

Anything else is forwarded to eval.smoke (e.g. -v).

Examples:
  $0
  $0 --n 50 -v
  $0 --adapter-dir artifacts/run-qwen3/adapter
  $0 --base-only             # run base model only — sanity check
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --adapter-dir)  ADAPTER_DIR="$2"; shift 2 ;;
        --prompts)      PROMPTS="$2"; shift 2 ;;
        --n)            N="$2"; shift 2 ;;
        --report-out)   REPORT_OUT="$2"; shift 2 ;;
        --base-only)    EXTRA+=(--base-only); shift ;;
        --no-4bit)      EXTRA+=(--no-4bit); shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              EXTRA+=("$1"); shift ;;
    esac
done

say() { printf '\033[1;36m▸ %s\033[0m\n' "$*" >&2; }

if [[ ! -x "$PYTHON" ]]; then
    echo "PYTHON='$PYTHON' is not executable. Set PYTHON= or create a venv." >&2
    exit 2
fi
if [[ ! -f "$PROMPTS" ]]; then
    echo "no prompts file at $PROMPTS — run ./scripts/build_dataset.sh first" >&2
    exit 2
fi

args=(--model "$MODEL" --prompts "$PROMPTS" --n "$N" --report-out "$REPORT_OUT")
if [[ -d "$ADAPTER_DIR" ]]; then
    args+=(--adapter-dir "$ADAPTER_DIR")
else
    say "adapter dir $ADAPTER_DIR not found — running base-only"
    args+=(--base-only)
fi

exec "$PYTHON" -m eval.smoke "${args[@]}" "${EXTRA[@]}"
