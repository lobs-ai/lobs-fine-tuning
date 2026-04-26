#!/usr/bin/env bash
#
# train.sh — single-command training launcher.
#
# Sources .env, (optionally) rebuilds the dataset for the chosen MODEL,
# then calls `python -m train.train --model "$MODEL"`. Pass any extra args
# through to train.train.
#
# Usage:
#   ./scripts/train.sh                                  # defaults from .env
#   ./scripts/train.sh --model Qwen/Qwen3-8B-Instruct
#   ./scripts/train.sh --skip-build                     # skip dataset rebuild
#   MODEL=meta-llama/Llama-3.1-8B-Instruct ./scripts/train.sh
#   ./scripts/train.sh --lora-r 32 --epochs 3           # forwarded to train.train

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Source .env if present (so MODEL, HF_TOKEN, WANDB_API_KEY, etc. are available)
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SKIP_BUILD=0
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 [--model NAME] [--skip-build] [-- <train.train flags>...]

Common:
  --model NAME       HF model id (default: \$MODEL=$MODEL)
  --skip-build       skip the build_dataset.sh rebuild step
  -h | --help        show this message

Anything after --model/--skip-build is forwarded to train.train.

Examples:
  $0
  $0 --model Qwen/Qwen3-8B-Instruct
  $0 --model meta-llama/Llama-3.1-8B-Instruct --skip-build
  $0 --lora-r 32 --epochs 3 --output-dir artifacts/run2
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

say() { printf '\033[1;36m▸ %s\033[0m\n' "$*" >&2; }

if [[ ! -x "$PYTHON" ]]; then
    echo "PYTHON='$PYTHON' is not executable. Set PYTHON= or create a venv." >&2
    exit 2
fi

# Quick CUDA preflight — Unsloth needs CUDA. Don't hard-fail (CPU/MPS smoke
# tests sometimes useful), but make it loud.
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    say "WARNING: torch.cuda.is_available() is False — Unsloth requires CUDA."
    say "         If you're on Mac, run this on your GPU host instead."
fi

if [[ "$SKIP_BUILD" == "0" ]]; then
    say "rebuilding dataset for $MODEL"
    MODEL="$MODEL" ./scripts/build_dataset.sh
else
    say "[skip] dataset rebuild"
fi

say "starting training: $MODEL"
exec "$PYTHON" -m train.train --model "$MODEL" "${EXTRA_ARGS[@]}"
