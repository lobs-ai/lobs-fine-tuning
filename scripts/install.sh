#!/usr/bin/env bash
#
# install.sh — set up a Python venv with the right extras for the chosen mode.
#
# Modes (one venv per mode is fine; pick what the host is for):
#   local — base + dev. For data generation + tests on Mac/Linux.
#   gpu   — base + train (Unsloth, bitsandbytes, torch+CUDA). Linux only.
#   eval  — base + eval (vLLM, httpx). For LoRA-served inference.
#   all   — everything (Linux GPU host).
#
# Usage:
#   ./scripts/install.sh                       # default: --mode local
#   ./scripts/install.sh --mode gpu            # on the GPU host
#   ./scripts/install.sh --mode all --upgrade  # reinstall and bump
#   ./scripts/install.sh --venv .venv-gpu      # use a separate venv path
#
# What it does:
#   1. Creates the venv (skips if present).
#   2. Upgrades pip/wheel/setuptools.
#   3. pip install -e with the matching extras from pyproject.toml.
#   4. Copies .env.example → .env if missing.
#   5. Prints next-step hints (gemini CLI install, smoke test, etc.).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f pyproject.toml ]]; then
    echo "pyproject.toml not found — are you in the repo root?" >&2
    exit 2
fi

MODE="local"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${VENV:-.venv}"
UPGRADE=0

usage() {
    cat <<EOF
Usage: $0 [flags]

Modes:
  --mode local       base + dev — for data gen, tests, Mac dev (default)
  --mode gpu         base + train — Linux GPU host (Unsloth + CUDA torch)
  --mode eval        base + eval — for vLLM inference / eval harness
  --mode all         everything (only on Linux GPU host)

Other flags:
  --python BIN       python interpreter to seed the venv [\$PYTHON_BIN=$PYTHON_BIN]
  --venv PATH        venv path                            [\$VENV=$VENV]
  --upgrade          reinstall packages, bumping versions
  -h | --help        show this help

Environment overrides: PYTHON_BIN, VENV both work as env vars.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)     MODE="$2"; shift 2 ;;
        --python)   PYTHON_BIN="$2"; shift 2 ;;
        --venv)     VENV="$2"; shift 2 ;;
        --upgrade)  UPGRADE=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

case "$MODE" in
    local|gpu|eval|all) ;;
    *) echo "bad --mode: $MODE (want local|gpu|eval|all)" >&2; exit 2 ;;
esac

say() { printf '\033[1;36m▸ %s\033[0m\n' "$*" >&2; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*" >&2; }
ok() { printf '\033[1;32m✓ %s\033[0m\n' "$*" >&2; }

# ── refuse GPU/all on Mac (bitsandbytes / unsloth need CUDA) ────────────────
if [[ "$MODE" == "gpu" || "$MODE" == "all" ]]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "✗ --mode $MODE requires CUDA; macOS isn't supported." >&2
        echo "  On Mac, use: $0 --mode local" >&2
        echo "  Run --mode gpu on your remote GPU host (RunPod, Lambda, etc.)." >&2
        exit 2
    fi
fi

# ── venv ────────────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    say "creating venv at $VENV (python: $PYTHON_BIN)"
    "$PYTHON_BIN" -m venv "$VENV"
else
    say "venv exists at $VENV"
fi

PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

# Some platforms ship venv pythons without ensurepip; bootstrap if missing.
if [[ ! -x "$PIP" ]]; then
    "$PYTHON" -m ensurepip --upgrade
fi

PIP_FLAGS=(install --upgrade)
[[ "$UPGRADE" == "1" ]] && PIP_FLAGS+=(--upgrade-strategy eager)

say "upgrading pip toolchain"
"$PIP" "${PIP_FLAGS[@]}" pip wheel setuptools

# ── extras ──────────────────────────────────────────────────────────────────
case "$MODE" in
    local) extras="dev" ;;
    gpu)   extras="train,dev" ;;
    eval)  extras="eval,dev" ;;
    all)   extras="train,eval,dev" ;;
esac

INSTALL_FLAGS=(install -e ".[${extras}]")
[[ "$UPGRADE" == "1" ]] && INSTALL_FLAGS+=(--upgrade --upgrade-strategy eager)

say "installing project: extras=[$extras]"
"$PIP" "${INSTALL_FLAGS[@]}"

# ── .env scaffold ───────────────────────────────────────────────────────────
if [[ ! -f .env && -f .env.example ]]; then
    cp .env.example .env
    say ".env scaffolded from .env.example — edit it with real keys"
fi

# ── gemini CLI hint (for local/all modes) ───────────────────────────────────
if [[ "$MODE" == "local" || "$MODE" == "all" ]]; then
    if ! command -v gemini >/dev/null 2>&1 && [[ -z "${GEMINI_BIN:-}" || ! -x "${GEMINI_BIN:-}" ]]; then
        warn "gemini CLI not found on PATH (data gen will fail until installed)"
        if command -v npm >/dev/null 2>&1; then
            echo "    install:  npm i -g @google/gemini-cli && gemini  # log in once" >&2
        else
            echo "    npm not installed — install Node.js first, then: npm i -g @google/gemini-cli" >&2
        fi
    else
        ok "gemini CLI found"
    fi
fi

# ── GPU smoke check (warning, not error) ────────────────────────────────────
if [[ "$MODE" == "gpu" || "$MODE" == "all" ]]; then
    if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        warn "torch installed but CUDA is not visible — training will fail."
        warn "  check: nvidia-smi; CUDA driver version; reinstall torch with proper index URL if needed"
    else
        ok "torch + CUDA verified"
    fi
fi

# ── next steps ──────────────────────────────────────────────────────────────
echo
ok "install complete (mode=$MODE, venv=$VENV)"
echo "  activate:   source $VENV/bin/activate"
case "$MODE" in
    local)
        echo "  test:       $PYTHON -m pytest -q"
        echo "  generate:   ./scripts/generate_data.sh --n 10"
        ;;
    gpu)
        echo "  smoke:      ./scripts/train.sh --smoke-test"
        echo "  train:      ./scripts/train.sh --model Qwen/Qwen2.5-7B-Instruct"
        ;;
    eval)
        echo "  serve:      vllm serve <base> --enable-lora --lora-modules ft=artifacts/adapter"
        ;;
    all)
        echo "  smoke:      ./scripts/train.sh --smoke-test"
        echo "  generate:   ./scripts/generate_data.sh --n 50"
        echo "  train:      ./scripts/train.sh --model Qwen/Qwen2.5-7B-Instruct"
        ;;
esac
