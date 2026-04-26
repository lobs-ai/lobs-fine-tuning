# lobs-fine-tuning

Tool-calling fine-tune of any HF instruct model (Qwen2 / Qwen2.5 / Qwen3 / Llama-3.x / Mistral, …), distilled from Claude Sonnet, targeting the [squad](../squad) agent system.

The base model is configured via the `MODEL` env var or the `model.name` field in `train/config.yaml`. The same id drives both data formatting (chat template + tokenizer) and training (base weights), so switching base models is one variable change followed by a re-run of `build_dataset.sh`.

See `SPEC.md` (in chat history / project root once added) for the full project spec and `DECISIONS.md` for the choices that deviate from it.

## Pipeline

```
  Gemini CLI (synth)  ─┐
  squad transcripts  ──┼──► harvester ──► canonical trajectories
  manual exports     ──┘                       │
                                                ▼
                                              filter
                                                │
                                                ▼
                                  HF chat-template formatter ──► train.jsonl
                                                │
                                                ▼
                                       Unsloth QLoRA training
                                                │
                                                ▼
                                  vLLM-served LoRA ──► squad eval harness
```

## Install

```bash
# local dev (Mac/Linux) — data gen, tests, no GPU
./scripts/install.sh                          # creates .venv, installs base+dev extras
npm i -g @google/gemini-cli && gemini         # one-time gemini login (data gen)

# GPU host — full training
./scripts/install.sh --mode gpu               # adds Unsloth, torch+CUDA, bitsandbytes, etc.

# all-in-one (Linux GPU host doing everything)
./scripts/install.sh --mode all
```

`./scripts/install.sh --help` for flags. `.env` is auto-scaffolded from `.env.example` on first run; edit it with real API keys before running anything that calls out.

## Quick start

End-to-end, two commands on the GPU host:

```bash
./scripts/generate_data.sh --n 200          # synth trajectories via Gemini CLI + run build_dataset.sh
./scripts/train.sh --model Qwen/Qwen2.5-7B-Instruct
```

Or, more granularly:

### Generate synthetic data (Gemini CLI)

```bash
# 50 mixed-category trajectories with 4 parallel gemini calls, then run build_dataset.sh
./scripts/generate_data.sh

# focus on debugging trajectories
./scripts/generate_data.sh --n 30 --category debugging

# pin a specific gemini model and skip the post-build step
./scripts/generate_data.sh --n 100 --model gemini-2.5-pro --skip-build

# see all flags
./scripts/generate_data.sh --help
```

Output lands in `data/trajectories_raw/full/{run_id}.json` — exactly where `build_dataset.sh` looks. Failures land in `data/trajectories_raw/full/.failed/{run_id}.txt` for inspection.

### Build the dataset

One script runs the harvest → scrub → filter → format pipeline with sensible defaults:

```bash
# happy path — all defaults from .env
./scripts/build_dataset.sh

# pick the base model (drives tokenizer + chat template + later training)
MODEL=Qwen/Qwen3-8B-Instruct ./scripts/build_dataset.sh
./scripts/build_dataset.sh --model meta-llama/Llama-3.1-8B-Instruct

# common overrides
./scripts/build_dataset.sh --sessions-dir ~/some/other/sessions
./scripts/build_dataset.sh --strict-tools --max-turns 12
./scripts/build_dataset.sh --skip-format            # stop before tokenizing
./scripts/build_dataset.sh --dry-run                # print commands only
./scripts/build_dataset.sh --help                   # see every flag and default
```

Every flag has a matching env var (e.g. `--sessions-dir` ↔ `SESSIONS_DIR`). The script reports the resolved values up front so you can tell what it actually used.

If you'd rather drive each stage by hand, the underlying modules are independent:

```bash
python -m data_gen.harvester --out data/trajectories_canonical/all.jsonl
python -m data_gen.secret_scrub --in data/trajectories_canonical/all.jsonl --out data/trajectories_canonical/all.scrubbed.jsonl --report data/_scrub_report.json
python -m data_gen.filter --in data/trajectories_canonical/all.scrubbed.jsonl --out data/trajectories_filtered/all.jsonl
python -m data_gen.format_for_training --in data/trajectories_filtered/all.jsonl --train-out data/train.jsonl --val-out data/val.jsonl --tokenizer "$MODEL" --verify
```

### Train

Needs a CUDA GPU; Unsloth doesn't run on Mac.

```bash
# one-command launcher — sources .env, rebuilds the dataset for $MODEL, then trains
./scripts/train.sh
./scripts/train.sh --model Qwen/Qwen3-8B-Instruct
./scripts/train.sh --skip-build --lora-r 32 --epochs 3

# or call the trainer directly
python -m train.train --model Qwen/Qwen2.5-7B-Instruct --lora-r 32
python -m train.train --config train/config.yaml train.gradient_accumulation_steps=8
```

## Layout

```
tools/             vendored schemas (FROZEN per DECISIONS.md)
data_gen/          harvester, filter, formatter, runner
data_gen/synth/    Gemini-CLI-driven synthetic trajectory generator
train/             Unsloth training loop + config
eval/              eval harness + held-out tasks
data/              all generated artifacts (gitignored)
scripts/           build_dataset.sh, generate_data.sh, train.sh
tests/             unit tests (mostly format round-trip checks)
```
