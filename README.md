# lobs-fine-tuning

Tool-calling fine-tune of Qwen2.5-7B-Instruct, distilled from Claude Sonnet, targeting the [squad](../squad) agent system.

See `SPEC.md` (in chat history / project root once added) for the full project spec and `DECISIONS.md` for the choices that deviate from it.

## Pipeline

```
squad transcripts ──► harvester ──► canonical trajectories
                                          │
                                          ▼
                                       filter
                                          │
                                          ▼
                                   Qwen formatter ──► train.jsonl
                                                          │
                                                          ▼
                                               Unsloth QLoRA training
                                                          │
                                                          ▼
                                          vLLM-served LoRA ──► squad eval harness
```

## Quick start

One script runs the whole pipeline (harvest → scrub → filter → format) with sensible defaults:

```bash
# happy path — all defaults from .env
./scripts/build_dataset.sh

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
python -m data_gen.format_for_training --in data/trajectories_filtered/all.jsonl --train-out data/train.jsonl --val-out data/val.jsonl --tokenizer Qwen/Qwen2.5-7B-Instruct --verify
```

Then train (needs GPU):

```bash
python -m train.train --config train/config.yaml
```

## Layout

```
tools/         vendored schemas (FROZEN per DECISIONS.md)
data_gen/     harvester, filter, formatter, runner
train/        Unsloth training loop + config
eval/         eval harness + held-out tasks
data/         all generated artifacts (gitignored)
tests/        unit tests (mostly format round-trip checks)
```
