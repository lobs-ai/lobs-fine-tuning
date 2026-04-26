"""
Phase 5 — model-agnostic Unsloth QLoRA SFT loop.

Consumes the pre-tokenized JSONL produced by `data_gen.format_for_training`
(records of `{input_ids, attention_mask, labels}`) and trains a LoRA adapter
on whatever HF causal LM you point `model.name` at. The labels already have
-100 on every non-assistant token, so we use the plain HF Trainer with a
collator that pads input_ids/labels/attention_mask to the longest item in
each batch.

Why pre-tokenized JSONL instead of SFTTrainer's text path:
- Loss masking is fragile; we already verified it byte-for-byte at format
  time. Re-templating here would risk the masks drifting.
- Lets us swap base models without retokenizing as long as the chat template
  is compatible. (When it isn't — e.g., switching tokenizer family —
  re-run `build_dataset.sh` with the new MODEL.)

Run:
    python -m train.train --model Qwen/Qwen2.5-7B-Instruct
    python -m train.train --model Qwen/Qwen3-8B-Instruct --lora-r 32
    python -m train.train --model meta-llama/Llama-3.1-8B-Instruct --output-dir artifacts/llama
    MODEL=Qwen/Qwen2.5-7B-Instruct python -m train.train      # MODEL env fallback
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── tiny config loader ──────────────────────────────────────────────────────
#
# We intentionally avoid pulling in OmegaConf/Hydra. yaml.safe_load + dotted
# CLI overrides covers the whole surface we need.


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # PyYAML is a transitive dep of transformers/datasets.

    with path.open() as f:
        return yaml.safe_load(f) or {}


def _set_dotted(cfg: dict[str, Any], key: str, value: str) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = _coerce(value)


def _coerce(s: str) -> Any:
    sl = s.strip().lower()
    if sl in ("true", "false"):
        return sl == "true"
    if sl in ("null", "none", "~"):
        return None
    try:
        if "." in s or "e" in sl:
            return float(s)
        return int(s)
    except ValueError:
        return s


# ── dataset ─────────────────────────────────────────────────────────────────


def _load_dataset(path: Path):  # noqa: ANN202 — datasets type
    from datasets import Dataset

    if not path.exists():
        return None
    rows: list[dict[str, list[int]]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "input_ids": rec["input_ids"],
                    "attention_mask": rec["attention_mask"],
                    "labels": rec["labels"],
                }
            )
    if not rows:
        return None
    return Dataset.from_list(rows)


@dataclass
class PadCollator:
    """Pad input_ids / attention_mask / labels to the longest item in a batch.

    `labels` pads with -100 (ignore-index for cross-entropy); `input_ids`
    pads with the tokenizer's pad token; `attention_mask` pads with 0.
    """

    pad_token_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids: list[list[int]] = []
        attn: list[list[int]] = []
        labels: list[list[int]] = []
        for b in batch:
            pad_n = max_len - len(b["input_ids"])
            input_ids.append(list(b["input_ids"]) + [self.pad_token_id] * pad_n)
            attn.append(list(b["attention_mask"]) + [0] * pad_n)
            labels.append(list(b["labels"]) + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT against any HF instruct model. Defaults come from --config; "
        "first-class flags below override that file; trailing key=value pairs override anything.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m train.train --model Qwen/Qwen2.5-7B-Instruct\n"
            "  python -m train.train --model Qwen/Qwen3-8B-Instruct --lora-r 32 --epochs 3\n"
            "  python -m train.train --model meta-llama/Llama-3.1-8B-Instruct --output-dir artifacts/llama\n"
            "  MODEL=Qwen/Qwen2.5-7B-Instruct python -m train.train\n"
        ),
    )
    parser.add_argument("--config", default="train/config.yaml", help="YAML config (default: train/config.yaml)")
    parser.add_argument(
        "--model",
        default=None,
        help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct. Overrides model.name in config. "
        "Falls back to $MODEL env var.",
    )
    parser.add_argument("--max-seq-length", type=int, default=None, help="override model.max_seq_length")
    parser.add_argument("--lora-r", type=int, default=None, help="override lora.r")
    parser.add_argument("--lora-alpha", type=int, default=None, help="override lora.alpha")
    parser.add_argument("--epochs", type=float, default=None, help="override train.num_train_epochs")
    parser.add_argument("--lr", type=float, default=None, help="override train.learning_rate")
    parser.add_argument("--output-dir", default=None, help="override train.output_dir")
    parser.add_argument("--adapter-dir", default=None, help="override save.adapter_dir")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run 2 training steps and exit — verifies the loop without committing to a full run",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="resolve and print the config (yaml + flags + overrides) and exit without training",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help='extra dotted overrides like train.gradient_accumulation_steps=8',
    )
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))

    # First-class flags map straight onto dotted keys.
    flag_map: dict[str, tuple[str, Any]] = {
        "model.name": ("model", args.model),
        "model.max_seq_length": ("max_seq_length", args.max_seq_length),
        "lora.r": ("lora_r", args.lora_r),
        "lora.alpha": ("lora_alpha", args.lora_alpha),
        "train.num_train_epochs": ("epochs", args.epochs),
        "train.learning_rate": ("learning_rate", args.lr),
        "train.output_dir": ("output_dir", args.output_dir),
        "save.adapter_dir": ("adapter_dir", args.adapter_dir),
    }
    for key, (_, value) in flag_map.items():
        if value is not None:
            _set_dotted(cfg, key, str(value))

    for ov in args.overrides:
        if "=" not in ov:
            print(f"bad override (need key=value): {ov!r}", file=sys.stderr)
            return 2
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k.strip(), v.strip())

    model_name = cfg.get("model", {}).get("name") or os.environ.get("MODEL", "")
    if not model_name:
        print(
            "no model — pass --model <hf-id>, set model.name in the config, "
            "or export MODEL=<hf-id>",
            file=sys.stderr,
        )
        return 2
    cfg["model"]["name"] = model_name
    print(f"▸ training {model_name}", file=sys.stderr)

    if args.print_config:
        import yaml as _yaml

        print(_yaml.safe_dump(cfg, sort_keys=False))
        return 0

    train_path = Path(cfg["data"]["train_path"])
    val_path = Path(cfg["data"]["val_path"])
    train_ds = _load_dataset(train_path)
    val_ds = _load_dataset(val_path)
    if cfg["data"].get("require_nonempty", True) and (train_ds is None or len(train_ds) == 0):
        print(f"no training rows in {train_path} — run scripts/build_dataset.sh first", file=sys.stderr)
        return 2

    # ── unsloth/transformers imports are heavy; do them after arg validation.
    import torch

    if not torch.cuda.is_available():
        print(
            "CUDA not available — Unsloth requires an NVIDIA GPU with CUDA. "
            "Run this on your GPU host.",
            file=sys.stderr,
        )
        return 2

    from unsloth import FastLanguageModel
    from transformers import TrainingArguments, Trainer

    # Unsloth's optimized fast path requires lora_dropout=0 and bias="none".
    # If you flip these, Unsloth still works but silently disables a
    # noticeable speedup. Warn loudly so it's not a surprise.
    lc = cfg["lora"]
    if float(lc.get("dropout", 0.0)) != 0.0:
        print(
            f"⚠ lora.dropout={lc.get('dropout')} disables Unsloth's fast path. "
            f"Set to 0.0 to keep the speedup.",
            file=sys.stderr,
        )
    if lc.get("bias", "none") != "none":
        print(
            f"⚠ lora.bias={lc.get('bias')!r} disables Unsloth's fast path. "
            f"Use 'none' to keep the speedup.",
            file=sys.stderr,
        )

    # Auto-detect bf16 support; fall back to fp16 on older GPUs (V100/T4).
    tc = cfg["train"]
    want_bf16 = bool(tc.get("bf16", True))
    if want_bf16 and not torch.cuda.is_bf16_supported():
        print(
            "▸ bf16 requested but GPU doesn't support it — falling back to fp16",
            file=sys.stderr,
        )
        tc["bf16"] = False
        tc["fp16"] = True

    # Sequence-length sanity: if the formatted dataset has rows longer than
    # the configured max_seq_length, training will silently truncate (and
    # break the loss masks). The format step already drops > max_len rows,
    # but the user may have set inconsistent values across runs.
    mc = cfg["model"]
    max_seq = int(mc["max_seq_length"])
    if train_ds is not None:
        longest = max(len(r["input_ids"]) for r in train_ds)
        if longest > max_seq:
            print(
                f"⚠ longest training row is {longest} tokens but model.max_seq_length={max_seq}. "
                f"Re-run build_dataset.sh with --max-len {max_seq} (or raise max_seq_length).",
                file=sys.stderr,
            )

    print(
        f"loading {model_name} "
        f"(4bit={mc.get('load_in_4bit')}, max_seq_length={max_seq}, "
        f"train_rows={len(train_ds) if train_ds else 0}, val_rows={len(val_ds) if val_ds else 0})",
        file=sys.stderr,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        dtype=mc.get("dtype"),
        load_in_4bit=bool(mc.get("load_in_4bit", True)),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc.get("dropout", 0.0),
        bias=lc.get("bias", "none"),
        target_modules=lc["target_modules"],
        use_gradient_checkpointing=lc.get("use_gradient_checkpointing", "unsloth"),
        random_state=lc.get("random_state", 42),
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # Fall back to eos so the collator can still build tensors. The
        # attention mask zeroes these positions out anyway.
        pad_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = pad_id
    collator = PadCollator(pad_token_id=pad_id)

    if tc.get("report_to") and "wandb" in tc["report_to"]:
        os.environ.setdefault("WANDB_PROJECT", tc.get("wandb_project", "lobs-fine-tuning"))

    # Smoke test: 2 steps, no checkpoint, no eval, no wandb. Just verify
    # the loop is wired up before committing to a full run.
    if args.smoke_test:
        print("▸ SMOKE TEST: 2 steps, no save/eval/wandb", file=sys.stderr)
        tc["save_strategy"] = "no"
        tc["eval_strategy"] = "no"
        tc["report_to"] = []
        tc["logging_steps"] = 1

    training_args_kwargs: dict[str, Any] = dict(
        output_dir=tc["output_dir"],
        num_train_epochs=tc.get("num_train_epochs", 2),
        per_device_train_batch_size=tc.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 16),
        learning_rate=tc.get("learning_rate", 2e-4),
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        warmup_ratio=tc.get("warmup_ratio", 0.03),
        weight_decay=tc.get("weight_decay", 0.0),
        logging_steps=tc.get("logging_steps", 5),
        eval_strategy=tc.get("eval_strategy", "no") if val_ds is not None else "no",
        eval_steps=tc.get("eval_steps", 50),
        save_strategy=tc.get("save_strategy", "steps"),
        save_steps=tc.get("save_steps", 100),
        save_total_limit=tc.get("save_total_limit", 3),
        bf16=bool(tc.get("bf16", True)),
        fp16=bool(tc.get("fp16", False)),
        optim=tc.get("optim", "adamw_8bit"),
        seed=tc.get("seed", 42),
        report_to=tc.get("report_to", []),
    )
    if args.smoke_test:
        training_args_kwargs["max_steps"] = 2
        training_args_kwargs["num_train_epochs"] = 1  # ignored when max_steps set; appease validator

    training_args = TrainingArguments(**training_args_kwargs)

    # transformers 4.46+ deprecates `tokenizer=` on Trainer in favor of
    # `processing_class=`. Detect at runtime so we work on either side of
    # the rename.
    import inspect

    trainer_params = inspect.signature(Trainer.__init__).parameters
    tokenizer_kwarg = "processing_class" if "processing_class" in trainer_params else "tokenizer"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        **{tokenizer_kwarg: tokenizer},
    )

    trainer.train()

    if args.smoke_test:
        print("▸ smoke test passed — training loop runs end-to-end", file=sys.stderr)
        return 0

    sc = cfg.get("save", {})
    adapter_dir = sc.get("adapter_dir", "artifacts/adapter")
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"saved LoRA adapter → {adapter_dir}", file=sys.stderr)

    if sc.get("save_merged"):
        merged_dir = sc.get("merged_dir", "artifacts/merged")
        Path(merged_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"saved merged-16bit weights → {merged_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
