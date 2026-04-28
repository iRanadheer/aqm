# Training (`ft/train.py`)

One PEP-723 entrypoint for every variant. Same script runs on HF Jobs,
Vast.ai, RunPod, or any cu128 GPU host without changes.

```
uv run ft/train.py --base-model <hf-id> [--joint] [--no-recot] [--lora-only]
```

## Recipe

Unsloth `FastLanguageModel` + PEFT LoRA + TRL `SFTTrainer`. `bf16`,
cosine LR, `adamw_8bit`, gradient checkpointing on. Best checkpoint
selected on `eval_loss`. Hyperparams (rank, lr, epochs, batch, max-seq) are
hardcoded as defaults; CLI flags override.

## Joint vs single

- **Single (default):** trains on the CARDS dataset only.
- **`--joint`:** also pulls a second dataset (the Wind chapter) and trains
  on the concatenation. Same backbone, two label vocabularies.

## RECoT vs no-RECoT

Selected by `--recot/--no-recot`. The flag picks which pre-built JSONL to
load; the data pipeline produced both variants from the same row indices.

## Output

Always pushes a LoRA adapter to the Hub. By default also merges LoRA into
the base and pushes the full BF16 model (skip with `--lora-only`). Hub
identities come from `HF_USERNAME` env (with a sensible default).

## Variant naming

The output repo name is derived from the base model + flags. Override with
`--variant`.
