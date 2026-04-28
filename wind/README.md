# wind

Wind-energy opposition classifier. Three-level output:
`opposition_detected` (binary) / `frames` (N_*) / `claims` (C_*).

Design rationale and pre-registered protocols live in
[`docs/methodology.md`](docs/methodology.md) and
[`docs/augmentation.md`](docs/augmentation.md).

## Pipeline

```
data/raw/aerows_full.jsonl
    │ prepare_splits.py
    ▼
data/test/{val,test}.jsonl + data/train/train_labels.jsonl
    │ teacher.py             (RECoT generation via Opus)
    │ clean_recot.py         (strip rows where teacher second-guesses)
    │ generate_synthetic.py  (stratified positive / negative augmentation)
    ▼
data/train/train.jsonl
    │ ft/train.py            (Unsloth + LoRA SFT)
    │ ft/quantize.py         (FP8_DYNAMIC)
    ▼
HF Hub model
    │ infer.py               (run on benchmark JSONL — vllm/openai/openrouter)
    ▼
data/results/<split>/<slug>.jsonl
    │ generate_report.py
    ▼
metrics
```

## Top-level scripts

| File | Role |
|---|---|
| `prompts.py`            | codebooks (frames N_*, claims C_*) + system prompt + triggers |
| `prepare_splits.py`     | `aerows_full.jsonl` → stratified 30/70 val/test split + train splits |
| `teacher.py`            | Opus-generated RECoT reasoning over training pool |
| `clean_recot.py`        | drop rows where teacher's `<think>` block shows second-guessing |
| `generate_synthetic.py` | Opus-generated synthetic positives / near-miss negatives (`--preset {positives,negatives}`) |
| `infer.py`              | run any chat-completion model on a benchmark JSONL |
| `generate_report.py`    | parse predictions and compute detection / frames / claims metrics |
| `errors.py`             | extract error rows for blinded review |
| `ft/train.py`           | SFT (single or joint, recot or norecot) |
| `ft/quantize.py`        | FP8_DYNAMIC quantization |

## Data

```
data/
├── raw/                    # source files (annotated, synthetic seeds)
├── train/                  # SFT splits (train + train_eval)
├── test/                   # held-out val + test
├── results/{val,test}/     # per-model inference outputs (input to generate_report)
└── sampling/               # annotation-tool input recipe (see its README)
```

## Setup

```bash
cp .env.example .env   # then fill in OPENROUTER_API_KEY / HF_TOKEN
```

Each top-level script has its own PEP-723 dependency block (`uv run <script>.py`); no global venv required.
