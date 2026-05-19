# debunk

Climate-claim fact-checking benchmark over the Leippold et al. (2024)
veracity dataset (170 claims; 160 after de-dup).

The model is asked to assess a single claim and emit a label. Two prompt
variants are supported:

| Prompt        | Label space (raw)                                          | Collapses to |
|---------------|------------------------------------------------------------|--------------|
| `veracityV1`  | TRUE / MISLEADING / FALSE / UNVERIFIABLE                   | identity     |
| `climinator`  | 12-label Climate Feedback / Climinator scheme              | 4-class veracity via [`prompts.CLIMINATOR_TO_VERACITY`](prompts.py) |

Both prompts live in [`prompts.py`](prompts.py), ported from the
production TypeScript files (`debunkingMessages_*.ts`).

## Pipeline

```
data/raw/Leippold2024_veracity.csv
    │ prepare_splits.py            (dedup, schema-normalise)
    ▼
data/test/test.jsonl
    │ infer.py                     (run any chat model — vllm/openai/openrouter)
    ▼
data/results/test/<slug>-<prompt>.jsonl
    │ generate_report.py           (parse Assessment line, score vs. true_veracity)
    ▼
data/results/test/metrics_summary.{json,md}
```

## Top-level scripts

| File | Role |
|---|---|
| `prompts.py`         | system prompts (veracityV1, climinator) + label-collapse maps |
| `prepare_splits.py`  | `Leippold2024_veracity.csv` → `data/test/test.jsonl` |
| `infer.py`           | run any chat-completion model on the benchmark JSONL |
| `generate_report.py` | parse `**Assessment:**` lines, compute accuracy / macro-F1 vs. gold veracity |

## Data record format

Test rows (`data/test/test.jsonl`):

```json
{
  "itemId":          "leippold_001",
  "claim":           "<verbatim claim>",
  "source":          "<source attribution>",
  "date":            "DD-MMM-YY",
  "content":         "<claim>",
  "true_veracity":   "FALSE",
  "true_cfb_label":  "Incorrect",
  "true_climinator": "incorrect",
  "duplicated":      false
}
```

Result rows are the test record plus a `response` field (the raw model
output, or `"ERROR: ..."` on a backend failure).

## Setup

```bash
cp .env.example .env   # then fill in OPENAI_API_KEY / OPENROUTER_API_KEY
pip install -r requirements.txt
```

## Quick run

```bash
python prepare_splits.py
python infer.py --backend openrouter --model anthropic/claude-opus-4.7 --prompt veracityV1
python infer.py --backend openrouter --model anthropic/claude-opus-4.7 --prompt climinator
python generate_report.py
```
