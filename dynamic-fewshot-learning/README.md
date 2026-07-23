# dynamic-fewshot-learning

Can **dynamic few-shot** prompting — retrieving semantically similar labelled
examples per test item and putting them in the prompt — match a **fine-tuned**
model, with *no training at all*? This chapter answers that on the two tasks
whose FT baselines already exist in the monorepo: [`cards/`](../cards/)
(hierarchical climate-contrarian claims) and [`wind/`](../wind/) (wind-energy
opposition, 3-level).

No fine-tuning, same held-out test/val and same scorers the LoRA models were
measured on. **Zero-shot is not re-run here** — it already exists as the base
runs in `cards/`/`wind/` (same slim RECoT prompt + thinking). The *only* delta
is the added text+label demo turns:

| Regime | Demos in prompt |
|---|---|
| (`zeroshot`) | none — reuse the existing `*-base` results |
| `static`   | one fixed, stratified sample of *k* demos, reused for every item |
| `dynamic`  | the *k* corpus items most cosine-similar to the item being classified |

Everything else matches the base zero-shot runs exactly: the **slim RECoT system
prompt with thinking ON**, so the model still reasons on the query. Demos carry
**text + labels only** (no reasoning) to keep cost down at higher *k*.

**Fair comparison.** The few-shot corpus is `train + train_eval` — the exact
labelled pool the fine-tuned models learned from — while `test`/`val` stay
untouched and decontaminated. So every regime and every FT model see the same
labelled data and are scored on the same test set.

> **Deliberate cross-chapter import.** Unlike other chapters, this one imports
> `cards/` and `wind/` `prompts.py` and `generate_report.py` directly (see
> `tasks.py`, `score.py`). That is the point: the prompts and metrics must be
> *byte-identical* to the FT runs for the comparison to be fair.

## Layout

| File | Role |
|---|---|
| `tasks.py`        | task adapters (cards, wind): corpus/split loaders, norecot system prompt, demo renderer, result schema |
| `embedder.py`     | embeddings via an OpenAI-compatible endpoint (Qwen3-Embedding-0.6B on vLLM) + disk cache |
| `build_corpus.py` | embed a task's `train+train_eval` corpus, cache to `data/embeddings/` |
| `retrieval.py`    | `StaticSelector` (fixed stratified sample) and `DynamicSelector` (per-item top-k cosine) |
| `infer.py`        | run a chat model on a task under one regime → `data/results/<task>/<split>/<slug>.jsonl` |
| `score.py`        | score a result file with the sibling chapter's own metrics |
| `docs/experiments.md` | running log: what was run, what was learned |

## Pipeline

```
# 1. serve the embedder (dynamic regime only)
vllm serve Qwen/Qwen3-Embedding-0.6B          # -> :8000/v1/embeddings

# 2. cache corpus embeddings (once per task)
uv run build_corpus.py --task cards
uv run build_corpus.py --task wind

# 3. run static + dynamic for a model (zero-shot reused from base runs)
uv run infer.py --task cards --model qwen/qwen3.5-9b --regime static  --k 10
uv run infer.py --task cards --model qwen/qwen3.5-9b --regime dynamic --k 10

# 4. score against the base/FT-table metrics
uv run score.py --task cards data/results/cards/test/qwen-qwen3.5-9b-dynamic-k10.jsonl
```

## Models

Base (no FT) Qwen3.5 {4B, 9B, 27B} and Gemma-4 {12B, 27B} — matching the FT
sizes in `cards/`/`wind/` — plus frontier anchors (Opus 4.8, GPT-5.6). Every
model runs all three regimes. Retrieval embedder: `Qwen/Qwen3-Embedding-0.6B`.

## Setup

```bash
cp .env.example .env    # fill in OPENROUTER_API_KEY / OPENAI_API_KEY
```

Each script has its own PEP-723 dependency block — run with `uv run <script>.py`.
The paper source lives in `.archives/dynamic-fewshot-learning/`.
