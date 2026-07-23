# Experiments log — dynamic few-shot

Running narrative for the dynamic-few-shot chapter: what I ran, what I learned,
why. Numbers live in `data/results/<task>/<split>/`. Sister logs:
[`cards/docs/experiments.md`](../../cards/docs/experiments.md) and
[`wind/docs/experiments.md`](../../wind/docs/experiments.md).

---

## 1. Goal

The cards/wind chapters showed a small LoRA-fine-tuned model can reach ~96% of
frontier-API performance at a fraction of inference cost. This chapter asks the
complementary question:

> Can **dynamic few-shot** (retrieval-augmented prompting, *no fine-tuning*)
> reach the same performance on the same tasks and test sets?

We compare static and dynamic few-shot against the existing zero-shot (base)
and LoRA-FT columns: zero / static / dynamic / fine-tuned. **Zero-shot is not
re-run** — it already exists as the `*-base` results in cards/wind (same slim
RECoT prompt + thinking). The only thing this chapter adds to that setup is the
demo turns.

## 2. Design (fairness)

- **Corpus** = `train + train_eval`, the exact labelled pool the FT models
  learned from. cards: `training_recot_opus.jsonl` (1,791). wind:
  `train_labels.jsonl` + `train_eval_labels.jsonl` (811).
- **Eval** = untouched, decontaminated `test`/`val` — the same files the FT
  table was scored on.
- **Same scorer** — `score.py` imports each sibling's `generate_report.py`; no
  metric reimplementation.
- **Same prompt + thinking as the base runs** — slim RECoT system prompt,
  `enable_thinking=True`. The model reasons on the query exactly as in zero-shot;
  the ONLY delta is the added demo turns, so few-shot and zero-shot are directly
  comparable.
- **Demos are text + labels only** (YAML, no `<think>` reasoning): keeps cost
  bounded at higher k, and makes static vs dynamic differ only in *which*
  examples, not their form.
- **Same k** for static and dynamic, so the only variable is selection.

## 3. Models

Base (no FT): Qwen3.5 {4B, 9B, 27B}, Gemma-4 {12B, 27B}. Frontier anchors:
Opus 4.8, GPT-5.6. Retrieval embedder: `Qwen/Qwen3-Embedding-0.6B` (vLLM,
OpenAI-compatible endpoint).

## 4. Open questions / caveats

- **Class imbalance vs top-k.** cards is ~80% `0_0_0`; pure top-k cosine may
  return mostly no-claim demos. Starting with top-k regardless of class (old
  paper's strategy); may add class-balanced retrieval as an ablation.
- **k sweep.** Old paper used 5. Small models are context-limited; frontier
  models are not. Plan: k ∈ {5, 10, 20}, note the context/cost trade-off.
- **Embedder ablation.** Qwen3-Embedding-0.6B is the default; could compare a
  second embedder to test retrieval-quality sensitivity.

## 5. Runs

_(to be filled in as runs complete)_
