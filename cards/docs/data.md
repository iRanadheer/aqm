# Data

Three artifact groups live in `data/`:

1. **Training data** — fine-tunes a student model on teacher-distilled CARDS reasoning.
2. **Congressional benchmark** (`cards_val.jsonl`, `cards_test.jsonl`) — held-out U.S. congressional speech.
3. **Twitter benchmark** (`cards_twitter.jsonl`) — expert-labeled climate tweets.

```
training.csv ──teacher.py──> training_recot_opus.jsonl ─┐
                                                         │
                                                         ├─ prepare_splits.py
                                                         │
congress_test.csv ───────────────────────────────────────┘                      ──> cards_train{,_eval}{,_norecot}.jsonl
                                                                                    cards_val.jsonl, cards_test.jsonl

augmented_cards.csv ──> chapter sample ──┐
                                          ├──> cards_twitter.jsonl  (built once, frozen — no script in this repo)
        + ai-assisted + hamburg zeros ────┘
```

All `prepare_splits.py` outputs are deterministic with `random_state=42`; running the script reproduces the checked-in artifacts byte-for-byte (verified).

---

## 1. Training data

### `teacher.py` — distill teacher reasoning

Calls a teacher LLM on each `(text, true_claims)` row from
`data/training.csv` (1,791 rows) and asks it to produce expert-level
`<think>` reasoning that arrives at the given labels, plus a YAML category
list. Output is appended JSONL, resume-safe (skips rows whose text prefix
is already present in the output). Backend is LiteLLM, so any provider
works.

```
python teacher.py \
    --input data/training.csv \
    --model anthropic/claude-opus-4-20250514 \
    --output data/training_recot_opus.jsonl \
    --concurrency 5
```

The checked-in `training_recot_opus.jsonl` was distilled from
`claude-opus-4-6` at `temperature=0`, `max_tokens=4000`. Re-running will
produce a **methodologically equivalent but not byte-identical** output —
LLM calls aren't bit-stable even at temperature 0, model snapshots get
deprecated and replaced, and the concurrent writer reorders rows by
completion time. Downstream `cards_train*.jsonl` will shift accordingly.

For exact reproduction of published metrics, use the checked-in JSONL.
For new experiments (different teacher, refreshed labels), rerun
`teacher.py` — it costs ≈ 1.8k Opus calls.

### `prepare_splits.py --train`

90/10 stratified split of `training_recot_opus.jsonl` on the first category
code; labels with count < 2 are bucketed as `_rare_` so `train_test_split`
accepts them. The same indices are reused for the no-RECoT mirror.

| File | Rows | Notes |
|---|---|---|
| `cards_train.jsonl` | 1,611 | SFT messages, full RECoT |
| `cards_train_eval.jsonl` | 180 | early-stopping mirror, full RECoT |
| `cards_train_norecot.jsonl` | 1,611 | same rows, `<think>` stripped, no CoT trigger |
| `cards_train_eval_norecot.jsonl` | 180 | same, eval mirror |

The `_norecot` variants strip `<think>...</think>` from the assistant turn
and drop the CoT trigger from the user turn — same rows, same boundary, no
reasoning supervision.

---

## 2. Congressional benchmark — `cards_val.jsonl` / `cards_test.jsonl`

### Source

`data/congress_test.csv` — 2,051 U.S. congressional speech excerpts from:

> Coan, T.G., Malla, R., Nanko, M.O. et al. *Large language model reveals an
> increase in climate contrarian speech in the United States Congress.*
> Commun. Sustain. **1**, 37 (2026). <https://doi.org/10.1038/s44458-025-00029-z>

This is **our own paper** — `cards_val/test.jsonl` are the eval splits used to
report metrics in that work.

### `prepare_splits.py --test`

30/70 stratified split, `random_state=42`, stratified on the first
`true_claims` code with the same `_rare_` bucketing as the training split.
After the partition is fixed, labels are promoted from
`data/mapping/final_claims_dict.json` — the post-review label set used in the
Nature paper. ~5% of items have a revised label set; the partition itself
is unchanged. Without this override, 67 test + 26 val rows would carry
older seed labels and diverge from published metrics.

| File | Rows | Notes |
|---|---|---|
| `cards_val.jsonl` | 615 (30%) | prompt iteration / early stopping |
| `cards_test.jsonl` | 1,436 (70%) | held-out final eval |

Both keep only `{id, text, true_claims}`.

---

## 3. Twitter benchmark — `cards_twitter.jsonl`

744 expert-labeled climate tweets, assembled in two stages from the
expert-labeled "golden" subset of `augmented_cards.csv`. The script that
produced this file is **not** in the current repo — `cards_twitter.jsonl` is
a frozen artifact. The recipe below is the historical record.

### Stage 1 — chapter sample from the `golden` subset

**Source.** `augmented_cards.csv` (75,653 rows) combines three datasets:

| DATASET | Rows | PARTITION-wise | | |
|---|---:|---:|---:|---:|
| | | **TRAIN** | **TEST** | **VALID** |
| waterloo | 43,943 | 35,154 | 4,395 | 4,394 |
| cards | 28,999 | 23,460 | 2,931 | 2,608 |
| golden | 2,711 | 0 | 2,711 | 0 |

The 2,711 `golden` rows correspond exactly to
`Expert_annotated_Climate_Tweets.csv` — expert-labeled climate tweets. The
file has parallel claim columns: expert (`claim`, `full_claim`, `claim_5.3`)
and model (`cards_pred`, `hcards_complete_pred`).

Zero/non-zero distribution in the source expert file (2,711 rows):

| Column | Zeros | Other | Nulls |
|---|---:|---:|---:|
| `claim` (expert) | 1,049 (38.7%) | 1,592 (58.7%) | 70 (2.6%) |
| `cards_pred` (model) | 1,640 (60.5%) | 1,001 (36.9%) | 70 (2.6%) |
| `hcards_complete_pred` (model) | 1,893 (69.8%) | 818 (30.2%) | 0 |

In `augmented_cards.csv`, the derived `acards_claim` column has 1,049 zeros
and 1,662 non-zero rows in the golden subset (the 70 originally-null rows
were imputed before this stage).

**Filter.**
1. Golden only (2,711 rows).
2. Drop `acards_claim == "0_0"` exactly (1,049 rows) — leaves 1,662
   claim-bearing rows. Categories with a zero subscript like `1_0`, `3_0` are
   real top-level subclaims and are kept.

**Sample.** Threshold-based stratified — cap at 50 per `acards_claim`
category, keep all rows for under-50 categories, `random_state=42`. Yields
~730 rows. 24 unique categories; 15 of them have <50 samples.

Stage 1 outputs:
- `augmented_cards_sample_chapter.csv` — full sampled dataframe
- `augmented_cards_sample_chapter.jsonl` — `{id, text}`, IDs formatted
  `augCards_chapter_{index}`

Of these ~730 chapter rows, **172** end up in the final benchmark (those
tagged `source = "cards-chapter"`).

#### Intercoder reliability on the chapter sample

Two coders (`travcoan`, `mirjamnanko`) independently applied taxonomy codes
to a 50-item subset of the chapter sample. Source labels live in
`data/cards_icr.json`; running `python icr_analysis.py` writes:

- [`docs/icr_report.md`](icr_report.md) — Krippendorff's Alpha, percent
  exact agreement, and mean Jaccard at each hierarchy level, plus per-claim
  frequencies.
- `data/icr_disagreements.csv` — per-item disagreements at Level 3.

Headline numbers from the checked-in report:

| Level | Krippendorff's Alpha | % Exact Agreement | Mean Jaccard |
|-------|----------------------|-------------------|--------------|
| 1 (top-level) | 0.81 | 68.0% | 0.81 |
| 2 (sub-category) | 0.82 | 62.0% | 0.78 |
| 3 (claim) | 0.79 | 54.0% | 0.73 |

Codes are normalized before scoring — redundant parent codes are dropped
when a strictly more specific child is present in the same set (e.g. `2_1_0`
is dropped when `2_1_4` is also present).

### Stage 2 — assembly

`cards_twitter.jsonl` (744 rows) combines:

| `source` | Rows | Notes |
|---|---:|---|
| `cards-chapter` | 172 | from Stage 1, expert-labeled |
| `cards-ai-assisted` | 338 | additional expert-annotated examples |
| `hamburg_test_set3` | 234 | zero-claim filler from Hamburg test set |
| **Total** | **744** | |

The 172 + 338 = **510** expert rows were labeled by **two experts**. The 234
hamburg zeros push class balance toward the target ~40% zero / ~60%
non-zero, matching the natural balance in the expert-labeled climate
tweets (~39% / ~59%).

---

## CLI reference

End-to-end from scratch:

```
# 1. Distill teacher reasoning from training.csv (slow + costs Opus calls)
python teacher.py --input data/training.csv \
    --model anthropic/claude-opus-4-20250514 \
    --output data/training_recot_opus.jsonl

# 2. Build SFT train splits + congressional eval splits (deterministic, fast)
python prepare_splits.py            # train + test
python prepare_splits.py --train    # train splits only
python prepare_splits.py --test     # test splits only (cards_val + cards_test)
```

Inputs to `prepare_splits.py`: `data/training_recot_opus.jsonl`,
`data/congress_test.csv`, `data/mapping/final_claims_dict.json`. All outputs
land in `data/`.

The Twitter benchmark is a frozen artifact — no script in this repo
regenerates it.
