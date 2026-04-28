# Sampling Methodology for `augmented_cards_sample_chapter`

## Data Source

- **Input file**: `augmented_cards.csv` — a combined dataset of 75,653 records from three sources:
  - `waterloo` (43,943 rows)
  - `cards` (28,999 rows)
  - `golden` (2,711 rows)
- **Partitions**: TRAIN (58,614), TEST (10,037), VALID (7,002)

### Origin of the `golden` subset

The 2,711 `golden` rows correspond exactly to `Expert_annotated_Climate_Tweets.csv` (expert-labeled climate tweets). That file contains multiple parallel claim columns — both expert annotations (`claim`, `full_claim`, `claim_5.3`) and model predictions (`cards_pred`, `hcards_complete_pred`).

**Zero vs non-zero distribution in the source expert file** (2,711 rows):

| Column | Zeros (no-claim) | Other claims | Nulls |
|--------|------------------|--------------|-------|
| `claim` (expert) | 1,049 (38.7%) | 1,592 (58.7%) | 70 (2.6%) |
| `full_claim` (expert, with subcodes) | 1,049 (38.7%) | 1,592 (58.7%) | 70 (2.6%) |
| `claim_5.3` (expert, refined) | 1,049 (38.7%) | 1,592 (58.7%) | 70 (2.6%) |
| `cards_pred` (model) | 1,640 (60.5%) | 1,001 (36.9%) | 70 (2.6%) |
| `hcards_complete_pred` (model) | 1,893 (69.8%) | 818 (30.2%) | 0 |

The expert ground truth is ~39% no-claim vs ~59% claim; model predictions skew more aggressively toward "no claim". In `augmented_cards.csv`, the `acards_claim` column is derived such that the golden subset has 1,049 zeros and 1,662 non-zero rows (no nulls — the 70 originally-null rows appear to have been imputed/filled before this stage).

## Filtering

1. **Golden subset only**: Only the `golden` dataset (2,711 rows) was used for sampling.
2. **Remove non-claims**: Rows where `acards_claim == "0_0"` (no claim detected, 1,049 rows) were excluded, leaving 1,662 claim-bearing rows.
   - Note: only the literal `"0_0"` no-claim bucket was filtered. Other categories with a zero subscript (e.g. `1_0`, `3_0`) represent actual top-level subclaims and were **kept**.
   - **Rationale**: `augmented_cards.csv` is treated as the source of truth for the no-claim label — if `acards_claim` says a row is `0_0`, it's a zero. We do not second-guess that flag against other parallel claim columns.

## Category Distribution (Pre-Sampling)

The `acards_claim` column had 24 unique categories with highly imbalanced counts:

| Category | Count | Category | Count |
|----------|-------|----------|-------|
| 5_2      | 498   | 4_4      | 46    |
| 5_3      | 200   | 1_6      | 41    |
| 2_1      | 154   | 3_2      | 31    |
| 4_1      | 103   | 1_1      | 28    |
| 5_1      | 96    | 1_4      | 27    |
| 1_7      | 89    | 3_3      | 23    |
| 1_3      | 61    | 2_3      | 22    |
| 4_2      | 61    | 1_2      | 20    |
| 4_5      | 50    | 3_0      | 12    |
|          |       | 4_3      | 9     |
|          |       | 3_1      | 8     |
|          |       | 1_0      | 6     |
|          |       | 1_8      | 5     |
|          |       | 3_4      | 1     |
|          |       | 3_6      | 1     |

- 15 categories had fewer than 50 samples (totaling 280 rows).

## Sampling Strategy

A **threshold-based stratified sampling** approach was used (`sample_by_category` function):

- **Threshold**: 50 samples per category
- **Categories with >= 50 samples**: Randomly sampled down to exactly 50 (using `random_state=42` for reproducibility)
- **Categories with < 50 samples**: Kept all rows (no oversampling)

This balances the dataset by capping over-represented categories while preserving all data from under-represented ones.

## Output Files

1. **`augmented_cards_sample_chapter.csv`** — Full sampled dataframe with all columns
2. **`augmented_cards_sample_chapter.jsonl`** — JSONL with `id` and `text` columns only
   - IDs follow the format `augCards_chapter_{index}` (e.g., `augCards_chapter_0`, `augCards_chapter_1`, ...)

## Summary

| Step | Records |
|------|---------|
| Full dataset | 75,653 |
| Golden subset | 2,711 |
| After removing non-claims | 1,662 |
| After threshold sampling (cap at 50) | ~730 (estimated) |

---

# Twitter Benchmark Dataset

A separate benchmark dataset is being constructed for evaluation on Twitter-style text.

## Composition

| Source | Rows | Notes |
|--------|------|-------|
| Expert-annotated examples | 510 | Annotated by **two experts** |
| Zeros from `hamburg_test_set3` | 234 | Added to balance the no-claim ratio |
| **Total** | **744** | |

## Target Distribution

The benchmark targets approximately **40% zeros (no-claim) / 60% non-zeros (claim-bearing)**, mirroring the natural class balance observed in the expert-labeled climate tweets (`Expert_annotated_Climate_Tweets.csv` had ~39% zeros / ~59% claims among labeled rows).

The 234 hamburg test-set-3 zeros were added on top of the 510 expert-annotated examples to reach the target zero share.
