# Error analysis — 27B on test

Blinded review files for an expert to adjudicate disagreements between the
27B model's predictions and the human-annotated gold labels on the test set.

## Files

| File | Purpose |
|---|---|
| `27b-test-errors.jsonl` | All 363 error rows (47% of 773 test rows). |
| `27b-test-errors.key.jsonl` | Unblinding key for the full file. **Do not share with the expert.** |
| `27b-test-errors.sample.jsonl` | Stratified sample (25 per error-type signature → 100 rows). |
| `27b-test-errors.sample.key.jsonl` | Unblinding key for the sample. **Do not share with the expert.** |

## What counts as an error

A test row is an "error" if the parsed prediction disagrees with the gold
label on at least one of:

- `detection` — `opposition_detected` flipped
- `frames` — frame set differs
- `claims` — claim set differs

## Why blinded A/B

If the expert is told *which* labels came from the model vs. the human, they
anchor on one side as authoritative and only fix the obvious mistakes — the
review becomes a sanity check, not a real adjudication. To get unbiased
fixes, for each row we randomly assign the gold and predicted label sets to
slots `a` and `b` (50/50 per row, seed 42). The expert doesn't know which is
which until the key is applied.

The expert verdict per row is one of: `A`, `B`, `both`, `neither`, or
`custom` (with their own labels in `expert_annotation`).

## Stratification

When taking a sample, we stratify by the row's error-type signature so the
expert sees each failure mode in roughly equal volume. Test set has 4
signatures:

| Signature | Count | Notes |
|---|---:|---|
| `frames+claims` | 126 | Detection right, both label sets wrong. |
| `claims` | 106 | Detection + frames right, claims wrong. |
| `detection+frames+claims` | 95 | Detection wrong; frames/claims mismatches are forced by the flip — really a "detection error" stratum. |
| `frames` | 36 | Detection + claims right, frames wrong. |

The other 3 theoretically-possible signatures (`detection`,
`detection+frames`, `detection+claims`) are empty: a detection flip zeros
out one side's frames and claims, so it always produces full-set mismatches.

## Row format

Generic envelope so the same shape can be reused for other projects/tasks
(only the contents of `a` / `b` / `expert_annotation` change per task):

```json
{
  "itemId": "...",
  "content": "...",
  "a": {"opposition_detected": true, "frames": [...], "claims": [...]},
  "b": {"opposition_detected": false, "frames": [], "claims": []},
  "expert_choice": "",            // "A" | "B" | "both" | "neither" | "custom"
  "expert_annotation": {},        // same shape as a/b, when expert_choice = "custom"
  "expert_notes": ""
}
```

Key file row:

```json
{"itemId": "...", "a_is": "gold", "b_is": "pred", "error_types": ["frames", "claims"]}
```

## Regenerate

```bash
python3 errors.py \
  --input data/results/test/iranadheer-windy-qwen3-5-27b.jsonl \
  --out-prefix data/results/error_analysis/27b-test-errors \
  --sample-per-stratum 25
```

Same `--seed` reproduces the same blinding and sample.

## Caveats

- Two source rows have `"itemId": NaN` (lines 772–773 of the test
  predictions file — itemId was lost upstream). The content is intact, so
  the expert can still review by text.
