# Synthetic Augmentation Plan

Canonical list of codes targeted for synthetic augmentation, with the
rationale for each. This doc is the authoritative source — the commands
below should match it exactly.

See `pipeline_learnings.md` (§9) for methodology. Summary:

- Positives are selected from **train-only** signal (support < 10).
- Negatives are selected from **val** signal (FP ≥ 12). Val is a dev
  set; test remains untouched and is the basis for the paper's final
  reported numbers.
- Target total support per code: 30.
- Generator uses a task-specific system prompt (POSITIVES_SYSTEM or
  NEGATIVES_SYSTEM) that embeds the full codebook. Output is pydantic
  structured JSON (list of texts). No regex parsing.
- Labeler uses the production classification prompt
  (`full_system_instruction`) and treats each generated text as an
  independent classification. Quality gate keeps rows where teacher
  confirms (positives) or denies (negatives) the target code.
- Both calls cache the system prompt via `cache_control: ephemeral`.

---

## Positives — add more examples of codes the model under-predicts

**Selection rule (pre-registered)**: a code receives synthetic positive
augmentation if EITHER:

- (a) **Train support < 10** — addresses under-representation / class
  imbalance. Derivable from training set alone, no val consultation.
- (b) **Val FN > 10** — indicates boundary confusion / insufficient
  training coverage of the code's variations, even when train support
  is adequate. Only val aggregate counts are used; specific val texts
  never enter training.

Rule (a) catches data-starved codes; rule (b) catches codes where
training diversity (not quantity) is the bottleneck. A code qualifying
under either rule receives augmentation.

Target total-support per code: 30. Justification: matches the median
support of the frames vocabulary (27) and the maximum support observed
in the claims vocabulary (30).

### Target codes

**Rule (a) — train support < 10 (42 claims + 1 frame):**

All claims with train support below threshold:
```
C_25_5 C_17_2 C_28_0 C_18_0          (train = 1)
C_28_1 C_28_2 C_28_3 C_33_0
  C_18_1 C_2_1                        (train = 2)
C_25_4 C_19_0                         (train = 3)
C_35_0 C_7_0  C_2_2  C_4_0  C_8_0     (train = 4)
C_37_0 C_10_0 C_17_1 C_38_0 C_9_0
  C_22_0 C_31_0 C_17_0 C_25_0 C_36_0  (train = 5)
C_28_4 C_28_5 C_29_0 C_21_0           (train = 6)
C_17_3 C_3_0  C_16_0 C_25_2           (train = 7)
C_14_0 C_15_0 C_25_3 C_13_0           (train = 8)
C_2_0  C_6_0  C_0_1                   (train = 9)
```

Frames with train support below threshold:
```
N_0  (train = 9)
```

**Rule (b) — val FN > 10 with train support ≥ 10 (3 frames):**

| Code | Train | Val FN | Val FP | Val F1 (4B) | Pattern |
|---|---:|---:|---:|---:|---|
| N_1 | 32 | 16 | 22 | 0.46 | Boundary confusion (both FN and FP high) |
| N_3 | 91 | 19 | 21 | 0.78 | Under-coverage on variations |
| N_5 | 27 | 14 | 43 | 0.24 | Boundary confusion (also in negatives) |

These three frames have adequate train support by quantity but still
under-recall on val. Synthetic positives add diversity; the paired
negatives (see below) sharpen the boundary.

### Total codes to augment with positives: 43 claims + 4 frames = 47.

### Command

```bash
./run_augment.sh positives
```

Full code list is hardcoded in `run_augment.sh::POSITIVES`. Target 30 per code,
~47 codes → ~1400 target rows. Expected kept (after quality gate): ~1000-1200.

---

## Negatives — add near-miss examples for codes the model over-predicts

**Selection rule**: val FP ≥ 12 AND FP > FN. Over-prediction is only
observable by running the model on a labeled set and comparing to
gold — val is the only legitimate source for this signal. Specific val
texts never enter training; only the aggregate FP counts inform the
code list.

Teacher-generated near-miss texts are labeled by the teacher from
scratch; rows where the teacher confirms the target code is present are
DROPPED (inverse quality gate). Kept rows become training signal "this
is NOT code X."

### Target codes (7)

| Code | Train | Val FN | Val FP | Val F1 (4B) | Notes |
|---|---:|---:|---:|---:|---|
| N_5 | 27 | 14 | 43 | 0.24 | Over-prediction — regulation framing |
| C_2_0 | 9 | 6 | 28 | 0.19 | Over-prediction — alternative energy comparison (also in positives) |
| N_6 | 30 | 5 | 23 | 0.48 | Over-prediction — unethical practices framing |
| N_1 | 42 | 16 | 22 | 0.46 | Over-prediction — environment framing |
| N_3 | — | 19 | 21 | 0.78 | Over-prediction — cost framing (borderline but rule-compliant) |
| C_32_0 | — | 5 | 17 | 0.58 | Over-prediction — umbrella claim |
| C_33_0 | 2 | 9 | 12 | 0.09 | Balanced (also in positives) |

Note: `C_2_0` and `C_33_0` appear in both positives and negatives — they
have both FN and FP problems. Running both passes for these codes is
intentional: positives teach the model to recognize the code; negatives
teach it not to over-apply the code. The two roles are complementary.

### Command

```bash
uv run generate_synthetic.py --negatives \
  --codes N_5,C_2_0,N_6,N_1,N_3,C_32_0,C_33_0 \
  --per-code-target 30 --concurrency 15
```

Or via the bash wrapper:
```bash
./run_augment.sh negatives
```

Expected output: ~140-180 kept rows appended to
`data/train/train_synthetic_labels.jsonl`.

---

## Claims with val FN > 10 but NOT currently augmented (open question)

The expanded rule would also cover three claims with train ≥ 10 but
val FN > 10:

| Code | Train | Val FN | Val F1 (4B) | Prior note |
|---|---:|---:|---:|---|
| C_27_0 | 16 | 24 | 0.40 | Earlier flagged as codebook-clarity issue |
| C_26_0 | 22 | 28 | 0.67 | Well-supported; FN but F1 OK |
| C_20_0 | 15 | 16 | 0.61 | Similar profile |

Not currently in `run_augment.sh::POSITIVES`. Decision pending — the
uniform rule argues for inclusion; the prior analysis argued that
C_27_0's problem is codebook, not data. For now kept out; if frame
positives improve overall macro but these claims remain weak, add them
in a follow-up pass.

---

## Running both passes

```bash
# 1. Positives + 2. Negatives (code lists hardcoded in run_augment.sh)
./run_augment.sh all

# 3. Teacher RECoT reasoning on the combined synthetic labels
uv run teacher.py --inputs data/train/train_synthetic_labels.jsonl

# 4. Merge into training set
cat data/train/train.jsonl data/train/train_synthetic.jsonl \
  > data/train/train_augmented.jsonl

# 5. Upload train_augmented.jsonl to HF, retrain, benchmark on val.
```

---

## Expected impact (for reference, validated via ablation after retraining)

Before retraining we cannot claim gains. Post-retraining, we'll report
both val and test results. Target: close the teacher–student gap on
claims macro_F1 (sup≥4) from ~0.55 (current 9B slim) toward Opus slim
(~0.72). A single augmentation pass is unlikely to fully close the gap;
we'll iterate codebook + augmentation until plateau or budget exhausts.
