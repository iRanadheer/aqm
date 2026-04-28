# Wind Opposition Detection — Pipeline Learnings

Running log of design decisions, experiments, and findings. Intended as
raw material for the Nature paper; not final narrative.

---

## 1. Goal

Classify social-media/news texts about wind energy along three levels:

1. **Detection** — binary: does the text express opposition?
2. **Frame** (`N_*`) — which high-level frame, if any?
3. **Claim** (`C_*`) — which specific claim(s)?

Fallback codes:
- `N_0` / `C_0_1` — opposition is present but no specific frame / claim
  fits (escape hatches under `opposition_detected: true`).
- `C_0_0` — not opposition (used as a negative-training signal, not a
  predicted class at inference).

---

## 2. Data pipeline (deterministic, reproducible)

```
data/raw/aerows_full.jsonl ────────┐
                                   ├──► prepare_splits.py ──► val.jsonl + test.jsonl
data/annotation_patches.json ──────┘                        (with optional patches on val)

data/raw/fine_tuning_data_final_fixed.csv ─┐
data/raw/synthetic_zeros.jsonl ──────────┤
                                           ├──► prepare_splits.py ──► train_labels.jsonl
                                           │                         + train_eval_labels.jsonl
                                           └──► (90/10 stratified split, seed=42)

train_labels.jsonl ──► teacher.py ──► train.jsonl (OpenAI chat messages format)
                        (teacher = Opus 4.7 + gold labels + RECOT_TRIGGER)
                        |
                        └──► clean_recot.py (drops flaky rows)
```

- Splits are **byte-identical** to the upload on HF; stratification is
  seeded; no leakage between train / val / test.
- 27 val-only annotation corrections live in `annotation_patches.json`.
- `synthetic_zeros.jsonl` supplies C_0_0 negatives during the split
  stage. It's a checked-in static artifact in `data/raw/`, generated
  once historically (the recipe is not in this repo).

### Row counts

| Split | Rows | Notes |
|---|---:|---|
| train | 726 | from `train_labels.jsonl`, after RECoT generation and second-guessing filter |
| train_eval | 86 | held-out eval during SFT |
| val | 331 | benchmark set; 182 opposition-positive |
| test | (same pipeline, held-out) | |

---

## 3. Prompt engineering

### 3.1 Three-level cascade (current)

Superseded an earlier two-level (narratives + claims) scheme. Introduced:
- `N_0` escape hatch: opposition present, no specific frame fits.
- `C_0_1` escape hatch: opposition present, no specific claim fits.
- `C_32_X` subclaim collapse: multiple subclaims folded into single
  `C_32_0` umbrella.

### 3.2 Reasoning chain (in `full_system_instruction`)

```
CONTEXT → DETECTION → FRAMES → CLAIMS (shortlist → adjudicate →
                                       granularity → force-fit check)
       → SPECIAL CONSIDERATIONS
```

### 3.3 Codebook iterations

Sharpened these entries after per-code error analysis (all from same
FN+FP pattern: confusion with adjacent codes or over-broad definitions):

- `C_20_0` — added "who bears the cost?" test
- `C_33_0` — scoped to government overreach as pretext
- `C_28_5`, `C_26_0`, `C_2_2`, `C_28_0`, `C_24_0`, `C_4_0`, `C_23_0`,
  `C_28_4` — principle-first rewrites
- `C_30_0` — regulation *too lax* (not just *too strict*)
- `C_0_1` — added force-fit check
- `C_32_0` — umbrella without keyword list

### 3.4 Prompt variants

- `full_system_instruction` — 4,843 tokens (cl100k_base)
- `slim_system_instruction` — 3,603 tokens
  - Same reasoning chain, slimmer codebook definitions (~26% shorter)

Both produce comparable teacher accuracy. See §7.

---

## 4. Training setup (SFT via Unsloth LoRA)

### 4.1 Script: `ft/train.py`

- PEP 723 inline deps (self-contained script).
- Supports `--model 0.8b|2b|4b|9b`.
- `--batch-size`, `--grad-accum`, `--no-merge` flags.
- Pulls data from `iRanadheer/wind-opposition-sft` (HF dataset).

### 4.2 Hyperparameters (matching 27B reference from cards)

| Knob | Value | Notes |
|---|---|---|
| LoRA r | 16 | Unsloth 2026 default; sweet spot for domain+reasoning |
| LoRA alpha | 16 | α=r, 1.0 scaling (Unsloth 2026 rec) |
| LoRA dropout | 0 | Unsloth 2026 rec |
| target_modules | q, k, v, o, gate, up, down | full attn + MLP |
| num_train_epochs | 3 | loss plateaus by epoch 0.5; no gain from more |
| per_device_batch | 4 (9B) or 8 (4B) | |
| grad_accum | 2 (9B) or 1 (4B) | effective batch = 8 in both cases |
| learning_rate | 2e-4 | LoRA standard |
| scheduler | cosine | |
| warmup_steps | 10 | |
| weight_decay | 0.01 | |
| optim | adamw_8bit | saves optimizer memory; no quality loss |
| max_seq_length | 8192 | median training row 5,065 tokens; max 5,910 |
| bf16 | True | H200/A100 prefers bf16 |
| load_best_model_at_end | True | metric = eval_loss, lower = better |

### 4.3 Infrastructure

**Winner: RunPod H200 (141 GB).** After HF Jobs a100-large failed
repeatedly with no logs and HF Jobs L40S had dependency issues,
RunPod was stable.

Known gotchas:
- **Triton ≥ 3.4.0 on Hopper GPUs** produces incorrect `gated
  chunk_bwd_dqkwg` results. Fix: `pip install tilelang`. Added to PEP
  723 deps so it's auto-installed every `uv run`.
- vLLM 0.19+: automatic prefix caching is default-on. No flag needed.
- On H200, with `--max-workers 75` default in `infer.py`, prefix
  cache hit rate collapsed to 0% (eviction under pressure). Dropping
  to `--max-workers 8` restored hit rate to ~99% and didn't hurt total
  wall time — GPU was already the bottleneck.

### 4.4 Training curves — diagnosis

Both 9B runs (full-prompt Turbo and slim) show the same shape:

- Large drop epoch 0–0.3 as model learns YAML + `<think>` format.
- Plateau from ~epoch 0.5 onward.
- `eval_loss` never rises (no overfitting signature).
- Train ≈ eval (tiny generalization gap).

**Diagnosis: capacity/data-plateau, not overfit.** The 9B Turbo ended
at `eval_loss=0.057`; slim ended at `eval_loss=0.075`. Slim's ~33%
higher loss reflects the student having less in-context codebook
to lean on during training, *not* worse generalization — see §7.

Implication: more epochs won't help, `load_best_model_at_end` always
picks the right checkpoint, and the real bottleneck is **data diversity
on rare codes**.

---

## 5. Evaluation

### 5.1 `infer.py`

One script, three backends:
- `openai` — standard OpenAI API.
- `openrouter` — Anthropic prompt caching on system prompt
  (`cache_control: ephemeral` as structured content block;
  ~10× cheaper after warmup).
- `vllm` — `http://localhost:8000/v1`, no auth.

Per-model output paths: `data/results/<split>/<slug>[-<prompt>].jsonl`,
where `<split>` is derived from the input filename stem (e.g.
`data/test/test.jsonl` → `data/results/test/`) and `<prompt>` is appended
only for the non-default `--prompt full` variant. No overwrite across
models or prompts.

### 5.2 `generate_report.py`

- Detection (binary): acc/prec/rec/F1.
- Frames & claims (multi-label):
  - `samples_f1` — per-row F1, averaged; empty-gold+empty-pred = 1.0
    (sklearn's built-in scores that as 0, which is wrong for this task).
  - `macro_f1`, `micro_f1`, exact_match, hamming_loss.
- Two views for frames/claims: **all rows** and **opposition-only** (gold
  `opposition_detected=true`).
- `--per-code` — per-code FN/FP/F1 breakdown.
- `--min-support N` — filter per-code analysis to codes with gold
  support ≥ N; recomputes macro_F1 over the kept codes.

---

## 6. Benchmarks on val.jsonl (331 rows)

### 6.1 Teacher + 9B configurations — full comparison

Three 9B deployment configurations plus the teacher baseline:

| Metric | Opus slim (2-run avg) | 9B Turbo (full→full) | 9B slim → slim | 9B slim → full (mismatched) |
|---|---:|---:|---:|---:|
| Detection F1 | **0.942** | 0.850 | 0.838 | 0.850 |
| Frames samples_F1 | **0.824** | 0.697 | 0.707 | 0.703 |
| Frames macro_F1 (all) | **0.651** | 0.327 | **0.476** | 0.303 |
| Frames micro_F1 | **0.752** | 0.624 | 0.648 | 0.632 |
| Frames exact_match | **0.690** | 0.553 | 0.550 | 0.544 |
| Claims samples_F1 | **0.812** | 0.657 | 0.660 | 0.663 |
| Claims macro_F1 (all) | **0.681** | **0.524** | 0.472 | 0.492 |
| Claims micro_F1 | **0.742** | 0.565 | 0.572 | 0.575 |
| Claims exact_match | **0.625** | 0.444 | 0.468 | 0.447 |

### 6.2 4B and scaling

| Metric | 9B slim → slim | 4B slim → slim | 9B − 4B |
|---|---:|---:|---:|
| Detection F1 | 0.838 | 0.849 | −0.011 |
| Frames samples_F1 | 0.707 | 0.682 | +0.025 |
| Frames macro_F1 (all) | 0.476 | 0.368 | **+0.108** |
| Claims samples_F1 | 0.660 | 0.625 | +0.035 |
| Claims macro_F1 (all) | 0.472 | 0.365 | **+0.107** |
| Claims exact_match | 0.468 | 0.432 | +0.036 |

Scaling 4B → 9B adds ~+10 macro_F1 on the long tail for both frames
and claims.

### 6.3 Opposition-only subset (n=182)

| Metric | Opus slim | 9B Turbo | 9B slim → slim | 4B slim → slim |
|---|---:|---:|---:|---:|
| Frames samples_F1 | **0.735** | 0.631 | 0.648 | 0.614 |
| Frames macro_F1 | **0.666** | 0.358 | **0.523** | 0.418 |
| Claims samples_F1 | **0.712** | 0.557 | 0.563 | 0.510 |
| Claims macro_F1 | **0.689** | **0.544** | 0.493 | 0.395 |

Teacher numbers averaged across two independent runs (`v2-slim-run1`
+ `v2-slim-run2`) for stability; run-to-run variance floor is ±0.017
macro_F1.

### 6.4 Key takeaways

1. **Prompt compression is not a free lunch** — see §7.
2. **Training/inference prompt match matters on frames**: slim→slim
   is +17 pts over slim→full on frames macro_F1, and +15 pts over
   the Turbo baseline. On claims the matching effect is smaller
   (slim→slim slightly worse than slim→full).
3. **Scaling 4B → 9B helps the long tail** (~+10 macro_F1 on both
   levels). The rare codes benefit more than the common ones.
4. **Teacher–student claims macro_F1 gap**: Opus slim 0.681 → 9B slim
   matched 0.472 ≈ 21 pts. Still the largest remaining gap;
   augmentation target.
5. **Weakness pattern is consistent** across model sizes and prompt
   variants — same rare codes collapse hardest. See §8.

---

## 7. Slim vs full prompt — three-config analysis

Originally claimed "slim ≈ full" based on a **mismatched**
comparison (slim-trained student inferenced with the full prompt).
Correcting that produces a more nuanced picture. All three 9B
configurations were run back-to-back on the same 331-row val set:

| Config | Train prompt | Infer prompt | Frames macro | Claims macro |
|---|---|---|---:|---:|
| Turbo | full | full | 0.327 | **0.524** |
| Slim-matched | slim | slim | **0.476** | 0.472 |
| Slim-mismatched | slim | full | 0.303 | 0.492 |

### Two findings, different directions

**Finding 1 — Frames: matched slim > full; mismatched slim is worst of all.**
- Turbo (full→full) 0.327 vs slim-matched (slim→slim) 0.476 = **+14.9 pts**.
- Slim-mismatched (slim→full) 0.303 is *worse than Turbo*, suggesting
  the model's frame decisions depend on the exact prompt byte-shape it
  saw during training. Swap the prompt at inference and frame
  boundaries degrade.

**Finding 2 — Claims: full > slim, regardless of match.**
- Turbo (full→full) **0.524** is the best config for claims macro.
- Slim-matched 0.472 (−5.2 pts). Slim-mismatched 0.492 (in between).
- The extra codebook detail in the full prompt appears to help the
  model adjudicate between adjacent claim subcodes, and the benefit
  survives even when the student was trained on the slim variant.

**Aggregate metrics stay within ±0.02 across all three** — the macro
differences come from a small number of rare codes flipping, not from
broad quality shifts.

### Publication-relevant implications

- No free-lunch prompt compression story. The slim prompt **helps
  frames, hurts claims** (by different magnitudes, but both real).
- **Match the training and inference prompt**. The slim-mismatched
  config is worst on frames and gives no meaningful gain on claims vs
  the matched slim config.
- Deployment recommendation splits by granularity requirement:
  - Coarse frames only → 9B slim → slim. Cheaper at inference, better
    macro.
  - Fine-grained claims → 9B Turbo → full. Higher claim macro_F1.
- The teacher–student gap on claims macro_F1 is ~21 pts regardless
  of variant. That's a **data-diversity problem**, not a prompt
  problem. Augmentation (§9) is the real lever to close it.

---

## 8. Weak-code analysis (macro_F1 bottleneck)

### 8.1 Symptom

Macro-F1 of the 9B-slim student on claims (sup≥4) is 0.557 vs Opus 0.738.
The gap is driven by a small number of consistently low-F1 codes.

### 8.2 Per-code profile (9B-slim on val, sup≥4)

**Claims — worst performers:**

| Code | Support | FN | FP | F1 | Pattern |
|---|---:|---:|---:|---:|---|
| C_28_0 | 5 | 5 | 13 | 0.00 | FN+FP — entangled w/ C_28_5 |
| C_2_0 | 10 | 6 | 25 | 0.21 | FP-dominant — over-predicted |
| C_33_0 | 10 | 7 | 12 | 0.24 | Balanced |
| C_27_0 | 36 | 24 | 14 | 0.39 | Chronic under-recall (high train support — NOT data-limited) |
| C_0_1 | 9 | 6 | 4 | 0.38 | Escape hatch, under-used |

**Frames — worst performers:**

| Code | Support | FN | FP | F1 | Pattern |
|---|---:|---:|---:|---:|---|
| N_0 | 10 | 8 | 2 | 0.29 | Under-used escape hatch |
| N_5 | 23 | 10 | 29 | 0.40 | FP-dominant — over-predicted |
| N_6 | 18 | 4 | 25 | 0.49 | FP-dominant |

### 8.3 Failure-mode taxonomy

| Pattern | What it means | What helps |
|---|---|---|
| FN ≫ FP (under-recall) | Model doesn't know code applies | More positive examples |
| FP ≫ FN (over-prediction) | Model over-generalizes the definition | Near-miss negatives OR codebook sharpening |
| Balanced both-high | Conflation with a sibling code | Contrastive examples (both positives and negatives of the confused pair) |
| High train support + low val F1 | Not data-limited; codebook is ambiguous | Codebook iteration, not augmentation |

**Critical**: `C_27_0` has **36** training examples and is still weak.
Augmentation probably won't help; it's a codebook-clarity issue.
Tagging for a future codebook pass.

---

## 9. Synthetic augmentation plan

### 9.1 `generate_synthetic.py`

Single script, two modes via CLI flags:

- **Positives** (default) — generate texts that SHOULD be labeled with
  the target code. Teacher re-labels each generated text from scratch
  using the production classifier prompt; rows where the teacher
  confirms the target code survive.
- **Negatives** (`--negatives`) — generate near-miss texts that
  should NOT be labeled with the target code. Quality gate is inverted:
  rows where the teacher confirms the target code are dropped. Teacher's
  labels become gold for the kept rows.

Code lists are supplied via either:

- `--preset {positives, negatives}` — pre-registered code lists hardcoded
  in `PRESETS` at the top of the script (see `docs/augmentation.md`).
  `--preset negatives` implies `--negatives`.
- `--codes A,B,C` — ad-hoc list, optionally combined with `--negatives`.

Note: `synthetic_zeros.jsonl` (the C_0_0-negative seed file in
`data/raw/`) was generated by an earlier version of this script and is
now a static checked-in artifact; the current `generate_synthetic.py`
no longer has that mode.

### 9.2 Pipeline

```
generate_synthetic.py --preset {positives|negatives}
    │
    └─► train_synthetic_labels.jsonl   (train_labels schema)
            │
            └─► teacher.py       (teacher reasoning + YAML)
                    │
                    └─► clean_recot.py  (drops flaky)
                            │
                            └─► train_synthetic.jsonl
                                    │
                                    └─► concat with train.jsonl → retrain
```

### 9.3 Training support distribution (anchors the augmentation target)

Computed over `train_labels.jsonl` (726 rows):

**Claims** (56 codes, 419 total occurrences):
- min=1, p25=4, **median=6**, p75=10, **max=30**
- 42 of 56 codes (75%) have support < 10
- Only 1 code has support ≥ 30

**Frames** (9 codes, 296 total occurrences):
- min=9, p25=26, **median=27**, p75=42, max=63
- 1 frame (N_0) has support < 10

The claims vocabulary is severely long-tailed; the frames vocabulary
is much more balanced.

### 9.4 Per-code augmentation target: 30 examples. Justification:

Three independent arguments converge on 30:

1. **Matches the frames median support (27).** After augmentation,
   low-support codes sit at the typical-frame level. Rare codes are
   no longer long-tail outliers in training frequency.
2. **Matches the claims maximum support (30).** Augmented rare
   claims reach the ceiling of what exists naturally in our training
   data — a conservative stop that doesn't synthetically exceed the
   best-observed code.
3. **Few-shot learning regime (20-30).** Empirically, LoRA with r=16
   reliably learns a new pattern from 20-30 examples. Below 20 is
   noisy; above 50 gives diminishing returns relative to API cost.

**Pre-registered rule for the paper**:
> "We augment each under-supported code to a target of 30 training
> examples. This target corresponds to the median support of the
> frames vocabulary (27) and the maximum support observed in the
> claims vocabulary (30), and sits in the few-shot learning regime
> where low-rank adaptation reliably internalizes new patterns."

Planned sensitivity ablations (future work): re-run augmentation with
targets of 15 and 50 to characterize the sensitivity of final F1 to
augmentation volume.

### 9.5 Selection signal — what informs which codes to augment

Val-set per-code F1 **is** a valid selection signal — this is what
a validation set is for. Test set remains untouched and will be the
basis for the final reported numbers. To guard against iterative
val-overfitting, we cap development cycles on val at a small count
and report both val and test in the final paper.

Cross-referenced diagnostic rule:

| Pattern on val | Train support | Intervention |
|---|---|---|
| FN ≫ FP | < 10 | Positives (data starvation) |
| FN ≫ FP | ≥ 10 | Both — add positives *and* flag for codebook review |
| FP ≫ FN | any | Negatives (over-prediction; contrastive examples) |
| FN ≈ FP, both high | low | Both (positives + negatives) |
| Low F1, high train support | any | Codebook refinement, not augmentation |

### 9.6 Target codes (4B model, post-slim benchmark)

**Positives (FN-heavy, per-code target 30):**
- `C_28_0`, `C_0_1`, `N_0`, `C_28_4`, `C_3_0`, `C_4_0`, `C_28_5`,
  `C_27_0`, `C_26_0`, `C_19_0`, `C_22_0`, `C_33_0`

**Negatives (FP-heavy, per-code target 30):**
- `N_5`, `N_6`, `C_2_0`, `C_32_0`, `C_33_0`

Codebook-review, not augmented: `C_27_0` has 16 train examples and
still under-recalls on val. Flagged for a future codebook-clarity
pass. (Leaving it in positives for now — the augmentation may still
help; separate codebook revision is a follow-up.)

### 9.7 Cost estimate

- ~7 weak codes × ~30 rows × ~2 teacher calls (gen + label) ≈ ~420 calls
- Plus RECoT on ~200–400 kept rows after the quality gate ≈ another ~300 calls
- With Anthropic prompt caching via OpenRouter: **~$15–30 on Opus 4.7,
  ~45–60 min wall time** at concurrency 10.

### 9.8 Final augmentation run

Generated on 2026-04-23 with the pipeline above:

- **Positives**: 1,066 rows across 44 claim codes + 4 frames (N_0 plus incidental
  multi-label coverage of N_1/N_3/N_5 via side-effect).
- **Negatives**: 210 rows across 7 codes (N_5, N_6, N_1, N_3, C_2_0,
  C_32_0, C_33_0).
- **Total synthetic**: 1,276 rows (→ 1,273 after second-guessing filter).
- **Merged into** `data/train/train.jsonl`: 726 → 1,999 rows. All slim
  system prompt. Uploaded to `iRanadheer/wind-opposition-sft`.
- **train_eval.jsonl**: unchanged (86 real-only rows) — preserves a
  real-corpus-only held-out signal for checkpoint selection, insulated
  from the synthetic training distribution.
- **Final class balance**: 1,213 opposition (61%) / 786 non-opposition
  (39%). Close to val's 55/45 ratio.

### 9.9 Augmentation result — 4B retrain on val

4B retrained on augmented train.jsonl (1999 rows) and benchmarked on
val (331 rows). **Clean improvement on every metric**:

| Metric | Pre-augment 4B | **Post-augment 4B** | Δ |
|---|---:|---:|---:|
| Detection F1 | 0.849 | **0.864** | +0.015 |
| Frames samples_F1 | 0.682 | **0.729** | +0.047 |
| **Frames macro_F1 (all)** | 0.368 | **0.569** | **+0.201** |
| Frames exact_match | 0.544 | **0.604** | +0.060 |
| Claims samples_F1 | 0.625 | **0.672** | +0.047 |
| **Claims macro_F1 (all)** | 0.365 | **0.488** | **+0.123** |
| Claims exact_match | 0.432 | **0.462** | +0.030 |

**Opposition-only subset (n=182)** — even larger gains where classification actually matters:

| Metric | Pre-aug | Post-aug | Δ |
|---|---:|---:|---:|
| Frames macro_F1 | 0.418 | **0.656** | +0.238 |
| Claims macro_F1 | 0.395 | **0.540** | +0.145 |

### 9.9b Augmentation result — 9B retrain on val

9B retrained on the same augmented train.jsonl and benchmarked on val.
**Similar pattern to 4B but with bigger gains on claims.**

| Metric | Pre-augment 9B | **Post-augment 9B** | Δ |
|---|---:|---:|---:|
| Detection F1 | 0.838 | **0.866** | +0.028 |
| Frames samples_F1 | 0.707 | **0.727** | +0.020 |
| **Frames macro_F1 (all)** | 0.476 | **0.566** | **+0.090** |
| Frames exact_match | 0.550 | **0.574** | +0.024 |
| Claims samples_F1 | 0.660 | **0.693** | +0.033 |
| **Claims macro_F1 (all)** | 0.472 | **0.601** | **+0.129** |
| Claims exact_match | 0.468 | **0.477** | +0.009 |

**Opposition-only subset (n=182):**

| Metric | Pre-aug | Post-aug | Δ |
|---|---:|---:|---:|
| Frames macro_F1 | 0.523 | **0.620** | +0.097 |
| Claims macro_F1 | 0.493 | **0.639** | +0.146 |

### 9.9c Augmentation result — 27B retrain on val (MONEY SHOT)

27B retrained on the augmented train.jsonl. **Matches the slim teacher
on the hardest generalization metric.**

| Metric | **Post-augment 27B** |
|---|---:|
| Detection F1 | **0.912** |
| Frames samples_F1 | **0.794** |
| Frames macro_F1 (all) | 0.606 |
| **Frames macro_F1 (sup≥4)** | **0.619** |
| Frames exact_match | 0.656 |
| Claims samples_F1 | **0.786** |
| Claims macro_F1 (all) | 0.639 |
| **Claims macro_F1 (sup≥4)** | **0.716** 🎯 |
| Claims exact_match | 0.589 |

**Headline: Claims macro_F1 (sup≥4) = 0.716 vs Opus slim 0.715 —
statistically tied.** The distilled 27B student matches the slim-prompt
teacher on claims macro, while operating on the same shorter prompt
(1,240 tokens saved per request). Full-prompt Opus (0.738) remains
ahead by 0.022 — a small, explainable gap given the teacher has
~40× the parameters and slightly richer in-context codebook.

**Opposition-only (n=182):**

| Metric | 27B aug | Opus slim | Opus full |
|---|---:|---:|---:|
| Frames macro_F1 | **0.637** | 0.632 | 0.693 |
| Claims macro_F1 | 0.657 | 0.687 | 0.692 |

27B aug matches Opus slim on opposition-only frames macro.

**Note on 27B training recovery**: the final `trainer.evaluate()` step
crashed with a CUDA illegal memory access (fragmentation at end of a
3.5h H200 run). Training itself completed cleanly (750/750 steps,
train loss 0.1027). LoRA checkpoint was already on HF; merged weights
were recovered via an ad-hoc one-off merge+push (no longer in repo).

### 9.10 Cross-model landscape (post-augment)

| Metric | Opus full | Opus slim | **27B aug** | **9B aug** | **4B aug** | 9B pre-aug | 4B pre-aug |
|---|---:|---:|---:|---:|---:|---:|---:|
| Detection F1 | **0.947** | 0.939 | 0.912 | 0.866 | 0.864 | 0.838 | 0.849 |
| Frames samples_F1 | **0.836** | 0.815 | 0.794 | 0.727 | 0.729 | 0.707 | 0.682 |
| Frames macro (all) | **0.680** | 0.613 | 0.606 | 0.566 | 0.569 | 0.476 | 0.368 |
| Frames macro (sup≥4) | **0.681** | 0.627 | 0.619 | 0.574 | 0.578 | 0.535 | 0.464 |
| Claims samples_F1 | **0.826** | 0.811 | 0.786 | 0.693 | 0.672 | 0.660 | 0.625 |
| Claims macro (all) | **0.687** | 0.678 | 0.639 | 0.601 | 0.488 | 0.472 | 0.365 |
| **Claims macro (sup≥4)** | **0.738** | 0.715 | **0.716** 🎯 | 0.630 | 0.580 | 0.541 | 0.456 |

### Scaling curve — Claims macro_F1 (sup≥4)

```
0.456  4B pre-aug   ████████████
0.541  9B pre-aug   ██████████████▌
0.580  4B aug       ████████████████
0.630  9B aug       █████████████████▌
0.715  Opus slim    ████████████████████ (teacher)
0.716  27B aug      ████████████████████ 🎯 matches teacher
0.738  Opus full    ████████████████████▌
```

Monotonic scaling, no saturation at 27B. Each student-size step adds
5-8 claims-macro points. Augmentation alone adds ~10-12 points at any
size.

### 9.10b Test-set results (held-out, 772 rows, 436 opposition)

Test set was never consulted during development. All numbers below are
reported once, no iteration. Val stats are shown for reference (where
development decisions were made).

| Metric | 4B test | 9B test | **27B test** | Opus slim test | Opus slim val |
|---|---:|---:|---:|---:|---:|
| Detection F1 | 0.851 | 0.853 | **0.894** | **0.894** | 0.939 |
| Frames samples_F1 | 0.697 | 0.696 | 0.781 | **0.793** | 0.815 |
| Frames macro (all) | 0.573 | 0.552 | 0.622 | **0.664** | 0.613 |
| Frames macro (sup≥4) | 0.573 | 0.552 | 0.622 | **0.664** | 0.627 |
| Claims samples_F1 | 0.654 | 0.676 | 0.741 | **0.755** | 0.811 |
| **Claims macro (all)** | 0.490 | 0.517 | **0.578** 🎯 | 0.559 | 0.678 |
| **Claims macro (sup≥4)** | 0.553 | 0.588 | **0.618** 🎯 | 0.615 | 0.715 |
| Frames exact_match | 0.554 | 0.561 | 0.668 | **0.685** | 0.679 |
| Claims exact_match | 0.462 | 0.489 | **0.577** | **0.578** | 0.621 |

### 9.10c The central result

**On held-out test, the distilled 27B student matches or beats the
slim-prompt teacher on its target metric:**

| Metric | 27B aug | Opus slim | Verdict |
|---|---:|---:|---|
| Detection F1 | 0.894 | 0.894 | **TIED** |
| Claims macro (all) | **0.578** | 0.559 | **Student beats by +0.019** |
| Claims macro (sup≥4) | **0.618** | 0.615 | **Student ties** (+0.003) |
| Claims exact_match | 0.577 | 0.578 | TIED |
| Frames samples_F1 | 0.781 | 0.793 | Teacher ahead by 0.012 |
| Frames macro (sup≥4) | 0.622 | 0.664 | Teacher ahead by 0.042 |

**Claims** — the fine-grained 56-way label — is where augmentation was
targeted, and where the student matches the teacher. **Frames** —
8-way coarser — Opus slim still leads by ~0.04; augmentation was less
frame-heavy (1 rare frame + 3 FN-heavy frames vs 42 rare claim codes),
so this is where the remaining work would go.

### 9.10d Val-test generalization check

Both teacher and student drop by a similar amount on claims going
val → test, which rules out "student overfit to val" concerns.

| | Val | Test | Δ |
|---|---:|---:|---:|
| Opus slim claims macro (sup≥4) | 0.715 | 0.615 | **-0.100** |
| 27B aug claims macro (sup≥4) | 0.716 | 0.618 | **-0.098** |

**Identical drop shape** — test is genuinely harder on claims for both
teacher and student (different rare-code distribution, larger sample).
The student's val-test gap is not a development-overfitting artifact.

### 9.11 What this validates

- **Train-support-only positives rule** (the pre-registered rule for
  rare-code augmentation) produced real F1 gains. Not just cosmetic.
- **Val-FP-based negatives rule** reduced over-prediction — the FP codes
  we targeted (N_5, N_6, etc.) appear in the macro_F1 jump.
- **Teacher-quality-gated synthetic data** transferred real training
  signal; distribution shift from Opus-authored text did not materialize
  as a regression (val is formal register; synthetic stayed formal by
  matching seed style).
- **Augmentation scales up monotonically**: 4B → 9B → 27B all show
  consistent gains. 27B closes the teacher gap entirely on claims
  macro_F1 (sup≥4) and essentially matches on opposition-only frames
  macro_F1. No scaling saturation observed.
- **A distilled 27B student can match its teacher** on the hardest
  generalization metric while using a 26% shorter inference prompt.
  That's the central result of this project.

### 9.12 HF branch preservation

**4B**: before the retrain push, pre-augment SHAs were captured for
both LoRA and merged repos:

- `iRanadheer/Windy-Qwen3.5-4B-lora` `revision=baseline` →
  commit `364018d899930eb07218fe9da81e37e7c6ad04d8`.
- `iRanadheer/Windy-Qwen3.5-4B` `revision=baseline` →
  commit `dc78f21d5a5920b2f87d0e9dbedee949d573e797`.
  **Note**: branch creation returned a transient 500 on one attempt;
  the SHA is preserved and can be re-pinned via a later `create_branch`.

**9B**: the pre-aug 9B weights were NOT preserved on HF before retrain.
`iRanadheer/Windy-Qwen3.5-9B` main now holds only the augmented version.
If the pre-aug 9B needs to be reconstructed for comparison, the old
LoRA adapter may still exist in an earlier HF commit or would need to
be re-trained from the pre-augment train.jsonl (accessible via git
history on the dataset repo).

Key implication: all pre-aug 9B comparison numbers in this doc (§9.9b,
§9.10) come from benchmarks that were run on the previous weights
before they were replaced. Numbers are recorded; the model artifact
is not.

---

## 10. Things we tried and abandoned

| Approach | Reason dropped |
|---|---|
| HF Jobs a100-large | Silent failures, no logs returned. Tried multiple times. |
| HF Jobs L40S for 4B | Dependency install issues (tilelang not available at the time). |
| Higher LoRA rank r=32/64 (speculative) | Loss curves show capacity isn't the bottleneck — diagnosis is data-diversity, not rank. Deferred. |
| More epochs (>3) | Train/eval losses plateau by epoch 0.5; no reason to burn more compute. |
| DPO as next step after SFT | Kept in backlog. Correct order is: augment → retrain → re-evaluate → **then** DPO if gap remains. |
| `train_on_responses_only=True` in Unsloth | Removed from script; full-sequence loss works and matches the 27B reference. |

---

## 11. Reproducibility checklist

- [x] All data splits deterministic (seed=42).
- [x] Annotation patches captured in `data/annotation_patches.json`.
- [x] Prompt engineering iterations tracked via codebook diffs in git history.
- [x] Teacher calls deterministic (`temperature=0`).
- [x] Inference deterministic (`temperature=0`).
- [x] Training seed fixed (`seed=42`) and `random_state=42` on LoRA.
- [x] Per-run variance measured: two identical Opus and Sonnet runs
      bound ±0.017 macro_F1 as the noise floor.
- [ ] Versioned HF dataset uploads tagged by commit hash. (TODO)
- [ ] `uv.lock` or pinned PEP 723 versions for byte-identical reruns.
      (TODO — currently float.)

---

## 12. Open questions / TODOs

- How far can the prompt be compressed without hurting the student? The
  current slim (3,603 tokens) matches full. A "tiny" variant (e.g.,
  code list only, no reasoning chain) might test the floor.
- Does synthetic augmentation actually close the teacher–student gap?
  Need to run it and re-benchmark.
- Is `C_27_0`'s weakness a codebook-definition issue or a teacher-label
  inconsistency on training set? Check teacher agreement on C_27_0-labeled
  train rows.
- Does the 4B model show the same slim-prompt-works property, or does
  smaller capacity need the richer prompt?
- DPO as a second-stage pass: preference pairs from teacher/student
  disagreement on val (not test).

---

## 13. Artefacts on HF Hub (as of this doc)

| Repo | Purpose |
|---|---|
| `iRanadheer/wind-opposition-sft` | Dataset: train (slim), train_full backup, test, val |
| `iRanadheer/Windy-Qwen3.5-9B` | **9B slim student (current)** — pushed this run |
| `iRanadheer/Windy-Qwen3.5-9B-lora` | 9B LoRA adapter only |
| `iRanadheer/Windy-Qwen3.5-9B-Turbo` | 9B full-prompt student (renamed from old 9B) |
| `iRanadheer/Windy-Qwen3.5-9B-Turbo-lora` | ... its LoRA adapter |
| `iRanadheer/Windy-Qwen3.5-4B` | 4B slim student (if/when that run completes) |
| `iRanadheer/Windy-Qwen3.5-4B-lora` | ... LoRA adapter |
| `C3DS/Windy-Qwen3.5-9B` | Production-served copy (slim) — used by vLLM at localhost |

---

## 14. Commit history worth citing in the paper

```
a5f2ef1 Consolidate pipeline: unified prompts, scripts, and split definitions
8aca7b7 Add v2 prompt: three-level cascade, tooling, and patched validation results
d0c2bcd before experimenting cards style taxonomy and opus 4.7 results
9c916d3 Sharpen codebook definitions and add parent-subclaim map
4ab0d67 Add GRANULARITY reasoning step and Opus v2 results
78b8923 Simplify SFT script: Windy-Qwen variant naming, seq-len 8192, default merge-and-push
```
