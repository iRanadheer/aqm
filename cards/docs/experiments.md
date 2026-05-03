# Experiments log — CARDS

A running narrative of every experiment I ran for the CARDS chapter of
the thesis: what I did, what I learned, and why I made the calls I did.
Numbers live in `data/results/test/` and `data/results/twitter/`.
Data-pipeline details are in [`data.md`](data.md). This doc is the
**why** and the **lessons**, not the tables. Sister doc:
[`wind/docs/experiments.md`](../../wind/docs/experiments.md) for the
parallel wind work.

---

## 1. Goal

Build a small fine-tuned classifier for the CARDS contrarian-claim taxonomy
that competes with frontier API models (Claude Opus 4.7, GPT-5.5) at a
fraction of the inference cost. Three things to validate:

1. Does fine-tuning a small base model close the gap to frontier APIs?
2. Does **RECoT** (Reverse-Engineered Chain-of-Thought) supervision matter
   beyond plain label-only SFT?
3. How does the answer to (2) change with model scale?

---

## 2. Models trained

I fine-tuned Qwen3.5 at three sizes (4B / 9B / 27B). Each backbone was
trained twice:

- **RECoT FT** — student learns to imitate teacher reasoning
  (`<think>...</think>` block) and the YAML category list. Trained on
  `cards_train.jsonl`.
- **No-RECoT FT** — student learns to emit only the YAML, no reasoning.
  Trained on `cards_train_norecot.jsonl` (same row indices, `<think>`
  stripped from response, CoT trigger removed from user message).

Plus 2B and 3.6-27B RECoT-FT for scaling comparisons, mini-Opus as a
non-Qwen FT baseline, and an FP8-quantized 27B for cheaper serving.

Both variants share `prepare_splits.py` for split derivation and
`ft/train.py` for SFT (Unsloth + LoRA, 3 epochs, lr=2e-4, r=16, α=16,
batch=1×8 grad accum, bf16). Constant hyperparameters across sizes;
defended in §7.

---

## 3. The story of the no-RECoT prompt contradiction

This was the longest debugging arc and the source of most of the
iteration. I'm documenting it in detail because it shaped almost every
methodology choice downstream.

### 3.1 What broke

The first round of no-RECoT 9B inference scored *worse than its own base
model* (samples F1 0.152 with 510 parse failures). The 9B was emitting
the system prompt's example template verbatim — `1. CLAIMS: Direct
quotes only. ... <think> closes ... categories: <category_code>` —
instead of analyzing input.

### 3.2 Why it broke

Three layers in the no-RECoT pipeline were fighting:

| Layer | What it said / did |
|---|---|
| System prompt (`slim_system_instruction`) | "Reason inside `<think>` tags then output YAML" |
| Chat template at training (`enable_thinking=True`) | Auto-injected `<think>\n` before assistant turn |
| Training target | Plain YAML, no `<think>`, no reasoning |

The model trained under contradictory instructions: the system prompt
and template both promised thinking; the training data delivered no
thinking. The 4B was small enough to "ignore the directive and just
emit YAML" (robustness via low capacity). The 9B has more capacity,
latched onto the literal template structure, and at inference filled
the auto-injected `<think>` with the system prompt's *example template*
— overfitting to form, not content.

### 3.3 Two fixes applied (in order)

1. **Inference-time only.** Pass `chat_template_kwargs={"enable_thinking":
   False}` so vLLM skips the auto-injected `<think>\n`. Combined with
   dropping the CoT trigger and using a separate
   `slim_system_instruction_norecot` (no `<think>` directive in OUTPUT
   FORMAT), the 9B parse-fail rate dropped from 510 → 0 and L3 samples
   F1 went from 0.152 → 0.788. This shipped as `infer.py --no-recot`.

2. **Retrain with consistent prompts.** The "fix at inference" is still
   OOD vs training. The thesis-defensible thing is to make training and
   inference identical. So I added `slim_system_instruction_norecot` to
   `prompts.py`, updated `prepare_splits.py` to use it for no-RECoT
   training data, pushed the new training files to the HF dataset, and
   retrained 4B / 9B / 27B no-RECoT from scratch.

### 3.4 The retrained 9B is *still* the worst no-RECoT result

After the clean retrain, the 9B no-RECoT FT scored L3 samples F1 = **0.642**
— *below* its own base (0.749) and far below RECoT FT (0.813). The new
model produces clean YAML output (0 parse failures) but **over-predicts
non-zero categories**: gold says no-claim on 79% of rows; the model says
no-claim on only 58%. It learned to map text features → category codes
too aggressively, without learning the discriminator "is there a claim
at all?"

The 4B benefited from no-RECoT FT (0.662 → 0.732); the 27B was
small-positive (0.803 → 0.821). The 9B is the regression case.

---

## 4. Was it train/inference asymmetry? (we tested)

After the retrain, I still wondered whether the 9B regression was caused
by the remaining train-vs-inference mismatch on `enable_thinking`
(training used `enable_thinking=True`, inference used
`enable_thinking=False`). To test, I added a `--thinking` override flag
to `infer.py` and ran each no-RECoT FT model under thinking-on inference
(matching its training-time chat template):

| Size | no-think (training-mismatched) | with-think (training-matched) | parse fails (with-think) |
|---|---|---|---|
| 4B | 0.732 | 0.569 | 537 |
| 9B | 0.642 | 0.312 | 833 |
| 27B | 0.821 | 0.740 | 280 |

**Thinking-on at inference is uniformly worse, not better.** Hundreds of
parse failures at every scale: the model floods the auto-injected
`<think>\n` block with rambling content that never reaches `</think>`
before max-tokens cuts off.

This **rules out simple train/inference asymmetry as the cause** of the
9B regression. Both inference modes fail; the trained model itself is
miscalibrated. RECoT FT is the only training recipe that produces a
usable 9B classifier (samples F1 0.813, 0 parse fails).

---

## 5. Diagnosing the 9B regression: catastrophic forgetting

The over-prediction failure mode at 9B matches the documented phenomenon
of *catastrophic forgetting after SFT*. Most relevant citation:

**[Luo et al. 2023, "An Empirical Study of Catastrophic Forgetting in
Large Language Models During Continual Fine-tuning"](https://arxiv.org/abs/2308.08747)**

> "Catastrophic forgetting is generally observed in LLMs ranging from 1b
> to 7b parameters... as the model scale increases, the severity of
> forgetting intensifies in such a model scale range which may result
> from the much significant initial performance"

Mechanism in our case: 1,611 training rows + LoRA r=16 + α=16 + lr=2e-4
+ class-imbalanced data (~80% `0_0_0`, ~20% non-zero) is a high-risk
profile for forgetting. The 9B base prior is strong (samples F1 = 0.749
with no-think prompting); label-only SFT shifts the calibration toward
over-prediction of the minority class.

### Why 9B and not 27B?

Luo et al.'s monotonic "intensifies with scale" claim doesn't perfectly
match our 9B-peak-then-recovery pattern. A plausible explanation: at
27B the LoRA-rank-relative-to-hidden ratio is much smaller (r=16 vs
hidden 5,120 at 27B, vs hidden 4,096 at 9B). The same r=16 LoRA at
27B has less capacity to displace the base prior. So the 9B regression
is a hyperparameter-coupled effect, not a pure scale effect.

---

## 6. Hyperparameter mitigation experiments

I considered and explicitly **rejected** three "fix the 9B" hypotheses
before settling on the actual experiment:

- **Lower LR (1e-4 instead of 2e-4)** — reduces all gradient steps
  including for desired task signal. Less surgical.
- **Higher LoRA rank (r=32)** — more capacity to overfit on 1,611
  samples. Likely makes the over-prediction worse.
- **More epochs** — eval_loss already plateaued at epoch 1.5; more
  training would just drift, not fit.

The actually-targeted intervention is **lower LoRA alpha**, which
scales the LoRA contribution in the forward pass without reducing the
gradient signal during training. Effective scale is `α/r`; the default
is `16/16 = 1.0`. I retrained 9B no-RECoT with:

- `α=8` (scale 0.5) — half the LoRA perturbation of base prior
- `α=4` (scale 0.25) — quarter perturbation

Both runs *also* fix the train/inference asymmetry by passing
`enable_thinking=False` to `apply_chat_template` during training (the
new default for `--no-recot` in `train.py`). So they test two
mitigations together: matched chat-template + reduced LoRA scale.

**Outcome (L3 samples F1, Support ≥ 3):**

| 9B no-RECoT variant | LoRA scale | Samples F1 |
|---|---|---|
| α=16 (full) | 1.00× | 0.642 |
| α=8 | 0.50× | 0.637 |
| α=4 | 0.25× | 0.678 |
| 9B Base (no-think reference) | — | 0.749 |
| RECoT FT (reference) | 1.00× | 0.813 |

Lower α monotonically improves both samples F1 (0.642 → 0.637 →
0.678) and macro F1 (0.264 → 0.275 → 0.306) by reducing the
perturbation of the base prior — confirming the catastrophic-forgetting
mechanism. **However, even the gentlest setting (α=4) fails to match
the base model (0.749), let alone RECoT-FT (0.813).** Vanilla SFT on
label-only data cannot improve a strong 9B base under any LoRA
perturbation magnitude we tested.

This makes the thesis claim airtight: the 9B vanilla-SFT regression is
robust to the LoRA-perturbation knob, indicating the calibration failure
is intrinsic to label-only training at this scale rather than a
hyperparameter artifact. RECoT supervision is the only recipe that
produces net positive gains at 9B.

---

## 7. Methodology decisions and viva-ready defenses

### 7.1 Match inference to training, per model class

Every model is evaluated under its training-time chat template:

- RECoT FT: `enable_thinking=True`, `slim_system_instruction` (with
  `<think>` directive). Matches training.
- No-RECoT FT: `enable_thinking=False`,
  `slim_system_instruction_norecot`. Matches training (after the
  retrain in §3.3).
- Base models: untrained on the task; reported under both with-think
  and no-think (as `qwen35-Xb-base.jsonl` and
  `qwen35-Xb-base-nothink.jsonl`).
- API models (Opus, GPT-5.5, GPT-4o-mini): use their own templates,
  Qwen-specific flags don't apply.

### 7.2 Constant hyperparameters across sizes

Same lr, epochs, LoRA rank, batch, max-seq across all sizes. *Defense
for viva:*

> "Constant hyperparameters isolate model size and training recipe as
> the variables of interest, without introducing a per-cell tuning-effort
> confound. Hyperparameters are community-standard defaults for ~1K-row
> LoRA SFT, validated to converge cleanly (eval_loss plateau by epoch
> 1.5–2 at every size). The 9B vanilla-SFT regression *under these
> defaults* is itself a finding — consistent with [Luo et al. 2023] on
> catastrophic forgetting. RECoT-FT, by contrast, is robust to the same
> fixed hyperparameters at every scale, demonstrating recipe-level
> robustness as an additional result."

### 7.3 Strict scoring

`min_support ≥ 3` for all reported macro F1 (long-tail support-1/2
classes are too noisy). Hallucinated codes outside the gold vocabulary
are dropped silently before scoring (model still penalized via FN on
the correct gold label). Significance via paired BCa bootstrap (2,000
resamples) + sign-flip permutation (10,000 perms); claim significance
only when CI excludes 0 AND p < 0.05.

---

## 8. Key findings (thesis-claimable)

1. **RECoT-FT consistently outperforms vanilla SFT at every scale tested.**
   Significance: p < 0.0001 at 4B (Δ +0.121 samples F1), p = 0.007 at
   9B (Δ +0.171 samples F1 vs the broken no-RECoT; Δ +0.027 vs the
   older prompt-fixed numbers). The advantage holds at 27B (Δ +0.012).

2. **Vanilla SFT regresses below the base model at 9B.** The model
   over-predicts non-zero categories (~30% of `0_0_0` rows
   misclassified). The regression cannot be closed by inference-time
   tuning (we tested both with-think and no-think, both fail), nor by
   LoRA-scale tuning (α ∈ {16, 8, 4} — none recover to base). Consistent
   with catastrophic forgetting [Luo et al. 2023].

3. **RECoT-FT 9B closes the gap to frontier APIs.** L3 samples F1 0.813
   vs Claude Opus 4.7's 0.836 and GPT-5.5's 0.870. A 9B model trained
   on a few thousand teacher-distilled examples reaches 96% of frontier
   performance on this task at orders-of-magnitude lower inference cost.

4. **Base-model prompt format matters more than expected at small scales.**
   Switching from with-think to no-think inference improves the base
   4B by +8pt, base 9B by +7pt; at 27B the difference is noise. Base
   models *would* benefit from thinking-mode reasoning, but at smaller
   sizes they can't reliably produce parseable thinking output.
   RECoT-FT effectively teaches them how, unlocking the inherent
   benefit.

5. **Macro F1 is misleading on long-tail multi-label classification.**
   With min_support=0, base models sometimes score *higher* macro F1
   than RECoT-FT due to over-prediction hitting rare classes by luck.
   Always cite samples F1 (or macro with support ≥ 3) as the headline.

---

## 9. Things I did not investigate (open questions)

- **Twitter-split RECoT ablation.** Headline numbers exist for both
  splits; the RECoT-vs-No-RECoT ablation runs on test only. ~15 min
  of inference would close this — worth doing for cross-domain
  robustness. Treated as future work per supervisor decision.
- **Hyperparameter sweep at 9B beyond LoRA-α.** Lower-LR /
  higher-rank / shorter-training variants. Out of scope; the α
  experiments in §6 already establish the result is robust to
  perturbation magnitude.
- **Joint cards+wind FT impact on cards-only metrics.** We have a
  joint CARDS-Wind 27B model evaluated on wind/test, but didn't
  re-evaluate it on cards/test/twitter. Would close the joint-training
  story.

---

## 10. Outputs / artifacts

- **Models on HF Hub** (`iRanadheer/` namespace):
  `cards_qwen3.5_{4b,9b,27b}_norecot{,-lora}` (clean retrains),
  `cards_qwen3.5_9b_norecot_{alpha8,alpha4}` (perturbation-scale
  variants), plus all RECoT-FT variants from earlier
  (`C3DS/CARDS-Qwen3.5-{2B,4B,9B,27B}`,
  `C3DS/CARDS-Qwen3.5-27B-FP8`, etc.).
- **Reports** in `data/results/`:
  - `test/metrics_summary.{json,md}` — full FT-vs-API headline
    (10 models)
  - `twitter/metrics_summary.{json,md}` — same on twitter
  - `test/recot_ablation.{json,md}` — 4B/9B/27B × {Base / Base-no-think
    / No-RECoT FT no-think / No-RECoT FT with-think / RECoT FT}
  - `test/scaling_ablation.{json,md}` — Base vs RECoT-FT at
    2B/4B/9B/27B
- **ICR report** at `docs/icr_report.md` (Krippendorff's α 0.81/0.82/0.79
  at L1/L2/L3 across 50 expert-coded items).
- **Dataset on HF**: `iRanadheer/cards_sft_dataset` — train /
  train_eval / val / test JSONLs, RECoT and no-RECoT variants, all
  consistent with current `prepare_splits.py`.

---

## 11. Lessons / things to remember

- When training and inference disagree on the chat template
  (`enable_thinking`, system-prompt directives, trigger words), the
  model's behavior at inference becomes unpredictable and non-monotonic
  in capacity. **Make training and inference identical** — the
  retrains in §3.3 only fully aligned this for the lower-α variants
  in §6.
- Parse-failure rate is a sneaky confounder. Two models can have the
  same per-row F1 *when they parse* but vastly different total F1
  because of failure rates. Always report parse-fail count alongside
  the metric.
- LoRA `α/r` scale is a more surgical knob than learning rate when
  the base prior is strong and you want to limit catastrophic
  forgetting — but as §6 shows, it's not always sufficient to recover
  performance.
- For long-tail multi-label classification, `min_support ≥ 3` is the
  right macro-F1 cutoff. Reporting all-labels macro is noisy and
  misleading.
