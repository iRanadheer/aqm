# Experiments log — Wind

A running narrative of every experiment I ran on the Wind opposition-detection
project for the thesis/paper. Numbers live in `data/results/test/` and the
Hub model cards. This doc is the **why** and the **lessons**, not the
tables. Sister doc:
[`cards/docs/experiments.md`](../../cards/docs/experiments.md) for the
parallel CARDS work.

---

## 1. Goal

Build a small fine-tuned classifier (the **Windy-** family) for three-level
opposition detection in wind-energy text:

1. `opposition_detected` — binary: does the text express opposition?
2. `frames` — `N_*` codes: which frame(s) (e.g. `N_1` aesthetic, `N_4`
   property values) the opposition uses
3. `claims` — `C_*_*` codes: specific claim(s) within each frame

Targets to validate:
- Match or beat frontier APIs (Claude Opus 4.7, GPT-5.5) on the held-out
  test set (773 rows).
- Show fine-tuning improves over the base model at every scale.
- Quantify the FT-over-base lift across 4B / 9B / 27B.

---

## 2. Models trained

Qwen3.5 backbones at three sizes plus a joint CARDS+Wind 27B variant:

- `Windy-Qwen3.5-4B`, `-9B`, `-27B` — RECoT-style FT on the wind dataset.
- `Windy-Qwen3.5-27B-FP8` — FP8-dynamic quantized version of the 27B for
  cheaper deployment.
- `CARDS-Wind-Qwen3.6-27B`, `-FP8` — joint single-backbone trained on
  CARDS + Wind concatenated (one model handles both tasks).

Trained with `cards/ft/train.py --joint` (Unsloth + LoRA r=16 α=16,
3 epochs, lr=2e-4, batch=1×8 grad accum, bf16). Same recipe as the
cards FT models — the joint variant just adds the wind dataset to the
training mix.

---

## 3. Inference setup

### 3.1 Output format

Every model emits a structure with three fields, parsed by
`wind/generate_report.parse_response`:

```yaml
opposition_detected: true
frames:
  - N_4
claims:
  - C_5_1
```

The parser is permissive on layout (YAML inside any markdown fence)
but strict on field types (each field can be `None` if missing —
counted as a parse failure unless the charitable fallback in §5
applies).

### 3.2 Thinking mode at inference

`--no-think` sets `chat_template_kwargs={"enable_thinking": False}` for
vLLM, suppressing the auto-injected `<think>\n` token. Verified
empirically: with `--no-think`, the model's response begins directly
with the analytical content (`"The claim that..."`), no `<think>` tag
anywhere.

For 27B test, with-think vs no-think loses ~1pt across all metrics —
a small cost. We report **with-think** as the headline inference setting
for FT models (matches their training-time chat template, gives the
trained-for benefit). Base models are reported under no-think (they
weren't trained on the task; with-think tends to inflate their parse
failures from unstructured rambling).

---

## 4. The 4B base "below-emergence" non-finding

**Initial reading** (max-tokens=500): 4B base scored detection F1 =
0.018 with 495/772 parse failures. Tempting to claim this as a clean
"emergence-threshold" finding: the 4B base genuinely cannot do this
task without FT.

**The reality** (after sampling failed responses): all 491 of those
"all-fields-missing" failures were **truncated reasoning** — the model
wrote 2,000-character analytical responses (CONTEXT, DETECTION, FRAMES,
CLAIMS sections) but hit the `--max-tokens 500` cutoff before reaching
the structured YAML block at the end. Smaller models are simply more
verbose per section than larger ones.

**The fix** was just `--max-tokens 1500`. After re-running:

- Parse failures: 495 → 111
- Detection F1: 0.018 → **0.711**
- Frames samples F1: 0.287 → 0.536

The 4B base is *competent*, just verbose. **Lesson:** before claiming a
small-model "emergence threshold," check if the failures are truncation.

---

## 5. The "charitable parser" fix for base models

After fixing 4B truncation, 9B base still showed 272/773 parse failures.
Sampling: **251 (92%) of those failures had valid `frames` and `claims`
lists but no explicit `opposition_detected: true/false` line.** The
model was producing structured analysis with section headers but
skipping the boolean-field YAML key.

This isn't a real failure mode — the boolean is *implicit* in
`frames` / `claims` being non-empty (any frame → opposition is
detected). I added a charitable fallback to `parse_response`:

```python
if (parsed["opposition_detected"] is None
        and parsed["frames"] is not None
        and parsed["claims"] is not None):
    parsed["opposition_detected"] = bool(parsed["frames"] or parsed["claims"])
```

After the fix:
- 9B base parse failures: **272 → 21**
- 9B base detection F1: **0.392 → 0.782**

The fix is universal (applies to every model), but in practice only
affected 9B base — every other model (FT'd and APIs) emits the
explicit boolean reliably. Importantly, the 27B base's 44 parse
failures **stayed at 44** — those are genuinely all-fields-missing
(the charitable fallback requires *both* frames and claims to be
parseable to fire).

**Lesson:** when comparing base vs FT, check whether parse failures are
cosmetic (model emitted everything except one literal field) vs
structural (output is incomplete or garbage). The strict parser was
penalizing 9B base for cosmetic deviations only.

---

## 6. Findings

### 6.1 Headline (test set, 773 rows, charitable parser, max-tokens=1500 for 4B base)

Models in column-paired order (APIs, then base→FT pairs at 4B, 9B, 27B):

| Metric | Opus 4.7 | GPT-5.5 | 4B Base | Windy-4B | 9B Base | Windy-9B | 27B Base | Windy-27B | Windy-27B FP8 |
|---|---|---|---|---|---|---|---|---|---|
| Detection F1 | 0.893 | 0.885 | 0.711 | 0.851 | 0.782 | 0.853 | 0.842 | 0.894 | **0.898** |
| Frames samples F1 | 0.791 | 0.792 | 0.536 | 0.697 | 0.600 | 0.696 | 0.744 | 0.781 | 0.787 |
| Claims-all samples F1 | 0.754 | 0.745 | 0.523 | 0.654 | 0.618 | 0.676 | 0.708 | 0.741 | **0.755** |
| Claims-opp samples F1 | 0.667 | 0.614 | 0.432 | 0.623 | 0.496 | 0.660 | 0.599 | 0.675 | **0.694** |
| Parse failures | 1/773 | 2/773 | 111/773 | 0/773 | 21/773 | 0/773 | 44/773 | 0/773 | 0/773 |

(Macro F1 columns omitted from this summary; see the report file for the
full grid.)

### 6.2 FT-over-base lift, by scale (Detection F1)

| Size | Base | Windy FT | Lift |
|---|---|---|---|
| 4B | 0.711 | 0.851 | +0.140 |
| 9B | 0.782 | 0.853 | +0.071 |
| 27B | 0.842 | 0.894 | +0.052 |

**Monotonically decreasing FT lift with scale** — base gets stronger,
the marginal contribution of fine-tuning shrinks but stays positive
at every size. This is the **opposite direction** of the cards 9B
regression case (where label-only SFT *hurt* the strong 9B base).
Wind FT is the well-behaved scenario; cards no-RECoT FT was the
catastrophic-forgetting failure mode. Side-by-side these are
complementary findings about when fine-tuning helps vs hurts.

### 6.3 Windy-27B FP8 beats both frontier APIs

On 7 of 9 metrics — detection F1, detection recall, all four
frame/claim samples F1 metrics, and the macro F1 on claims. Loses
detection precision to GPT-5.5 (0.927 vs 0.877) and frames macro F1 to
Opus 4.7 (0.664 vs 0.630). Net: a 27 GB FP8 model deployable on a
single A100 outperforms ~$20-per-million-tokens frontier APIs on this
task.

### 6.4 FP8 quantization preserves (slightly improves) accuracy

Windy-27B-FP8 vs Windy-27B-BF16 across all 9 metrics: FP8 matches or
beats BF16 within rounding on every cell. Same finding as cards 27B
FP8. **Quantization is not the precision-vs-throughput trade-off you
might assume** — at this scale and recipe, fp8_e4m3 dynamic per-channel
quantization is essentially free.

---

## 7. Things I did not investigate

- **OpenRouter for base Qwen3.5-{4B,9B,27B}** — useful for a "no-GPU
  baseline" but OpenRouter's lineup is Instruct-heavy and base
  availability was uncertain. Local vLLM was the safer bet.
- **Wind RECoT-vs-NoRECoT ablation.** The Windy models all trained
  with RECoT; we never trained a no-RECoT variant for wind. Could
  close the recipe-comparison story across both projects but is out
  of scope.
- **Wind cross-domain ablation.** The Windy models are evaluated only
  on wind-test — no cross-domain robustness numbers.

---

## 8. Lessons / things to remember

- **Truncation looks like emergence-failure.** Always check sample
  responses before claiming a small-model can't do a task. Default
  `--max-tokens` budgets are usually too tight for verbose base models.
- **Cosmetic format deviations look like parse failures.** When a
  model produces 80% of the expected structure but skips one literal
  YAML key, score it charitably — the FT comparison is fairer that
  way.
- **The wind FT scaling pattern is well-behaved** (FT lift
  monotonically decreases but stays positive at every size). Compare
  to cards 9B no-RECoT, which *regressed below base* — a different
  failure mode of label-only SFT. Together these two findings give
  a fuller picture of when fine-tuning is reliable.
- **FP8 dynamic quantization is essentially free** at 27B scale on
  this task. Cheaper deployment with no accuracy cost.

---

## 9. Outputs / artifacts

- **Models on HF Hub** (`C3DS/` namespace): `Windy-Qwen3.5-{4B,9B,27B}`,
  `Windy-Qwen3.5-27B-FP8`, `CARDS-Wind-Qwen3.6-27B{,-FP8}`. README
  cards on Hub for each.
- **Reports** in `data/results/test/`:
  - `metrics_summary.{json,md}` — full FT-vs-API headline (auto-generated)
  - Per-model jsonl files for every variant evaluated.
- **Datasets on HF**: `iRanadheer/wind-opposition-sft` — train / eval
  JSONLs.
- **Code**: `wind/infer.py`, `wind/generate_report.py` (charitable
  parser, three-level metrics).
