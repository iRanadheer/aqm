# Design decisions

Why this study is built the way it is. Every claim below is backed by a number
from `data/results/val/metrics_summary.md` or by the upstream data itself.

The target is Zanartu et al. 2024 (doi.org/10.1038/s41598-024-76139-w): a
fine-tuned DeBERTa-v2-xlarge (887M) scoring **macro F1 0.73** on a 256-item test
set, against **0.30** for their prompted GPT-4-0125 and **0.11** for Gemini-pro.

## Results so far — val (457 items), macro F1

| | `bare` | `deconstruct-published` | `deconstruct-revised` | Δ scaffold | Δ codebook |
|---|---:|---:|---:|---:|---:|
| Claude Opus 4.7 | 0.391 | 0.453 | **0.632** | +0.062 | **+0.179** |
| GPT-5.5 | 0.464 | 0.471 | **0.640** | +0.007 | **+0.169** |
| Qwen3.5-0.8B | 0.142 | — | — | | |

Zero unparseable responses on every frontier run. Qwen3.5-9B and Qwen3.5-27B
pending; test not yet run.

**The headline finding is not the one this study set out to find.** The reasoning
scaffold — the paper's own Table 7 method applied at inference — is worth +0.062
on one model and nothing on the other. The *definitions* are worth ~+0.17 on
both. So the result is not about prompting technique:

> Table 1 under-specifies the annotation practice. Correcting it from the
> training split is worth roughly ten times more than the reasoning scaffold,
> and unlike the scaffold it replicates across independent models.

Both models now beat the fine-tuned baseline on three classes — false
equivalence (0.69 / 0.59 vs 0.46), conspiracy theory (0.87 / 0.80 vs 0.80) and
ad hominem (0.82 / 0.77 vs 0.75) — and the gap to 0.73 has fallen from 0.34 at
`bare` to about 0.09.

---

## 1. Splits and labels are theirs, unmodified

`data/raw/*.csv` is vendored verbatim from github.com/fzanart/FLICC.
`prepare_splits.py` verifies every per-class count against the paper's Table 8
and fails loudly on mismatch. The label set is an exact string match to the
upstream CSVs — twelve labels, same spelling, same casing.

This matters because everything else in the study is a comparison against their
published numbers. If the splits or the label space drifted, nothing else would
mean anything.

## 2. Three prompts, each isolating one thing

| variant | contents |
|---|---|
| `bare` | their own prompt — twelve label names, nothing else |
| `deconstruct-published` | scaffold + Table 1's codebook, verbatim |
| `deconstruct-revised` | scaffold + a codebook derived from the training split |

The two deconstruct arms are **byte-identical apart from the codebook block**
(asserted in code). So `published` → `revised` measures the definitions and
nothing else, and `bare` → `published` measures the scaffold.

`bare` is a control, not a result. It cannot decompose their GPT-4 0.30 into
prompt effects versus model effects, because GPT-4-0125 is deprecated. It exists
to show the gain comes from scaffolding rather than from two years of model
progress — and on that it is informative: GPT-5.5 scores **0.464** on their
prompt against their reported 0.30, so newer models alone buy about +0.16 and
leave 0.27 unexplained.

An earlier `defs` rung (definitions, no scaffold) was cut. Its instruction block
already carried most of the scaffold's reasoning guidance, so it never separated
cleanly from `deconstruct`, and a rung that doesn't isolate anything is not an
ablation.

## 3. The scaffold is the paper's own method — and it is worth less than expected

`deconstruct` asks the model to reconstruct the argument — conclusion, stated
premises, hidden premise — then attribute the error to one element, then scan
the labels. That is Table 7 of the paper, which is how the annotators worked.
Using it at inference time is applying their methodology, not inventing one, and
a reviewer can check it against the published table.

It was motivated by the dominant error mode, not by taste. On `bare`, **44–46% of
all errors** were a structurally-defined fallacy being absorbed by one of four
residual labels (cherry picking, misrepresentation, oversimplification, slothful
induction), and the proportion was near-identical across Opus 4.7 and GPT-5.5 —
a property of the task, not of a model. Read at the surface, almost any climate
text looks like selective use of evidence. The labels live in the hidden
premise: nine of Table 7's twelve worked examples attribute the fallacy there.

**Measured against Table 1's codebook, it buys +0.062 on Opus and +0.007 on
GPT-5.5.** The reading that fits: GPT-5.5 already outscored Opus on `bare`
(0.464 vs 0.391), so instructing it to reconstruct the argument mostly formalises
something it was already doing. The scaffold is retained — it costs little, it
is the paper's own method, and it is the frame in which the SCAN step (§7) sits —
but it is not where the value is, and the writeup should not imply otherwise.

## 4. No word-based cues — because they did not replicate

An earlier revision of the codebook carried trigger phrases for the classes we
knew were failing. It scored well and was removed anyway.

| | `bare` | with cues | cue-free |
|---|---:|---:|---:|
| Opus 4.7 | 0.391 | 0.547 | 0.466 |
| GPT-5.5 | 0.464 | 0.525 | 0.519 |

The scaffold replicates: +0.075 and +0.055 on two independent models. **The cues
do not**: +0.081 on Opus, +0.006 on GPT-5.5. A gain that appears in one model
and vanishes in another is not a property of the task — it was patching an
Opus-specific blind spot on two classes. Reporting it would have meant
publishing a number another lab could not reproduce with a different model.

The reviewer objection is the secondary reason and it is real: a keyword rule
for a class you already know is failing is a lookup table, not a finding. But
the replication failure is the one that decided it.

## 5. No rules telling the model which label to prefer

Two mechanisms were tried and removed:

- **Entry gates** — "before using cherry picking you must point to cited
  evidence", and similar for the other three residual labels.
- **Precedence** — prefer the eight structural labels over the four residual
  ones.

Both were brakes on the absorption problem in §3, and both substitute the
author's expectations for the model's judgement. They also demonstrably harm:
the slothful-induction gate cost **−0.19** on Opus and **−0.17** on GPT-5.5 by
suppressing correct answers, and loosening it returned **+0.13**.

More fundamentally, they were a patch for the wrong cause. Table 1 does not
separate those four labels — "misrepresentation: distorts understanding" and
"oversimplification: distorts understanding" are nearly the same sentence. The
model was reasoning correctly from definitions that do not discriminate. Fixing
the definitions (§6) is the repair; a brake is a symptom-level workaround. If a
brake turns out to be load-bearing, that is evidence the definitions are still
wrong, and the right response is to keep working on them.

## 6. The revised codebook, and why it is defensible

Table 1 does not fully specify the annotation practice. Three mismatches are
large enough to name, all verifiable by reading the published table against the
published data:

- **`single cause`.** Table 1's structure is *"X caused Y; therefore X was the
  only cause of Y."* Training rows labelled `single cause` include *"frost fairs
  were held on the Thames when the river froze"* and *"The Sun's radiation
  output waxes and wanes periodically in predictable cycles"* — sentences that
  assert no cause at all. The operative pattern is invoking natural variation so
  as to displace human causation, with the causal claim left implicit.
- **`impossible expectations`.** Table 1 describes only demands for more
  research. The dominant training pattern is dismissing a measure because its
  individual effect is small.
- **`fake experts`.** Table 1 requires an *unqualified* source. Training rows
  include appeals to genuinely credentialled dissenters and to petition
  signatory counts. The operative idea is appeal to dissenting-expert authority,
  not lack of standing.

This is a narrow claim about reproducibility, not an attack. Table 1 is a
summary table in a Nature-family paper; annotators almost certainly worked from
a fuller guideline. The statement is: *as published, Table 1 is not sufficient
to reproduce the annotation.*

**Discipline.** Every revision was derived by reading the **training split
only** — never val, never test. Train is the 1,796 rows the baseline fitted on,
so using it costs nothing in fairness, and it keeps val a clean development
signal and test untouched. Revisions state argument patterns at category level;
no example phrases, no keyword lists (§4).

**Falsifiability.** The claim is settled empirically, not rhetorically: does the
revised codebook predict *their* labels better than their own codebook, same
model, same scaffold? If `revised` does not beat `published`, the claim fails
and Table 1 stands. Reported either way.

**Result: the condition is met, on both models.** `revised` beats `published` by
**+0.179** on Opus 4.7 and **+0.169** on GPT-5.5 — an effect an order of
magnitude larger than the scaffold's, and one that replicates where the word
cues of §4 did not. Opus improved on **12 of 12** classes, GPT-5.5 on 11 of 12
(slothful induction, −0.01).

The three named mismatches are the three biggest movers, which is the part that
distinguishes a diagnosis from a lucky edit:

| class | Opus Δ | GPT-5.5 Δ | predicted in advance? |
|---|---:|---:|---|
| impossible expectations | **+0.44** | +0.29 | yes |
| single cause | +0.26 | +0.24 | yes |
| oversimplification | +0.30 | +0.27 | no |
| false equivalence | +0.03 | +0.30 | no |
| fake experts | +0.04 | +0.11 | yes (n=12, noisy) |

`impossible expectations` now *beats* the fine-tuned baseline on Opus (0.79 vs
0.71) after being one of the worst classes in the study.

**Standing caveat.** The revised codebook was derived from train and is scored
here on val, so val is now partly a selection surface. Test is the real check
and some regression should be expected.

## 7. Every label is scanned, every time

Step 5 of the scaffold walks all twelve labels in codebook order, one KEEP or
REMOVE line each. This is coverage, not steering: it does not tell the model
what to conclude, only that it may not skip a label by never considering it.

The hole was real. On `bare`, Qwen3.5-0.8B predicted `anecdote` and `slothful
induction` **zero times in 457 items**, and Opus predicted `single cause` ten
times against 57 gold instances. It also removes an observed failure where the
model listed a single candidate and rubber-stamped it. Both frontier models
comply on 100% of items, median twelve lines.

## 8. Strict parsing, with deterministic recovery

Predictions that are missing, unparseable, or outside the twelve-label
vocabulary count as **wrong**. Never repaired, never dropped. This is harsher
than the paper, which hand-assigned "the most likely label" to 44 of 256 GPT-4
outputs and 8 Gemini outputs — a per-item human judgement on 17% of the test
set, which inflates their LLM baseline by an unquantifiable amount.

`extract_raw_label` adds three fallbacks beyond the output contract, documented
in its docstring. They are rules, not judgements, and applied to every run
alike: they changed **0 predictions on every frontier run**, so they can only
ever help a weak one. Whatever they cannot reach still counts as wrong.

This deliberately diverges from `debunk/prompts.py`. Consistency with the
sibling study is not worth scoring a small model's format failure as if it were
a classification failure.

`generate_report.py` and `evals/significance.py` both re-parse from the stored
`response` rather than trusting the `pred_label` written at generation time, so
parser fixes apply retroactively to runs already on disk and the two scripts
cannot drift apart on what a run predicted.

## 9. Zero-shot only; no few-shot

No labelled example ever enters a prompt. Few-shot and retrieval-based ICL are
the subject of the `icl` chapter; including them here would duplicate that
contribution. The convention that zero-shot cannot reach is supplied instead by
fine-tuning (§10), which is also the paper's own stated next step: *"a more
sustainable, long-term approach might involve fine-tuning a LLM."*

## 10. Where the ceiling is, and what comes next

Prompting reaches the fallacies identifiable from argument structure and stalls
on the rest. With `deconstruct-revised` measured, the boundary is sharp. Both
models beat the fine-tuned baseline on ad hominem, conspiracy theory and false
equivalence, and Opus beats it on impossible expectations. What remains far
below, on both models, is:

| class | Opus | GPT-5.5 | DeBERTa | Table 1 calls it |
|---|---:|---:|---:|---|
| misrepresentation | 0.39 | 0.34 | 0.63 | background knowledge |
| slothful induction | 0.48 | 0.45 | 0.65 | background knowledge |
| cherry picking | 0.61 | 0.56 | 0.67 | structural |
| false choice | 0.40 | 0.52 | 0.80 | structural (n=13) |
| fake experts | 0.71 | 0.81 | 0.96 | structural (n=12) |

Two of the three classes Table 1 itself marks as requiring **background
knowledge** are the two that resist every prompt we have written. That is the
paper's own prediction holding: those fallacies are not visible in the form of
the argument, so no definition reaches them — you have to know what the body of
evidence says. `false choice` and `fake experts` have support of 13 and 12, so
they are noisy and disproportionately expensive under macro F1.

What is left is annotation convention, and the honest way to transfer convention
is to train on the labelled data the baseline trained on: teacher traces over
the 1,796 training rows, then LoRA SFT. A labels-only control tests the paper's
own thesis — whether the deconstruction is what makes the dataset valuable, or
whether the labels alone suffice.

## 11. Protocol

Prompt development happens on **val** (457 items). **Test** (256 items) is run
once, per (model, variant), after the prompts are frozen. Significance is a 95%
BCa bootstrap, 9,999 resamples, seed 42; comparisons against the published 0.73
are one-sample and labelled as weaker than a paired test, because their model is
not re-run and there are no paired predictions on their side.
