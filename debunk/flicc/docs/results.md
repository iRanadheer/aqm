# FLICC results — all models

Twelve-way single-label fallacy classification, benchmarked against Zanartu et
al. 2024 (doi.org/10.1038/s41598-024-76139-w). Splits are theirs, verified
against their Table 8. Parse failures and out-of-vocabulary labels count as
WRONG and are never hand-repaired — stricter than the paper, which hand-assigned
"the most likely label" to 44 of 256 GPT-4 outputs.

Regenerate with `python generate_report.py --split val|test`.

## Modes

| mode | meaning |
|---|---|
| `bare` | Zanartu et al.'s own one-line prompt — twelve label names, nothing else |
| `published` | scaffold + Table 1's codebook, verbatim |
| `revised` | scaffold + a codebook rewritten from the training split |
| `revised + schema` | as `revised`, but the answer is constrained to a JSON schema of {reasoning, fallacy}. Used ONLY for base models, which cannot produce the output contract unaided. Constraining the baseline is the conservative direction — it shrinks the fine-tune's margin. |

## val split (n = 457)

| Model | Kind | Mode | Macro F1 | Accuracy | Unparsed |
|---|---|---|---:|---:|---:|
| FLICC-Qwen3.5-4B | fine-tuned | revised | **0.655** | 0.670 | 0 |
| FLICC-Qwen3.5-27B | fine-tuned | revised | **0.645** | 0.667 | 0 |
| GPT-5.5 | frontier | revised | **0.640** | 0.637 | 0 |
| Claude Opus 4.7 | frontier | revised | **0.632** | 0.654 | 0 |
| FLICC-Qwen3.5-9B | fine-tuned | revised | **0.623** | 0.635 | 0 |
| FLICC-Qwen3.5-2B | fine-tuned | revised | **0.555** | 0.565 | 0 |
| FLICC-Qwen3.5-0.8B | fine-tuned | revised | **0.500** | 0.519 | 0 |
| GPT-5.5 | frontier | published | **0.471** | 0.473 | 0 |
| Qwen3.5-27B | base | revised + schema | **0.469** | 0.499 | 3 |
| GPT-5.5 | frontier | bare | **0.464** | 0.473 | 0 |
| Claude Opus 4.7 | frontier | published | **0.454** | 0.479 | 0 |
| Claude Opus 4.7 | frontier | bare | **0.391** | 0.424 | 0 |
| Qwen3.5-9B | base | revised + schema | **0.362** | 0.383 | 23 |
| Qwen3.5-4B | base | revised + schema | **0.323** | 0.359 | 9 |
| Qwen3.5-2B | base | revised + schema | **0.127** | 0.155 | 0 |
| Qwen3.5-0.8B | base | revised + schema | **0.058** | 0.068 | 156 |
| *DeBERTa-v2-xlarge (Zanartu et al.)* | *baseline* | *fine-tuned* | *0.730* | *0.730* | — |

## test split (n = 256)

Both score columns are **accuracy** on the same 256 items.
Original accuracy uses single-label gold. Review-rule accuracy uses
the expert's full saved labels on the 93 reviewed items: a match to
any saved label counts; an empty set accepts none. Other items retain
the original gold. The DeBERTa run reproduced the published 0.73
macro F1 (0.733 here); the table shows its accuracy, not its F1.

| Model | Kind | Original accuracy | Review-rule accuracy | Change (pp) | Errors reviewed | Rescued / lost |
|---|---|---:|---:|---:|---:|---:|
| DeBERTa-v2-xlarge (reproduced) | reproduced baseline | 0.742 | 0.707 | -3.5 | 39/66 | 13/22 |
| FLICC-Qwen3.5-9B | fine-tuned | 0.695 | 0.738 | +4.3 | 63/78 | 25/14 |
| GPT-5.5 | frontier | 0.656 | 0.758 | +10.2 | 61/88 | 33/7 |
| Claude Opus 4.7 | frontier | 0.656 | 0.750 | +9.4 | 66/88 | 29/5 |
| FLICC-Qwen3.5-27B | fine-tuned | 0.637 | 0.824 | +18.8 | 93/93 | 48/0 |
| FLICC-Qwen3.5-2B | fine-tuned | 0.609 | 0.660 | +5.1 | 68/100 | 23/10 |
| FLICC-Qwen3.5-4B | fine-tuned | 0.574 | 0.637 | +6.2 | 77/109 | 22/6 |
| Qwen3.5-27B (base) | base | 0.520 | 0.617 | +9.8 | 73/123 | 31/6 |
| Claude Opus 4.7 (published prompt) | frontier | 0.480 | 0.543 | +6.2 | 62/133 | 22/6 |
| GPT-5.5 (bare prompt) | frontier | 0.473 | 0.559 | +8.6 | 68/135 | 25/3 |
| GPT-5.5 (published prompt) | frontier | 0.457 | 0.539 | +8.2 | 68/139 | 26/5 |
| FLICC-Qwen3.5-0.8B | fine-tuned | 0.453 | 0.504 | +5.1 | 76/140 | 18/5 |
| Claude Opus 4.7 (bare prompt) | frontier | 0.426 | 0.504 | +7.8 | 68/147 | 24/4 |
| Qwen3.5-9B (base) | base | 0.371 | 0.453 | +8.2 | 81/161 | 23/2 |
| Qwen3.5-4B (base) | base | 0.348 | 0.383 | +3.5 | 79/167 | 15/6 |
| GPT-4-0125 (Zanartu et al.) | published baseline | 0.320 | — | — | — | — |
| Gemini-pro (Zanartu et al.) | published baseline | 0.210 | — | — | — | — |
| Qwen3.5-2B (base) | base | 0.160 | 0.184 | +2.3 | 84/215 | 10/4 |
| Qwen3.5-0.8B (base) | base | 0.105 | 0.094 | -1.2 | 88/229 | 0/3 |

Every model with a review-rule score uses the same 93 items,
selected from the 27B's errors (93/93 covered).
Coverage of each other model's errors is shown in the table; in
particular, DeBERTa has 39/66 covered. Unreviewed errors still use
the original gold, so changes are not a fair adjusted ranking.
The expert confirmed that UI limitations sometimes required clicking
`neither` to enter corrected labels; all saved labels were retained.

## Reading these numbers

The test set has only 256 items, so small differences should not be treated as
stable rankings; model order also changes between validation and test.

- **Every fine-tune at 2B and above beats every base model on original test
  accuracy.** Even the 2B (0.609) exceeds base Qwen3.5-27B (0.520).
- **The revised codebook matters.** Original test accuracy rises from 0.480 to
  0.656 for Opus and from 0.457 to 0.656 for GPT-5.5 when moving from the
  published to the revised prompt.
- **Reference-label choices account for substantial measured error.** Zanartu
  et al. (2024) acknowledge that a passage can contain several fallacies,
  although their benchmark assigns one label. In a blinded expert review of the
  fine-tuned 27B's 93 test disagreements, the expert accepted its prediction on
  48 items and did not retain the original gold label on 45. Gold was also
  accepted on 48: this is evidence of incomplete or disputed *single-label*
  answers, not proof that every rejected label was an annotation mistake.
  These are the 27B's errors, not a random sample of the dataset.
- **The DeBERTa run tests whether the review benefits the published baseline.**
  It reproduces Zanartu et al.'s test result (0.733 macro F1). Among its 66
  original errors, 39 were reviewed: 13 gain credit under the expert's labels.
  Another 22 answers that matched original gold lose credit. Its remaining 27
  errors were not reviewed, so the two review-rule accuracies are not a
  fully adjudicated head-to-head comparison.

## Caveats

- Base models are measured with a JSON schema (reasoning free-form, label
  constrained). `Qwen3.5-0.8B` still fails to parse on ~34% of items: it
  degenerates into whitespace *inside* the JSON string value, where no grammar
  constraint applies. That is a property of the model, not of the protocol.
- The reasoning-length cap (`--reasoning-max-chars`) was 1200 for the 0.8B, 2B
  and 4B base runs and removed for 9B and 27B. It bound on 19% of 2B items and
  29% of 4B items, so those two base numbers are slight underestimates —
  understating the baseline, which is the safe direction.
- Fine-tunes are measured with no constraints at all: thinking on, strict
  parsing. All five produced 0 unparsed and 100% twelve-line SCAN compliance on
  both splits.
- We asked an expert to review, blind to source, the 93 test items on which
  `c3ds-flicc-qwen3-5-27b` disagreed with single-label gold (labels only,
  randomised slots and row order). The expert clicked gold on 30, model on 24,
  both on 11, and neither on 28. The expert confirmed that UI limitations
  required selecting `neither` to enter a corrected label set, even when that
  set included a displayed option. A click is **not** the final label set:
  `expert_labels` supplies the final answer, including on `neither` rows.
- The final sets (`evals/build_multilabel_gold.py`) contain 57 items with one
  label, 22 with two, five with three, two with four, and seven with none.
  Gold and model labels each occur in 48/93 sets. On 18/28 `neither` clicks,
  the final set still contains gold and/or model; 21 reviewed items also have
  labels beyond the two proposed options (28 additional label occurrences).
  We retain **all** final labels. Ten items have notes, which are kept for
  interpretation rather than automatically added to the final label set.
- `evals/score_adjudicated.py` compares original and review-rule **accuracy**.
  The latter accepts a prediction matching *any* final expert label on reviewed
  items (an empty set admits none). Other items keep their original gold. For
  the 27B, 48 of 93 original errors become accepted: accuracy changes from
  163/256 (0.637) to 211/256 (0.824). These items were
  selected from that model's errors, not sampled randomly; other models have
  errors outside the review. The reproduced DeBERTa run has 39/66 errors in
  the reviewed set: 13 errors gain credit, while 22 original matches to gold
  lose it. Both score columns in the test table use accuracy; its coverage
  column shows how much of each model's error was reviewed.
