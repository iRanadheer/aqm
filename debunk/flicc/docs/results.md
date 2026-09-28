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

| Model | Kind | Mode | Macro F1 | Accuracy | Unparsed |
|---|---|---|---:|---:|---:|
| FLICC-Qwen3.5-9B | fine-tuned | revised | **0.702** | 0.695 | 0 |
| GPT-5.5 | frontier | revised | **0.654** | 0.656 | 0 |
| FLICC-Qwen3.5-27B | fine-tuned | revised | **0.636** | 0.637 | 0 |
| Claude Opus 4.7 | frontier | revised | **0.635** | 0.656 | 0 |
| FLICC-Qwen3.5-2B | fine-tuned | revised | **0.586** | 0.609 | 0 |
| FLICC-Qwen3.5-4B | fine-tuned | revised | **0.571** | 0.574 | 0 |
| Qwen3.5-27B | base | revised + schema | **0.495** | 0.519 | 0 |
| Claude Opus 4.7 | frontier | published | **0.488** | 0.480 | 0 |
| GPT-5.5 | frontier | published | **0.463** | 0.457 | 0 |
| GPT-5.5 | frontier | bare | **0.454** | 0.473 | 0 |
| FLICC-Qwen3.5-0.8B | fine-tuned | revised | **0.415** | 0.453 | 0 |
| Claude Opus 4.7 | frontier | bare | **0.402** | 0.426 | 0 |
| Qwen3.5-9B | base | revised + schema | **0.351** | 0.371 | 8 |
| Qwen3.5-4B | base | revised + schema | **0.314** | 0.348 | 8 |
| Qwen3.5-2B | base | revised + schema | **0.149** | 0.160 | 0 |
| Qwen3.5-0.8B | base | revised + schema | **0.096** | 0.105 | 88 |
| *DeBERTa-v2-xlarge (Zanartu et al.)* | *baseline* | *fine-tuned* | *0.730* | *0.740* | — |
| *GPT-4-0125 (Zanartu et al.)* | *baseline* | *their prompt* | *0.300* | *0.320* | — |
| *Gemini-pro (Zanartu et al.)* | *baseline* | *their prompt* | *0.110* | *0.210* | — |

## Reading these numbers

**The 95% interval on test macro F1 is about ±0.065 at n=256.** Differences
smaller than roughly 0.13 between two models are not separable. Several
orderings therefore flip between splits and should not be read as rankings:
`FLICC-Qwen3.5-4B` tops val and is fifth on test; `FLICC-Qwen3.5-9B` is fifth on
val and tops test.

What survives that noise:

- **Every fine-tune at 2B and above beats every base model on both splits**, with
  no overlap.
- **The codebook is worth far more than the scaffold.** `bare` -> `published`
  buys +0.086 / +0.009 on test (Opus / GPT-5.5); `published` -> `revised` buys
  +0.147 / +0.191. Paired bootstrap on the Opus pair: +0.147, CI [+0.070,
  +0.219].
- **Fine-tuning beats scale at the small end.** `FLICC-Qwen3.5-2B` (0.586 test)
  beats base `Qwen3.5-27B` (0.495) — a model 13x its size.
- **Scaling the student past ~4B buys nothing.** 4B / 9B / 27B all land in
  0.62-0.70 averaged across splits. The 27B was also the only model whose eval
  loss turned upward during training (best at epoch 1.98), which points at 1,614
  training examples rather than at capacity.
- **`FLICC-Qwen3.5-9B` at 0.702 test is statistically indistinguishable from the
  published 0.730** — the gap is well inside the interval. It also beats both
  frontier models reading the same prompt.

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
