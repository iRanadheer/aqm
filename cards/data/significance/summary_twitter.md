# CARDS results — twitter set

Headline metric: **samples-F1** (how well each document is tagged), with a 95% BCa bootstrap confidence interval (`scipy.stats.bootstrap`, 9999 resamples). Micro/macro shown for completeness; macro is lower because rare categories are harder.

## Model scores

| Model | Samples-F1 (95% CI) | Micro-F1 | Macro-F1 |
|---|---|---|---|
| CARDS-Qwen3.5-4B (ours) | 0.557 [0.526, 0.590] | 0.550 | 0.434 |
| CARDS-Qwen3.5-9B (ours) | 0.577 [0.546, 0.609] | 0.577 | 0.479 |
| CARDS-Qwen3.5-27B (ours) | 0.643 [0.612, 0.672] | 0.645 | 0.535 |
| GPT-4o-mini (zero-shot) | 0.465 [0.433, 0.497] | 0.413 | 0.288 |
| CARDS-mini-opus (ours) | 0.628 [0.596, 0.657] | 0.618 | 0.515 |
| Claude Opus 4.7 (zero-shot) | 0.692 [0.663, 0.719] | 0.676 | 0.597 |
| GPT-5.5 (zero-shot) | 0.689 [0.659, 0.716] | 0.670 | 0.590 |

## Comparisons (samples-F1)

Gap = A − B with a 95% CI. `improves`/`lower` = CI clears 0; `comparable` = CI includes 0 (too close to call).

| Comparison | A | B | Gap (95% CI) | Verdict |
|---|---|---|---|---|
| **Fine-tuning helps (closed)** | | | | |
| CARDS-mini-opus vs GPT-4o-mini | 0.628 | 0.465 | +0.163 [+0.127, +0.198] | improves |
| **Does scale help?** | | | | |
| CARDS-9B vs 4B | 0.577 | 0.557 | +0.020 [-0.008, +0.049] | comparable |
| CARDS-27B vs 9B | 0.643 | 0.577 | +0.066 [+0.037, +0.095] | improves |
| **Vs frontier APIs** | | | | |
| CARDS-27B vs Claude Opus 4.7 | 0.643 | 0.692 | -0.048 [-0.074, -0.024] | lower |
| CARDS-27B vs GPT-5.5 | 0.643 | 0.689 | -0.045 [-0.072, -0.019] | lower |
| CARDS-mini-opus vs Claude Opus 4.7 | 0.628 | 0.692 | -0.064 [-0.090, -0.039] | lower |
| CARDS-mini-opus vs GPT-5.5 | 0.628 | 0.689 | -0.061 [-0.087, -0.035] | lower |
| **FP8 quantization** | | | | |
| CARDS-27B FP8 vs full | 0.652 | 0.643 | +0.008 [-0.008, +0.026] | comparable |
