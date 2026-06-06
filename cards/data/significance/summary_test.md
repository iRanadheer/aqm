# CARDS results — test set

Headline metric: **samples-F1** (how well each document is tagged), with a 95% BCa bootstrap confidence interval (`scipy.stats.bootstrap`, 9999 resamples). Micro/macro shown for completeness; macro is lower because rare categories are harder.

## Model scores

| Model | Samples-F1 (95% CI) | Micro-F1 | Macro-F1 |
|---|---|---|---|
| Qwen3.5-4B (base, zero-shot) | 0.579 [0.555, 0.603] | 0.557 | 0.243 |
| Qwen3.5-9B (base, zero-shot) | 0.678 [0.655, 0.701] | 0.679 | 0.365 |
| Qwen3.5-27B (base, zero-shot) | 0.808 [0.789, 0.827] | 0.786 | 0.468 |
| CARDS-Qwen3.5-4B (ours) | 0.777 [0.756, 0.797] | 0.749 | 0.353 |
| CARDS-Qwen3.5-9B (ours) | 0.811 [0.792, 0.830] | 0.779 | 0.372 |
| CARDS-Qwen3.5-27B (ours) | 0.842 [0.824, 0.859] | 0.810 | 0.496 |
| GPT-4o-mini (zero-shot) | 0.634 [0.610, 0.656] | 0.520 | 0.285 |
| CARDS-mini-opus (ours) | 0.840 [0.822, 0.857] | 0.810 | 0.415 |
| Claude Opus 4.7 (zero-shot) | 0.838 [0.820, 0.855] | 0.804 | 0.519 |
| GPT-5.5 (zero-shot) | 0.869 [0.853, 0.885] | 0.836 | 0.529 |

## Comparisons (samples-F1)

Gap = A − B with a 95% CI. `improves`/`lower` = CI clears 0; `comparable` = CI includes 0 (too close to call).

| Comparison | A | B | Gap (95% CI) | Verdict |
|---|---|---|---|---|
| **Fine-tuning helps (open)** | | | | |
| CARDS-4B vs base | 0.777 | 0.579 | +0.198 [+0.174, +0.223] | improves |
| CARDS-9B vs base | 0.811 | 0.678 | +0.133 [+0.111, +0.156] | improves |
| CARDS-27B vs base | 0.842 | 0.808 | +0.034 [+0.018, +0.050] | improves |
| **Fine-tuning helps (closed)** | | | | |
| CARDS-mini-opus vs GPT-4o-mini | 0.840 | 0.634 | +0.207 [+0.183, +0.231] | improves |
| **Does scale help?** | | | | |
| CARDS-9B vs 4B | 0.811 | 0.777 | +0.034 [+0.017, +0.053] | improves |
| CARDS-27B vs 9B | 0.842 | 0.811 | +0.031 [+0.014, +0.047] | improves |
| **RECoT format helps** | | | | |
| CARDS-4B vs No-RECoT | 0.777 | 0.569 | +0.208 [+0.186, +0.232] | improves |
| CARDS-9B vs No-RECoT | 0.811 | 0.312 | +0.499 [+0.472, +0.527] | improves |
| CARDS-27B vs No-RECoT | 0.842 | 0.740 | +0.102 [+0.086, +0.119] | improves |
| **Vs frontier APIs** | | | | |
| CARDS-27B vs Claude Opus 4.7 | 0.842 | 0.838 | +0.004 [-0.011, +0.019] | comparable |
| CARDS-27B vs GPT-5.5 | 0.842 | 0.869 | -0.027 [-0.043, -0.012] | lower |
| CARDS-mini-opus vs Claude Opus 4.7 | 0.840 | 0.838 | +0.002 [-0.014, +0.018] | comparable |
| CARDS-mini-opus vs GPT-5.5 | 0.840 | 0.869 | -0.029 [-0.045, -0.013] | lower |
| **FP8 quantization** | | | | |
| CARDS-27B FP8 vs full | 0.840 | 0.842 | -0.002 [-0.013, +0.008] | comparable |
