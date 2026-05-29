# CARDS results — test set

Headline metric: **samples-F1** (how well each document is tagged), with a 95% BCa bootstrap confidence interval (`scipy.stats.bootstrap`, 9999 resamples). Micro/macro shown for completeness; macro is lower because rare categories are harder.

## Model scores

| Model | Samples-F1 (95% CI) | Micro-F1 | Macro-F1 |
|---|---|---|---|
| Qwen3.5-4B (base, zero-shot) | 0.579 [0.555, 0.603] | 0.557 | 0.243 |
| Qwen3.5-9B (base, zero-shot) | 0.678 [0.655, 0.701] | 0.679 | 0.365 |
| Qwen3.5-27B (base, zero-shot) | 0.805 [0.786, 0.824] | 0.792 | 0.467 |
| CARDS-Qwen3.5-4B (ours) | 0.781 [0.760, 0.801] | 0.765 | 0.371 |
| CARDS-Qwen3.5-9B (ours) | 0.813 [0.793, 0.832] | 0.791 | 0.379 |
| CARDS-Qwen3.5-27B (ours) | 0.833 [0.815, 0.851] | 0.812 | 0.487 |
| GPT-4o-mini (zero-shot) | 0.634 [0.610, 0.656] | 0.539 | 0.296 |
| CARDS-mini-opus (ours) | 0.836 [0.818, 0.854] | 0.815 | 0.434 |
| Claude Opus 4.7 (zero-shot) | 0.835 [0.817, 0.852] | 0.809 | 0.527 |
| GPT-5.5 (zero-shot) | 0.869 [0.853, 0.885] | 0.843 | 0.531 |

## Comparisons (samples-F1)

Gap = A − B with a 95% CI. `improves`/`lower` = CI clears 0; `comparable` = CI includes 0 (too close to call).

| Comparison | A | B | Gap (95% CI) | Verdict |
|---|---|---|---|---|
| **Fine-tuning helps (open)** | | | | |
| CARDS-4B vs base | 0.781 | 0.581 | +0.200 [+0.175, +0.225] | improves |
| CARDS-9B vs base | 0.813 | 0.681 | +0.132 [+0.109, +0.155] | improves |
| CARDS-27B vs base | 0.833 | 0.805 | +0.028 [+0.012, +0.044] | improves |
| **Fine-tuning helps (closed)** | | | | |
| CARDS-mini-opus vs GPT-4o-mini | 0.836 | 0.634 | +0.203 [+0.179, +0.227] | improves |
| **Does scale help?** | | | | |
| CARDS-9B vs 4B | 0.813 | 0.781 | +0.032 [+0.015, +0.051] | improves |
| CARDS-27B vs 9B | 0.833 | 0.813 | +0.020 [+0.003, +0.037] | improves |
| **RECoT format helps** | | | | |
| CARDS-4B vs No-RECoT | 0.781 | 0.569 | +0.212 [+0.188, +0.235] | improves |
| CARDS-9B vs No-RECoT | 0.813 | 0.311 | +0.503 [+0.475, +0.530] | improves |
| CARDS-27B vs No-RECoT | 0.833 | 0.733 | +0.100 [+0.084, +0.117] | improves |
| **Vs frontier APIs** | | | | |
| CARDS-27B vs Claude Opus 4.7 | 0.833 | 0.835 | -0.002 [-0.017, +0.012] | comparable |
| CARDS-27B vs GPT-5.5 | 0.833 | 0.869 | -0.036 [-0.053, -0.021] | lower |
| CARDS-mini-opus vs Claude Opus 4.7 | 0.836 | 0.835 | +0.001 [-0.015, +0.016] | comparable |
| CARDS-mini-opus vs GPT-5.5 | 0.836 | 0.869 | -0.033 [-0.049, -0.017] | lower |
| **FP8 quantization** | | | | |
| CARDS-27B FP8 vs full | 0.835 | 0.833 | +0.002 [-0.008, +0.013] | comparable |
