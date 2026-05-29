# Debunk — veracityV2 (4-class)

Single-label fact-checking verdict vs `true_veracity` on n=160 claims. Headline metric: **accuracy** with a 95% BCa bootstrap CI (`scipy.stats.bootstrap`, 9999 resamples). MCC and macro-F1 (present classes) are point estimates. Parse failures / API errors count as wrong. **Small n → wide CIs**, so many gaps are `comparable` from limited power.

## Model scores

| Model | Accuracy (95% CI) | MCC | Macro-F1 |
|---|---|---|---|
| *Majority baseline* | 0.775 | 0.000 | — |
| Claude Opus 4.7 (offline) | 0.869 [0.806, 0.912] | 0.625 | 0.762 |
| Claude Opus 4.7 + RAG | 0.856 [0.794, 0.906] | 0.582 | 0.723 |
| DeepSeek V4 Flash (offline) | 0.750 [0.675, 0.812] | 0.438 | 0.662 |
| DeepSeek V4 Flash + RAG | 0.756 [0.688, 0.819] | 0.421 | 0.672 |
| GPT-4o-mini (offline) | 0.756 [0.681, 0.819] | 0.397 | 0.623 |
| GPT-4o-mini + RAG | 0.819 [0.756, 0.875] | 0.509 | 0.700 |
| GPT-5.5 (offline) | 0.844 [0.781, 0.894] | 0.578 | 0.746 |
| GPT-5.5 + RAG | 0.825 [0.762, 0.881] | 0.521 | 0.707 |
| Qwen3.5-9B (offline) | 0.812 [0.750, 0.869] | 0.465 | 0.626 |
| Qwen3.5-9B + RAG | 0.525 [0.444, 0.600] | 0.180 | 0.492 |
| Qwen3.5-27B (offline) | 0.844 [0.781, 0.894] | 0.549 | 0.700 |
| Qwen3.5-27B + RAG | 0.850 [0.787, 0.900] | 0.570 | 0.725 |

## Comparisons (accuracy)

Gap = A − B with a 95% CI. `improves`/`lower` = CI clears 0; `comparable` = CI includes 0 (too close to call).

| Comparison | A | B | Gap (95% CI) | Verdict |
|---|---|---|---|---|
| **RAG helps** | | | | |
| Claude Opus 4.7: +RAG vs offline | 0.856 | 0.869 | -0.013 [-0.056, +0.019] | comparable |
| DeepSeek V4 Flash: +RAG vs offline | 0.756 | 0.750 | +0.006 [-0.075, +0.081] | comparable |
| GPT-4o-mini: +RAG vs offline | 0.819 | 0.756 | +0.062 [+0.000, +0.131] | comparable |
| GPT-5.5: +RAG vs offline | 0.825 | 0.844 | -0.019 [-0.069, +0.025] | comparable |
| Qwen3.5-9B: +RAG vs offline | 0.525 | 0.812 | -0.287 [-0.369, -0.206] | lower |
| Qwen3.5-27B: +RAG vs offline | 0.850 | 0.844 | +0.006 [-0.044, +0.050] | comparable |
| **Does scale help? (Qwen)** | | | | |
| 27B vs 9B (offline) | 0.844 | 0.812 | +0.031 [-0.019, +0.081] | comparable |
| 27B vs 9B (+RAG) | 0.850 | 0.525 | +0.325 [+0.244, +0.406] | improves |
