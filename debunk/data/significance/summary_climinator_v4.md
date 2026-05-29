# Debunk — climinator_v4 (12-class)

Single-label fact-checking verdict vs `true_cfb_label` on n=160 claims. Headline metric: **accuracy** with a 95% BCa bootstrap CI (`scipy.stats.bootstrap`, 9999 resamples). MCC and macro-F1 (present classes) are point estimates. Parse failures / API errors count as wrong. **Small n → wide CIs**, so many gaps are `comparable` from limited power.

## Model scores

| Model | Accuracy (95% CI) | MCC | Macro-F1 |
|---|---|---|---|
| *Majority baseline* | 0.350 | 0.000 | — |
| Claude Opus 4.7 (offline) | 0.375 [0.300, 0.450] | 0.218 | 0.175 |
| Claude Opus 4.7 + RAG | 0.406 [0.331, 0.487] | 0.268 | 0.197 |
| DeepSeek V4 Flash (offline) | 0.338 [0.269, 0.412] | 0.163 | 0.174 |
| DeepSeek V4 Flash + RAG | 0.369 [0.294, 0.444] | 0.204 | 0.161 |
| GPT-4o-mini (offline) | 0.294 [0.225, 0.369] | 0.099 | 0.102 |
| GPT-4o-mini + RAG | 0.294 [0.225, 0.369] | 0.083 | 0.160 |
| GPT-5.5 (offline) | 0.381 [0.306, 0.456] | 0.226 | 0.233 |
| GPT-5.5 + RAG | 0.431 [0.356, 0.506] | 0.289 | 0.282 |
| Qwen3.5-9B (offline) | 0.294 [0.225, 0.369] | 0.079 | 0.105 |
| Qwen3.5-9B + RAG | 0.294 [0.225, 0.369] | 0.093 | 0.085 |
| Qwen3.5-27B (offline) | 0.394 [0.319, 0.469] | 0.220 | 0.191 |
| Qwen3.5-27B + RAG | 0.356 [0.287, 0.431] | 0.160 | 0.112 |
| Paper CLIM (recomputed) | 0.325 [0.256, 0.400] | 0.162 | 0.163 |

## Comparisons (accuracy)

Gap = A − B with a 95% CI. `improves`/`lower` = CI clears 0; `comparable` = CI includes 0 (too close to call).

| Comparison | A | B | Gap (95% CI) | Verdict |
|---|---|---|---|---|
| **RAG helps** | | | | |
| Claude Opus 4.7: +RAG vs offline | 0.406 | 0.375 | +0.031 [-0.019, +0.081] | comparable |
| DeepSeek V4 Flash: +RAG vs offline | 0.369 | 0.338 | +0.031 [-0.031, +0.094] | comparable |
| GPT-4o-mini: +RAG vs offline | 0.294 | 0.294 | +0.000 [-0.081, +0.081] | comparable |
| GPT-5.5: +RAG vs offline | 0.431 | 0.381 | +0.050 [+0.006, +0.106] | improves |
| Qwen3.5-9B: +RAG vs offline | 0.294 | 0.294 | +0.000 [-0.050, +0.050] | comparable |
| Qwen3.5-27B: +RAG vs offline | 0.356 | 0.394 | -0.037 [-0.088, +0.013] | comparable |
| **Does scale help? (Qwen)** | | | | |
| 27B vs 9B (offline) | 0.394 | 0.294 | +0.100 [+0.037, +0.163] | improves |
| 27B vs 9B (+RAG) | 0.356 | 0.294 | +0.062 [+0.006, +0.125] | improves |
| **Vs paper baseline (best config = +RAG)** | | | | |
| Claude Opus 4.7 +RAG vs Paper | 0.406 | 0.325 | +0.081 [+0.019, +0.150] | improves |
| DeepSeek V4 Flash +RAG vs Paper | 0.369 | 0.325 | +0.044 [-0.019, +0.112] | comparable |
| GPT-4o-mini +RAG vs Paper | 0.294 | 0.325 | -0.031 [-0.119, +0.056] | comparable |
| GPT-5.5 +RAG vs Paper | 0.431 | 0.325 | +0.106 [+0.044, +0.175] | improves |
| Qwen3.5-9B +RAG vs Paper | 0.294 | 0.325 | -0.031 [-0.094, +0.031] | comparable |
| Qwen3.5-27B +RAG vs Paper | 0.356 | 0.325 | +0.031 [-0.031, +0.094] | comparable |
