# Wind results — test set

Three tasks: binary **opposition detection** (all rows), and **frames**/**claims** multi-label F1 on **opposition-only** rows (where they apply). Each value is F1 with a 95% BCa bootstrap CI (`scipy.stats.bootstrap`, 9999 resamples). Parse failures / API errors are penalised, matching `generate_report.py`.

## Model scores

| Model | Detection F1 (95% CI) | Frames F1, opp (95% CI) | Claims F1, opp (95% CI) |
|---|---|---|---|
| Qwen3.5-4B (base, zero-shot) | 0.711 [0.675, 0.744] | 0.454 [0.414, 0.496] | 0.434 [0.395, 0.471] |
| Qwen3.5-9B (base, zero-shot) | 0.782 [0.749, 0.811] | 0.465 [0.428, 0.504] | 0.496 [0.459, 0.534] |
| Qwen3.5-27B (base, zero-shot) | 0.842 [0.813, 0.867] | 0.663 [0.624, 0.699] | 0.599 [0.564, 0.635] |
| Windy-Qwen3.5-4B (ours) | 0.851 [0.825, 0.874] | 0.699 [0.663, 0.731] | 0.623 [0.588, 0.655] |
| Windy-Qwen3.5-9B (ours) | 0.853 [0.827, 0.876] | 0.695 [0.658, 0.729] | 0.660 [0.627, 0.691] |
| Windy-Qwen3.5-27B (ours) | 0.894 [0.871, 0.914] | 0.747 [0.711, 0.778] | 0.675 [0.641, 0.708] |
| Windy-Qwen3.5-27B FP8 (ours) | 0.898 [0.876, 0.918] | 0.751 [0.715, 0.783] | 0.694 [0.661, 0.726] |
| CARDS-Wind-Qwen3.6-27B (ours, joint) | 0.886 [0.863, 0.907] | 0.729 [0.693, 0.762] | 0.675 [0.640, 0.707] |
| Claude Opus 4.7 (zero-shot) | 0.893 [0.870, 0.913] | 0.734 [0.696, 0.766] | 0.667 [0.632, 0.700] |
| GPT-5.5 (zero-shot) | 0.885 [0.860, 0.906] | 0.697 [0.660, 0.733] | 0.614 [0.577, 0.649] |

## Comparisons — Detection F1

Gap = A − B with a 95% CI. `improves`/`lower` = CI clears 0; `comparable` = CI includes 0.

| Comparison | A | B | Gap (95% CI) | Verdict |
|---|---|---|---|---|
| **Fine-tuning helps** | | | | |
| Windy-4B vs base | 0.851 | 0.711 | +0.140 [+0.110, +0.172] | improves |
| Windy-9B vs base | 0.853 | 0.782 | +0.070 [+0.044, +0.100] | improves |
| Windy-27B vs base | 0.894 | 0.842 | +0.052 [+0.032, +0.074] | improves |
| **Does scale help?** | | | | |
| Windy-9B vs 4B | 0.853 | 0.851 | +0.002 [-0.016, +0.019] | comparable |
| Windy-27B vs 9B | 0.894 | 0.853 | +0.041 [+0.021, +0.062] | improves |
| **Vs frontier APIs** | | | | |
| Windy-27B vs Claude Opus 4.7 | 0.894 | 0.893 | +0.001 [-0.018, +0.019] | comparable |
| Windy-27B vs GPT-5.5 | 0.894 | 0.885 | +0.009 [-0.010, +0.029] | comparable |
| CARDS-Wind-27B vs Claude Opus 4.7 | 0.886 | 0.893 | -0.007 [-0.025, +0.011] | comparable |
| CARDS-Wind-27B vs GPT-5.5 | 0.886 | 0.885 | +0.001 [-0.019, +0.022] | comparable |
| **Joint vs Wind-only** | | | | |
| CARDS-Wind-27B vs Windy-27B | 0.886 | 0.894 | -0.008 [-0.022, +0.007] | comparable |
| **FP8 quantization** | | | | |
| Windy-27B FP8 vs full | 0.898 | 0.894 | +0.004 [-0.005, +0.014] | comparable |
| CARDS-Wind-27B FP8 vs full | 0.891 | 0.886 | +0.005 [-0.006, +0.016] | comparable |

## Comparisons — Frames F1 (opp)

Gap = A − B with a 95% CI. `improves`/`lower` = CI clears 0; `comparable` = CI includes 0.

| Comparison | A | B | Gap (95% CI) | Verdict |
|---|---|---|---|---|
| **Fine-tuning helps** | | | | |
| Windy-4B vs base | 0.699 | 0.454 | +0.245 [+0.203, +0.285] | improves |
| Windy-9B vs base | 0.695 | 0.465 | +0.230 [+0.192, +0.269] | improves |
| Windy-27B vs base | 0.747 | 0.663 | +0.084 [+0.050, +0.117] | improves |
| **Does scale help?** | | | | |
| Windy-9B vs 4B | 0.695 | 0.699 | -0.004 [-0.029, +0.021] | comparable |
| Windy-27B vs 9B | 0.747 | 0.695 | +0.052 [+0.024, +0.081] | improves |
| **Vs frontier APIs** | | | | |
| Windy-27B vs Claude Opus 4.7 | 0.747 | 0.734 | +0.013 [-0.014, +0.042] | comparable |
| Windy-27B vs GPT-5.5 | 0.747 | 0.697 | +0.050 [+0.020, +0.081] | improves |
| CARDS-Wind-27B vs Claude Opus 4.7 | 0.729 | 0.734 | -0.005 [-0.034, +0.024] | comparable |
| CARDS-Wind-27B vs GPT-5.5 | 0.729 | 0.697 | +0.031 [+0.000, +0.064] | improves |
| **Joint vs Wind-only** | | | | |
| CARDS-Wind-27B vs Windy-27B | 0.729 | 0.747 | -0.018 [-0.041, +0.004] | comparable |
| **FP8 quantization** | | | | |
| Windy-27B FP8 vs full | 0.751 | 0.747 | +0.004 [-0.011, +0.020] | comparable |
| CARDS-Wind-27B FP8 vs full | 0.739 | 0.729 | +0.010 [-0.004, +0.027] | comparable |

## Comparisons — Claims F1 (opp)

Gap = A − B with a 95% CI. `improves`/`lower` = CI clears 0; `comparable` = CI includes 0.

| Comparison | A | B | Gap (95% CI) | Verdict |
|---|---|---|---|---|
| **Fine-tuning helps** | | | | |
| Windy-4B vs base | 0.623 | 0.434 | +0.189 [+0.150, +0.228] | improves |
| Windy-9B vs base | 0.660 | 0.496 | +0.163 [+0.127, +0.201] | improves |
| Windy-27B vs base | 0.675 | 0.599 | +0.076 [+0.046, +0.108] | improves |
| **Does scale help?** | | | | |
| Windy-9B vs 4B | 0.660 | 0.623 | +0.037 [+0.010, +0.064] | improves |
| Windy-27B vs 9B | 0.675 | 0.660 | +0.016 [-0.011, +0.042] | comparable |
| **Vs frontier APIs** | | | | |
| Windy-27B vs Claude Opus 4.7 | 0.675 | 0.667 | +0.009 [-0.017, +0.036] | comparable |
| Windy-27B vs GPT-5.5 | 0.675 | 0.614 | +0.061 [+0.032, +0.093] | improves |
| CARDS-Wind-27B vs Claude Opus 4.7 | 0.675 | 0.667 | +0.008 [-0.019, +0.035] | comparable |
| CARDS-Wind-27B vs GPT-5.5 | 0.675 | 0.614 | +0.060 [+0.032, +0.093] | improves |
| **Joint vs Wind-only** | | | | |
| CARDS-Wind-27B vs Windy-27B | 0.675 | 0.675 | -0.001 [-0.021, +0.018] | comparable |
| **FP8 quantization** | | | | |
| Windy-27B FP8 vs full | 0.694 | 0.675 | +0.018 [+0.003, +0.035] | improves |
| CARDS-Wind-27B FP8 vs full | 0.677 | 0.675 | +0.003 [-0.013, +0.021] | comparable |
