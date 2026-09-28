# Wind results — test set

Three tasks: binary **opposition detection** (all rows), and **frames**/**claims** multi-label tagging on **opposition-only** rows (where they apply). For frames/claims we report **Samples-F1** (per-row, support-independent — agreed-empty/non-opposition rows count as correct) and **Macro-F1** restricted to gold codes with support ≥ 3 (CARDS convention; hallucinated codes dropped). Each value is F1 with a 95% BCa bootstrap CI (`scipy.stats.bootstrap`, 9999 resamples). Parse failures / API errors are penalised, matching `generate_report.py`.

## Model scores

| Model | Detection F1 | Frames Samples-F1 (opp) | Frames Macro-F1 (opp, sup≥3) | Claims Samples-F1 (opp) | Claims Macro-F1 (opp, sup≥3) |
|---|---|---|---|---|---|
| Qwen3.5-4B (base, zero-shot) | 0.711 [0.675, 0.744] | 0.454 [0.414, 0.496] | 0.446 [0.391, 0.509] | 0.434 [0.395, 0.471] | 0.408 [0.373, 0.459] |
| Qwen3.5-9B (base, zero-shot) | 0.782 [0.749, 0.811] | 0.465 [0.428, 0.504] | 0.512 [0.461, 0.568] | 0.496 [0.459, 0.534] | 0.461 [0.425, 0.518] |
| Qwen3.5-27B (base, zero-shot) | 0.842 [0.813, 0.867] | 0.663 [0.624, 0.699] | 0.593 [0.543, 0.650] | 0.599 [0.564, 0.635] | 0.558 [0.521, 0.613] |
| Windy-Qwen3.5-4B (ours) | 0.851 [0.825, 0.874] | 0.699 [0.663, 0.731] | 0.631 [0.582, 0.683] | 0.623 [0.588, 0.655] | 0.571 [0.539, 0.634] |
| Windy-Qwen3.5-9B (ours) | 0.853 [0.827, 0.876] | 0.695 [0.658, 0.729] | 0.606 [0.554, 0.662] | 0.660 [0.627, 0.691] | 0.614 [0.577, 0.680] |
| Windy-Qwen3.5-27B (ours) | 0.894 [0.871, 0.914] | 0.747 [0.711, 0.778] | 0.659 [0.606, 0.713] | 0.675 [0.641, 0.708] | 0.642 [0.605, 0.698] |
| Windy-Qwen3.5-27B FP8 (ours) | 0.898 [0.876, 0.918] | 0.751 [0.715, 0.783] | 0.664 [0.612, 0.717] | 0.694 [0.661, 0.726] | 0.648 [0.609, 0.708] |
| CARDS-Wind-Qwen3.6-27B (ours, joint) | 0.886 [0.863, 0.907] | 0.729 [0.693, 0.762] | 0.646 [0.594, 0.697] | 0.675 [0.640, 0.707] | 0.622 [0.584, 0.678] |
| Claude Opus 4.7 (zero-shot) | 0.893 [0.870, 0.913] | 0.734 [0.696, 0.766] | 0.692 [0.646, 0.738] | 0.667 [0.632, 0.700] | 0.635 [0.600, 0.694] |
| GPT-5.5 (zero-shot) | 0.885 [0.860, 0.906] | 0.697 [0.660, 0.733] | 0.660 [0.614, 0.708] | 0.614 [0.577, 0.649] | 0.608 [0.573, 0.663] |

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

## Comparisons — Frames Samples-F1 (opp)

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

## Comparisons — Frames Macro-F1 (opp, sup≥3)

Gap = A − B with a 95% CI. `improves`/`lower` = CI clears 0; `comparable` = CI includes 0.

| Comparison | A | B | Gap (95% CI) | Verdict |
|---|---|---|---|---|
| **Fine-tuning helps** | | | | |
| Windy-4B vs base | 0.631 | 0.446 | +0.185 [+0.120, +0.252] | improves |
| Windy-9B vs base | 0.606 | 0.512 | +0.094 [+0.038, +0.150] | improves |
| Windy-27B vs base | 0.659 | 0.593 | +0.066 [+0.008, +0.126] | improves |
| **Does scale help?** | | | | |
| Windy-9B vs 4B | 0.606 | 0.631 | -0.025 [-0.074, +0.017] | comparable |
| Windy-27B vs 9B | 0.659 | 0.606 | +0.053 [+0.012, +0.097] | improves |
| **Vs frontier APIs** | | | | |
| Windy-27B vs Claude Opus 4.7 | 0.659 | 0.692 | -0.033 [-0.087, +0.012] | comparable |
| Windy-27B vs GPT-5.5 | 0.659 | 0.660 | -0.001 [-0.058, +0.048] | comparable |
| CARDS-Wind-27B vs Claude Opus 4.7 | 0.646 | 0.692 | -0.046 [-0.100, +0.002] | comparable |
| CARDS-Wind-27B vs GPT-5.5 | 0.646 | 0.660 | -0.014 [-0.071, +0.033] | comparable |
| **Joint vs Wind-only** | | | | |
| CARDS-Wind-27B vs Windy-27B | 0.646 | 0.659 | -0.013 [-0.045, +0.017] | comparable |
| **FP8 quantization** | | | | |
| Windy-27B FP8 vs full | 0.664 | 0.659 | +0.005 [-0.022, +0.028] | comparable |
| CARDS-Wind-27B FP8 vs full | 0.657 | 0.646 | +0.011 [-0.014, +0.042] | comparable |

## Comparisons — Claims Samples-F1 (opp)

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

## Comparisons — Claims Macro-F1 (opp, sup≥3)

Gap = A − B with a 95% CI. `improves`/`lower` = CI clears 0; `comparable` = CI includes 0.

| Comparison | A | B | Gap (95% CI) | Verdict |
|---|---|---|---|---|
| **Fine-tuning helps** | | | | |
| Windy-4B vs base | 0.571 | 0.408 | +0.163 [+0.117, +0.219] | improves |
| Windy-9B vs base | 0.614 | 0.461 | +0.152 [+0.102, +0.216] | improves |
| Windy-27B vs base | 0.642 | 0.558 | +0.083 [+0.043, +0.130] | improves |
| **Does scale help?** | | | | |
| Windy-9B vs 4B | 0.614 | 0.571 | +0.042 [+0.003, +0.087] | improves |
| Windy-27B vs 9B | 0.642 | 0.614 | +0.028 [-0.018, +0.068] | comparable |
| **Vs frontier APIs** | | | | |
| Windy-27B vs Claude Opus 4.7 | 0.642 | 0.635 | +0.006 [-0.039, +0.050] | comparable |
| Windy-27B vs GPT-5.5 | 0.642 | 0.608 | +0.034 [-0.005, +0.075] | comparable |
| CARDS-Wind-27B vs Claude Opus 4.7 | 0.622 | 0.635 | -0.013 [-0.056, +0.024] | comparable |
| CARDS-Wind-27B vs GPT-5.5 | 0.622 | 0.608 | +0.014 [-0.040, +0.062] | comparable |
| **Joint vs Wind-only** | | | | |
| CARDS-Wind-27B vs Windy-27B | 0.622 | 0.642 | -0.020 [-0.062, +0.014] | comparable |
| **FP8 quantization** | | | | |
| Windy-27B FP8 vs full | 0.648 | 0.642 | +0.007 [-0.021, +0.029] | comparable |
| CARDS-Wind-27B FP8 vs full | 0.618 | 0.622 | -0.004 [-0.030, +0.025] | comparable |
