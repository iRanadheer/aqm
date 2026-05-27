# Debunk — test set

| Model | veracityV2 MCC | veracityV2 Acc | veracityV2 F1 | climinator MCC | climinator Acc | climinator F1 | climinator_v2 MCC | climinator_v2 Acc | climinator_v2 F1 | climinator_v3 MCC | climinator_v3 Acc | climinator_v3 F1 | climinator_v4 MCC | climinator_v4 Acc | climinator_v4 F1 | climinator_v5 MCC | climinator_v5 Acc | climinator_v5 F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Majority baseline* | 0.000 | 0.775 | — | 0.000 | 0.350 | — | 0.000 | 0.350 | — | 0.000 | 0.350 | — | 0.000 | 0.350 | — | 0.000 | 0.350 | — |
| Claude Opus 4.7 offline | 0.625 | 0.869 | 0.762 | — | — | — | — | — | — | — | — | — | 0.218 | 0.375 | 0.175 | — | — | — |
| Claude Opus 4.7 + RAG (pplx-ctx) | 0.582 | 0.856 | 0.723 | — | — | — | — | — | — | — | — | — | 0.268 | 0.406 | 0.197 | — | — | — |
| GPT-4o-mini offline | 0.397 | 0.756 | 0.623 | — | — | — | — | — | — | — | — | — | 0.099 | 0.294 | 0.102 | — | — | — |
| GPT-4o-mini + RAG (pplx-ctx) | 0.509 | 0.819 | 0.700 | — | — | — | — | — | — | — | — | — | 0.083 | 0.294 | 0.160 | — | — | — |
| DeepSeek V4 Flash offline | 0.438 | 0.750 | 0.662 | — | — | — | — | — | — | — | — | — | 0.163 | 0.338 | 0.174 | 0.199 | 0.369 | 0.234 |
| DeepSeek V4 Flash + RAG (pplx-ctx) | 0.421 | 0.756 | 0.672 | — | — | — | — | — | — | — | — | — | 0.204 | 0.369 | 0.161 | 0.189 | 0.369 | 0.157 |
| GPT-5.5 offline | 0.578 | 0.844 | 0.746 | — | — | — | — | — | — | — | — | — | 0.226 | 0.381 | 0.233 | — | — | — |
| GPT-5.5 + RAG (pplx-ctx) | 0.521 | 0.825 | 0.707 | — | — | — | — | — | — | — | — | — | 0.289 | 0.431 | 0.282 | — | — | — |
| Qwen3.5-9B offline | 0.465 | 0.812 | 0.626 | — | — | — | — | — | — | — | — | — | 0.079 | 0.294 | 0.105 | — | — | — |
| Qwen3.5-9B + RAG (pplx-ctx) | 0.180 | 0.525 | 0.492 | — | — | — | — | — | — | — | — | — | 0.093 | 0.294 | 0.085 | — | — | — |
| Qwen3.5-27B offline | 0.549 | 0.844 | 0.700 | — | — | — | — | — | — | — | — | — | 0.220 | 0.394 | 0.191 | — | — | — |
| Qwen3.5-27B + RAG (pplx-ctx) | 0.570 | 0.850 | 0.725 | — | — | — | — | — | — | — | — | — | 0.160 | 0.356 | 0.112 | — | — | — |
| Paper CLIM (recomputed) | — | — | — | 0.162 | 0.325 | 0.163 | 0.162 | 0.325 | 0.163 | 0.162 | 0.325 | 0.163 | 0.162 | 0.325 | 0.163 | 0.162 | 0.325 | 0.163 |

*`veracityV2` majority class: `FALSE`; `climinator` majority class: `INACCURATE`; `climinator_v2` majority class: `INACCURATE`; `climinator_v3` majority class: `INACCURATE`; `climinator_v4` majority class: `INACCURATE`; `climinator_v5` majority class: `INACCURATE`*

## Climinator hierarchy (Leippold 2024 Fig. 3)
Same predictions rolled up to the 5/3/2-class credibility taxonomies.

| Model | L1 (12c) MCC | L1 Acc | L1 F1 | L2 (5c) MCC | L2 Acc | L2 F1 | L3 (3c) MCC | L3 Acc | L3 F1 | L4 (2c) MCC | L4 Acc | L4 F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Majority baseline* | 0.000 | 0.350 | — | 0.000 | 0.725 | — | 0.000 | 0.887 | — | 0.000 | 0.912 | — |
| Paper CLIM (recomputed) | 0.162 | 0.325 | 0.163 | 0.384 | 0.694 | 0.413 | 0.664 | 0.925 | 0.589 | 0.735 | 0.944 | 0.888 |

*Majority class per level: L1=`INACCURATE`; L2=`VERY LOW CREDIBILITY`; L3=`LOW CREDIBILITY`; L4=`NOT CREDIBLE`*
