# Debunk — test set

| Model | veracity MCC | veracity Acc | veracity F1 | climinator MCC | climinator Acc | climinator F1 |
|---|---|---|---|---|---|---|
| *Majority baseline* | 0.000 | 0.775 | — | 0.000 | 0.350 | — |
| Claude Opus 4.7 offline | 0.625 | 0.869 | 0.762 | 0.218 | 0.375 | 0.175 |
| Claude Opus 4.7 + RAG (pplx-ctx) | 0.582 | 0.856 | 0.723 | 0.268 | 0.406 | 0.197 |
| GPT-4o-mini offline | 0.397 | 0.756 | 0.623 | 0.099 | 0.294 | 0.102 |
| GPT-4o-mini + RAG (pplx-ctx) | 0.509 | 0.819 | 0.700 | 0.083 | 0.294 | 0.160 |
| DeepSeek V4 Flash offline | 0.438 | 0.750 | 0.662 | 0.163 | 0.338 | 0.174 |
| DeepSeek V4 Flash + RAG (pplx-ctx) | 0.421 | 0.756 | 0.672 | 0.204 | 0.369 | 0.161 |
| GPT-5.5 offline | 0.578 | 0.844 | 0.746 | 0.226 | 0.381 | 0.233 |
| GPT-5.5 + RAG (pplx-ctx) | 0.521 | 0.825 | 0.707 | 0.289 | 0.431 | 0.282 |
| Qwen3.5-9B offline | 0.465 | 0.812 | 0.626 | 0.079 | 0.294 | 0.105 |
| Qwen3.5-9B + RAG (pplx-ctx) | 0.180 | 0.525 | 0.492 | 0.093 | 0.294 | 0.085 |
| Qwen3.5-27B offline | 0.549 | 0.844 | 0.700 | 0.220 | 0.394 | 0.191 |
| Qwen3.5-27B + RAG (pplx-ctx) | 0.570 | 0.850 | 0.725 | 0.160 | 0.356 | 0.112 |
| Paper CLIM (recomputed) | — | — | — | 0.162 | 0.325 | 0.163 |

*`veracity` majority class: `FALSE`; `climinator` majority class: `INACCURATE`*

## Climinator hierarchy (Leippold 2024 Fig. 3)
Same predictions rolled up to the 5/3/2-class credibility taxonomies.

| Model | L1 (12c) MCC | L1 Acc | L1 F1 | L2 (5c) MCC | L2 Acc | L2 F1 | L3 (3c) MCC | L3 Acc | L3 F1 | L4 (2c) MCC | L4 Acc | L4 F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Majority baseline* | 0.000 | 0.350 | — | 0.000 | 0.725 | — | 0.000 | 0.887 | — | 0.000 | 0.912 | — |
| Claude Opus 4.7 offline | 0.218 | 0.375 | 0.175 | 0.428 | 0.756 | 0.430 | 0.812 | 0.963 | 0.639 | 0.929 | 0.988 | 0.963 |
| Claude Opus 4.7 + RAG (pplx-ctx) | 0.268 | 0.406 | 0.197 | 0.491 | 0.781 | 0.441 | 0.844 | 0.969 | 0.641 | 0.898 | 0.981 | 0.961 |
| GPT-4o-mini offline | 0.099 | 0.294 | 0.102 | 0.240 | 0.613 | 0.325 | 0.469 | 0.838 | 0.572 | 0.683 | 0.944 | 0.839 |
| GPT-4o-mini + RAG (pplx-ctx) | 0.083 | 0.294 | 0.160 | 0.238 | 0.625 | 0.406 | 0.470 | 0.825 | 0.578 | 0.611 | 0.906 | 0.887 |
| DeepSeek V4 Flash offline | 0.163 | 0.338 | 0.174 | 0.367 | 0.713 | 0.492 | 0.672 | 0.925 | 0.671 | 0.792 | 0.963 | 0.905 |
| DeepSeek V4 Flash + RAG (pplx-ctx) | 0.204 | 0.369 | 0.161 | 0.353 | 0.700 | 0.419 | 0.674 | 0.931 | 0.597 | 0.748 | 0.956 | 0.897 |
| GPT-5.5 offline | 0.226 | 0.381 | 0.233 | 0.413 | 0.706 | 0.468 | 0.672 | 0.931 | 0.599 | 0.782 | 0.963 | 0.903 |
| GPT-5.5 + RAG (pplx-ctx) | 0.289 | 0.431 | 0.282 | 0.469 | 0.744 | 0.506 | 0.750 | 0.950 | 0.624 | 0.844 | 0.975 | 0.938 |
| Qwen3.5-9B offline | 0.079 | 0.294 | 0.105 | 0.271 | 0.675 | 0.330 | 0.509 | 0.850 | 0.557 | 0.518 | 0.856 | 0.833 |
| Qwen3.5-9B + RAG (pplx-ctx) | 0.093 | 0.294 | 0.085 | 0.227 | 0.637 | 0.382 | 0.432 | 0.812 | 0.567 | 0.485 | 0.831 | 0.856 |
| Qwen3.5-27B offline | 0.220 | 0.394 | 0.191 | 0.513 | 0.794 | 0.469 | 0.739 | 0.950 | 0.610 | 0.840 | 0.975 | 0.918 |
| Qwen3.5-27B + RAG (pplx-ctx) | 0.160 | 0.356 | 0.112 | 0.401 | 0.750 | 0.473 | 0.838 | 0.969 | 0.638 | 0.922 | 0.988 | 0.961 |
| Paper CLIM (recomputed) | 0.162 | 0.325 | 0.163 | 0.384 | 0.694 | 0.413 | 0.664 | 0.925 | 0.589 | 0.735 | 0.944 | 0.888 |

*Majority class per level: L1=`INACCURATE`; L2=`VERY LOW CREDIBILITY`; L3=`LOW CREDIBILITY`; L4=`NOT CREDIBLE`*
