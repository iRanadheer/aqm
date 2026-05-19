# Debunk — test set

| Model | veracityV1 MCC | veracityV1 Acc | veracityV1 F1 | climinator MCC | climinator Acc | climinator F1 |
|---|---|---|---|---|---|---|
| *Majority baseline* | 0.000 | 0.775 | — | 0.000 | 0.350 | — |
| Sonar | 0.573 | 0.812 | 0.701 | 0.190 | 0.350 | 0.157 |
| Sonar Pro | 0.430 | 0.806 | 0.607 | 0.140 | 0.344 | 0.096 |
| Claude Opus 4.7 online | 0.532 | 0.831 | 0.709 | 0.254 | 0.400 | 0.178 |
| Claude Opus 4.7 offline | 0.580 | 0.850 | 0.732 | 0.236 | 0.388 | 0.195 |
| GPT-5.5 online | 0.610 | 0.856 | 0.776 | 0.263 | 0.425 | 0.213 |
| GPT-5.5 offline | 0.558 | 0.831 | 0.742 | 0.243 | 0.406 | 0.200 |
| Exa Answer | 0.228 | 0.569 | 0.433 | 0.106 | 0.244 | 0.090 |

*`veracityV1` majority class: `FALSE`; `climinator` majority class: `INACCURATE`*

## Climinator hierarchy (Leippold 2024 Fig. 3)
Same predictions rolled up to the 5/3/2-class credibility taxonomies.

| Model | L1 (12c) MCC | L1 Acc | L1 F1 | L2 (5c) MCC | L2 Acc | L2 F1 | L3 (3c) MCC | L3 Acc | L3 F1 | L4 (2c) MCC | L4 Acc | L4 F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Majority baseline* | 0.000 | 0.350 | — | 0.000 | 0.725 | — | 0.000 | 0.887 | — | 0.000 | 0.912 | — |
| Sonar | 0.190 | 0.350 | 0.157 | 0.360 | 0.662 | 0.418 | 0.665 | 0.938 | 0.571 | 0.717 | 0.956 | 0.858 |
| Sonar Pro | 0.140 | 0.344 | 0.096 | 0.270 | 0.706 | 0.337 | 0.691 | 0.944 | 0.581 | 0.752 | 0.963 | 0.874 |
| Claude Opus 4.7 online | 0.254 | 0.400 | 0.178 | 0.535 | 0.781 | 0.464 | 0.808 | 0.963 | 0.638 | 0.922 | 0.988 | 0.961 |
| Claude Opus 4.7 offline | 0.236 | 0.388 | 0.195 | 0.430 | 0.738 | 0.398 | 0.839 | 0.969 | 0.639 | 0.922 | 0.988 | 0.961 |
| GPT-5.5 online | 0.263 | 0.425 | 0.213 | 0.466 | 0.756 | 0.469 | 0.839 | 0.969 | 0.639 | 0.922 | 0.988 | 0.961 |
| GPT-5.5 offline | 0.243 | 0.406 | 0.200 | 0.430 | 0.738 | 0.458 | 0.839 | 0.969 | 0.639 | 0.922 | 0.988 | 0.961 |
| Exa Answer | 0.106 | 0.244 | 0.090 | 0.155 | 0.444 | 0.213 | 0.273 | 0.575 | 0.330 | 0.269 | 0.581 | 0.492 |

*Majority class per level: L1=`INACCURATE`; L2=`VERY LOW CREDIBILITY`; L3=`LOW CREDIBILITY`; L4=`NOT CREDIBLE`*
