# Synthetic augmentation ablation — val set

Before = pre-augmentation student (`*-preaug`); After = retrained on `train_augmented`. Delta = After - Before. Macro F1 restricted to codes with gold support >= 3 in val (same kept-code set before and after); Samples F1 and Detection F1 are support-independent.


## Windy-Qwen3.5-4B

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Detection F1 | 0.849 | 0.864 | **+0.015** |
| Frames (opp) Samples F1 | 0.614 | 0.727 | **+0.113** |
| Frames (opp) Macro F1 (sup>=3) | 0.513 | 0.655 | **+0.142** |
| Claims (opp) Samples F1 | 0.51 | 0.623 | **+0.113** |
| Claims (opp) Macro F1 (sup>=3) | 0.491 | 0.656 | **+0.165** |
| Claims (all) Samples F1 | 0.625 | 0.672 | **+0.047** |
| Claims (all) Macro F1 (sup>=3) | 0.467 | 0.613 | **+0.146** |

## Windy-Qwen3.5-9B

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Detection F1 | 0.838 | 0.866 | **+0.028** |
| Frames (opp) Samples F1 | 0.648 | 0.713 | **+0.065** |
| Frames (opp) Macro F1 (sup>=3) | 0.589 | 0.635 | **+0.046** |
| Claims (opp) Samples F1 | 0.563 | 0.651 | **+0.088** |
| Claims (opp) Macro F1 (sup>=3) | 0.611 | 0.693 | **+0.082** |
| Claims (all) Samples F1 | 0.66 | 0.693 | **+0.033** |
| Claims (all) Macro F1 (sup>=3) | 0.58 | 0.652 | **+0.072** |

## Reference points (no pre-aug counterpart)

| Metric | Windy-Qwen3.5-27B | Claude Opus 4.7 (teacher) |
|---|---|---|
| Detection F1 | 0.912 | 0.937 |
| Frames (opp) Samples F1 | 0.747 | 0.72 |
| Frames (opp) Macro F1 (sup>=3) | 0.655 | 0.648 |
| Claims (opp) Samples F1 | 0.733 | 0.713 |
| Claims (opp) Macro F1 (sup>=3) | 0.752 | 0.736 |
| Claims (all) Samples F1 | 0.786 | 0.809 |
| Claims (all) Macro F1 (sup>=3) | 0.732 | 0.728 |
