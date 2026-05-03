# CARDS — Scaling ablation (Base vs RECoT-FT, test set)

| Model | N | Parse fails |
|-------|---|-------------|
| Qwen3.5-2B Base | 1436 | 1413 |
| CARDS-Qwen3.5-2B | 1436 | 1 |
| Qwen3.5-4B Base | 1436 | 376 |
| CARDS-Qwen3.5-4B | 1436 | 1 |
| Qwen3.5-9B Base | 1436 | 247 |
| CARDS-Qwen3.5-9B | 1436 | 0 |
| Qwen3.5-27B Base | 1436 | 86 |
| CARDS-Qwen3.5-27B | 1436 | 0 |

## Support ≥ 3

### Samples F1

| Level | Qwen3.5-2B Base | CARDS-Qwen3.5-2B | Qwen3.5-4B Base | CARDS-Qwen3.5-4B | Qwen3.5-9B Base | CARDS-Qwen3.5-9B | Qwen3.5-27B Base | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|
| 1 | 0.003 | 0.758 | 0.621 | 0.838 | 0.721 | 0.872 | 0.844 | 0.884 |
| 2 | 0.003 | 0.726 | 0.597 | 0.809 | 0.694 | 0.84 | 0.823 | 0.857 |
| 3 | 0.003 | 0.699 | 0.579 | 0.781 | 0.678 | 0.813 | 0.805 | 0.833 |

### Macro F1

| Level | Qwen3.5-2B Base | CARDS-Qwen3.5-2B | Qwen3.5-4B Base | CARDS-Qwen3.5-4B | Qwen3.5-9B Base | CARDS-Qwen3.5-9B | Qwen3.5-27B Base | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 0.528 | 0.473 | 0.632 | 0.629 | 0.663 | 0.71 | 0.766 |
| 2 | 0.0 | 0.32 | 0.327 | 0.449 | 0.47 | 0.502 | 0.553 | 0.599 |
| 3 | 0.0 | 0.259 | 0.243 | 0.371 | 0.365 | 0.379 | 0.467 | 0.487 |

### Micro F1

| Level | Qwen3.5-2B Base | CARDS-Qwen3.5-2B | Qwen3.5-4B Base | CARDS-Qwen3.5-4B | Qwen3.5-9B Base | CARDS-Qwen3.5-9B | Qwen3.5-27B Base | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|
| 1 | 0.008 | 0.746 | 0.696 | 0.828 | 0.775 | 0.862 | 0.854 | 0.877 |
| 2 | 0.007 | 0.703 | 0.625 | 0.791 | 0.721 | 0.821 | 0.816 | 0.839 |
| 3 | 0.006 | 0.683 | 0.557 | 0.765 | 0.679 | 0.791 | 0.792 | 0.812 |

