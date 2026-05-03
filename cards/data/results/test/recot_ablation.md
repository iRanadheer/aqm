# CARDS — RECoT-FT ablation (Qwen3.5 4B + 9B, test set)

| Model | N | Parse fails |
|-------|---|-------------|
| Qwen3.5-4B Base (think) | 1436 | 376 |
| Qwen3.5-4B Base (no-think) | 1436 | 140 |
| CARDS-Qwen3.5-4B No RECoT (no-think) | 1436 | 0 |
| CARDS-Qwen3.5-4B | 1436 | 1 |
| Qwen3.5-9B Base (think) | 1436 | 247 |
| Qwen3.5-9B Base (no-think) | 1436 | 24 |
| CARDS-Qwen3.5-9B No RECoT (no-think) | 1436 | 2 |
| CARDS-Qwen3.5-9B | 1436 | 0 |
| Qwen3.5-27B Base (think) | 1436 | 86 |
| Qwen3.5-27B Base (no-think) | 1436 | 52 |
| CARDS-Qwen3.5-27B No RECoT (no-think) | 1436 | 0 |
| CARDS-Qwen3.5-27B | 1436 | 0 |

## Support ≥ 3

### Samples F1

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.621 | 0.717 | 0.795 | 0.838 | 0.721 | 0.803 | 0.704 | 0.872 | 0.844 | 0.852 | 0.865 | 0.884 |
| 2 | 0.597 | 0.685 | 0.765 | 0.809 | 0.694 | 0.774 | 0.669 | 0.84 | 0.823 | 0.827 | 0.844 | 0.857 |
| 3 | 0.579 | 0.662 | 0.732 | 0.781 | 0.678 | 0.749 | 0.642 | 0.813 | 0.805 | 0.803 | 0.821 | 0.833 |

### Macro F1

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.473 | 0.56 | 0.594 | 0.632 | 0.629 | 0.611 | 0.532 | 0.663 | 0.71 | 0.741 | 0.723 | 0.766 |
| 2 | 0.327 | 0.412 | 0.405 | 0.449 | 0.47 | 0.437 | 0.364 | 0.502 | 0.553 | 0.581 | 0.534 | 0.599 |
| 3 | 0.243 | 0.254 | 0.272 | 0.371 | 0.365 | 0.335 | 0.264 | 0.379 | 0.467 | 0.495 | 0.454 | 0.487 |

### Micro F1

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.696 | 0.74 | 0.776 | 0.828 | 0.775 | 0.795 | 0.693 | 0.862 | 0.854 | 0.858 | 0.855 | 0.877 |
| 2 | 0.625 | 0.651 | 0.724 | 0.791 | 0.721 | 0.746 | 0.646 | 0.821 | 0.816 | 0.817 | 0.825 | 0.839 |
| 3 | 0.557 | 0.55 | 0.642 | 0.765 | 0.679 | 0.685 | 0.613 | 0.791 | 0.792 | 0.774 | 0.787 | 0.812 |

