# CARDS — RECoT-FT inner ablation (Qwen3.5-4B, test set)

| Model | N | Parse fails |
|-------|---|-------------|
| Qwen3.5-4B Base | 1436 | 376 |
| CARDS-Qwen3.5-4B No RECoT | 1436 | 12 |
| CARDS-Qwen3.5-4B | 1436 | 1 |

## All labels

### Samples F1

| Level | Qwen3.5-4B Base | CARDS-Qwen3.5-4B No RECoT | CARDS-Qwen3.5-4B |
|-------|---|---|---|
| 1 | 0.621 | 0.732 | 0.838 |
| 2 | 0.592 | 0.689 | 0.808 |
| 3 | 0.577 | 0.657 | 0.78 |

### Macro F1

| Level | Qwen3.5-4B Base | CARDS-Qwen3.5-4B No RECoT | CARDS-Qwen3.5-4B |
|-------|---|---|---|
| 1 | 0.473 | 0.561 | 0.632 |
| 2 | 0.251 | 0.334 | 0.33 |
| 3 | 0.192 | 0.23 | 0.276 |

### Micro F1

| Level | Qwen3.5-4B Base | CARDS-Qwen3.5-4B No RECoT | CARDS-Qwen3.5-4B |
|-------|---|---|---|
| 1 | 0.696 | 0.723 | 0.828 |
| 2 | 0.597 | 0.635 | 0.784 |
| 3 | 0.529 | 0.546 | 0.738 |


## Support ≥ 3

### Samples F1

| Level | Qwen3.5-4B Base | CARDS-Qwen3.5-4B No RECoT | CARDS-Qwen3.5-4B |
|-------|---|---|---|
| 1 | 0.621 | 0.732 | 0.838 |
| 2 | 0.597 | 0.691 | 0.809 |
| 3 | 0.579 | 0.66 | 0.781 |

### Macro F1

| Level | Qwen3.5-4B Base | CARDS-Qwen3.5-4B No RECoT | CARDS-Qwen3.5-4B |
|-------|---|---|---|
| 1 | 0.473 | 0.561 | 0.632 |
| 2 | 0.327 | 0.395 | 0.449 |
| 3 | 0.243 | 0.292 | 0.371 |

### Micro F1

| Level | Qwen3.5-4B Base | CARDS-Qwen3.5-4B No RECoT | CARDS-Qwen3.5-4B |
|-------|---|---|---|
| 1 | 0.696 | 0.723 | 0.828 |
| 2 | 0.625 | 0.658 | 0.791 |
| 3 | 0.557 | 0.601 | 0.765 |

