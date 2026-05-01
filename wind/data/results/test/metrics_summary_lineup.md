# Wind — test set

| Model | Rows | API errors | Parse failures | Opposition-only n |
|-------|---|---|---|---|
| Windy-Qwen3.5-4B | 773 | 0 | 0 | 436 |
| Windy-Qwen3.5-9B | 773 | 0 | 0 | 436 |
| Windy-Qwen3.5-27B | 773 | 0 | 0 | 436 |
| Windy-Qwen3.5-27B FP8 | 773 | 0 | 0 | 436 |
| Claude Opus 4.7 | 773 | 0 | 1 | 436 |
| GPT-5.5 | 773 | 0 | 2 | 436 |

## Detection (binary)

| Metric | Windy-Qwen3.5-4B | Windy-Qwen3.5-9B | Windy-Qwen3.5-27B | Windy-Qwen3.5-27B FP8 | Claude Opus 4.7 | GPT-5.5 |
|---|---|---|---|---|---|---|
| Precision | 0.795 | 0.797 | 0.871 | 0.877 | 0.896 | 0.927 |
| Recall | 0.915 | 0.917 | 0.917 | 0.92 | 0.89 | 0.846 |
| F1 | 0.851 | 0.853 | 0.894 | 0.898 | 0.893 | 0.885 |

## Samples F1

| View | Windy-Qwen3.5-4B | Windy-Qwen3.5-9B | Windy-Qwen3.5-27B | Windy-Qwen3.5-27B FP8 | Claude Opus 4.7 | GPT-5.5 |
|---|---|---|---|---|---|---|
| Frames — all rows | 0.697 | 0.696 | 0.781 | 0.787 | 0.791 | 0.792 |
| Frames — opposition only | 0.699 | 0.695 | 0.747 | 0.751 | 0.734 | 0.697 |
| Claims — all rows | 0.654 | 0.676 | 0.741 | 0.755 | 0.754 | 0.745 |
| Claims — opposition only | 0.623 | 0.66 | 0.675 | 0.694 | 0.667 | 0.614 |

## Macro F1

| View | Windy-Qwen3.5-4B | Windy-Qwen3.5-9B | Windy-Qwen3.5-27B | Windy-Qwen3.5-27B FP8 | Claude Opus 4.7 | GPT-5.5 |
|---|---|---|---|---|---|---|
| Frames — all rows | 0.573 | 0.552 | 0.622 | 0.63 | 0.664 | 0.646 |
| Frames — opposition only | 0.631 | 0.606 | 0.659 | 0.664 | 0.692 | 0.66 |
| Claims — all rows | 0.49 | 0.517 | 0.578 | 0.597 | 0.559 | 0.55 |
| Claims — opposition only | 0.531 | 0.559 | 0.61 | 0.628 | 0.583 | 0.567 |

## Micro F1

| View | Windy-Qwen3.5-4B | Windy-Qwen3.5-9B | Windy-Qwen3.5-27B | Windy-Qwen3.5-27B FP8 | Claude Opus 4.7 | GPT-5.5 |
|---|---|---|---|---|---|---|
| Frames — all rows | 0.679 | 0.674 | 0.752 | 0.756 | 0.757 | 0.749 |
| Frames — opposition only | 0.744 | 0.74 | 0.792 | 0.796 | 0.788 | 0.768 |
| Claims — all rows | 0.606 | 0.643 | 0.686 | 0.704 | 0.698 | 0.678 |
| Claims — opposition only | 0.652 | 0.69 | 0.716 | 0.734 | 0.72 | 0.692 |
