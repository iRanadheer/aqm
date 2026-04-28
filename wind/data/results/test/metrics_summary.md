# Wind — test set

| Model | Rows | API errors | Parse failures | Opposition-only n |
|-------|---|---|---|---|
| Windy-Qwen3.5-4B | 773 | 0 | 0 | 436 |
| Windy-Qwen3.5-9B | 773 | 0 | 0 | 436 |
| Windy-Qwen3.5-27B | 773 | 0 | 0 | 436 |
| Windy-Qwen3.5-27B FP8 | 773 | 0 | 0 | 436 |
| CARDS-Wind-Qwen3.6-27B | 773 | 0 | 0 | 436 |
| CARDS-Wind-Qwen3.6-27B FP8 | 773 | 0 | 0 | 436 |
| Claude Opus 4.7 | 773 | 0 | 1 | 436 |

## Detection (binary)

| Metric | Windy-Qwen3.5-4B | Windy-Qwen3.5-9B | Windy-Qwen3.5-27B | Windy-Qwen3.5-27B FP8 | CARDS-Wind-Qwen3.6-27B | CARDS-Wind-Qwen3.6-27B FP8 | Claude Opus 4.7 |
|---|---|---|---|---|---|---|---|
| F1 | 0.851 | 0.853 | 0.894 | 0.898 | 0.886 | 0.891 | 0.894 |

## Samples F1

| View | Windy-Qwen3.5-4B | Windy-Qwen3.5-9B | Windy-Qwen3.5-27B | Windy-Qwen3.5-27B FP8 | CARDS-Wind-Qwen3.6-27B | CARDS-Wind-Qwen3.6-27B FP8 | Claude Opus 4.7 |
|---|---|---|---|---|---|---|---|
| Frames — all rows | 0.697 | 0.696 | 0.781 | 0.787 | 0.766 | 0.772 | 0.793 |
| Frames — opposition only | 0.699 | 0.695 | 0.747 | 0.751 | 0.729 | 0.739 | 0.734 |
| Claims — all rows | 0.654 | 0.676 | 0.741 | 0.755 | 0.735 | 0.738 | 0.755 |
| Claims — opposition only | 0.623 | 0.66 | 0.675 | 0.694 | 0.675 | 0.677 | 0.667 |

## Macro F1

| View | Windy-Qwen3.5-4B | Windy-Qwen3.5-9B | Windy-Qwen3.5-27B | Windy-Qwen3.5-27B FP8 | CARDS-Wind-Qwen3.6-27B | CARDS-Wind-Qwen3.6-27B FP8 | Claude Opus 4.7 |
|---|---|---|---|---|---|---|---|
| Frames — all rows | 0.573 | 0.552 | 0.622 | 0.63 | 0.615 | 0.616 | 0.664 |
| Frames — opposition only | 0.631 | 0.606 | 0.659 | 0.664 | 0.646 | 0.657 | 0.692 |
| Claims — all rows | 0.49 | 0.517 | 0.578 | 0.597 | 0.555 | 0.554 | 0.559 |
| Claims — opposition only | 0.531 | 0.559 | 0.61 | 0.628 | 0.584 | 0.57 | 0.583 |

## Micro F1

| View | Windy-Qwen3.5-4B | Windy-Qwen3.5-9B | Windy-Qwen3.5-27B | Windy-Qwen3.5-27B FP8 | CARDS-Wind-Qwen3.6-27B | CARDS-Wind-Qwen3.6-27B FP8 | Claude Opus 4.7 |
|---|---|---|---|---|---|---|---|
| Frames — all rows | 0.679 | 0.674 | 0.752 | 0.756 | 0.737 | 0.74 | 0.757 |
| Frames — opposition only | 0.744 | 0.74 | 0.792 | 0.796 | 0.779 | 0.783 | 0.788 |
| Claims — all rows | 0.606 | 0.643 | 0.686 | 0.704 | 0.687 | 0.686 | 0.698 |
| Claims — opposition only | 0.652 | 0.69 | 0.716 | 0.734 | 0.72 | 0.717 | 0.72 |
