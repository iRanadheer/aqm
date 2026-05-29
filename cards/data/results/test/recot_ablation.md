# CARDS — RECoT-FT ablation (Qwen3.5 4B + 9B, test set)

| Model | N | Parse fails |
|-------|---|-------------|
| Qwen3.5-4B Base (think) | 1436 | 376 |
| Qwen3.5-4B Base (no-think) | 1436 | 140 |
| CARDS-Qwen3.5-4B No RECoT (no-think) | 1436 | 0 |
| CARDS-Qwen3.5-4B No RECoT (think) | 1436 | 537 |
| CARDS-Qwen3.5-4B | 1436 | 1 |
| Qwen3.5-9B Base (think) | 1436 | 247 |
| Qwen3.5-9B Base (no-think) | 1436 | 24 |
| CARDS-Qwen3.5-9B No RECoT (no-think) | 1436 | 2 |
| CARDS-Qwen3.5-9B No RECoT (think) | 1436 | 833 |
| CARDS-Qwen3.5-9B | 1436 | 0 |
| Qwen3.5-27B Base (think) | 1436 | 86 |
| Qwen3.5-27B Base (no-think) | 1436 | 52 |
| CARDS-Qwen3.5-27B No RECoT (no-think) | 1436 | 0 |
| CARDS-Qwen3.5-27B No RECoT (think) | 1436 | 280 |
| CARDS-Qwen3.5-27B | 1436 | 0 |

## Support ≥ 3

### Samples F1

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B No RECoT (think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B No RECoT (think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B No RECoT (think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.621 | 0.717 | 0.795 | 0.579 | 0.838 | 0.721 | 0.803 | 0.704 | 0.338 | 0.872 | 0.844 | 0.852 | 0.865 | 0.758 | 0.884 |
| 2 | 0.597 | 0.685 | 0.765 | 0.573 | 0.809 | 0.694 | 0.774 | 0.669 | 0.325 | 0.84 | 0.823 | 0.827 | 0.844 | 0.749 | 0.857 |
| 3 | 0.579 | 0.662 | 0.732 | 0.569 | 0.781 | 0.678 | 0.749 | 0.642 | 0.312 | 0.813 | 0.805 | 0.803 | 0.821 | 0.74 | 0.833 |

### Samples Precision

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B No RECoT (think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B No RECoT (think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B No RECoT (think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.624 | 0.721 | 0.8 | 0.579 | 0.848 | 0.726 | 0.81 | 0.711 | 0.339 | 0.881 | 0.843 | 0.854 | 0.876 | 0.758 | 0.89 |
| 2 | 0.592 | 0.679 | 0.77 | 0.572 | 0.822 | 0.694 | 0.776 | 0.679 | 0.325 | 0.85 | 0.823 | 0.828 | 0.859 | 0.748 | 0.865 |
| 3 | 0.572 | 0.653 | 0.733 | 0.567 | 0.792 | 0.675 | 0.748 | 0.652 | 0.312 | 0.823 | 0.806 | 0.802 | 0.833 | 0.738 | 0.843 |

### Samples Recall

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B No RECoT (think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B No RECoT (think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B No RECoT (think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.623 | 0.719 | 0.798 | 0.579 | 0.837 | 0.722 | 0.803 | 0.703 | 0.341 | 0.87 | 0.85 | 0.857 | 0.861 | 0.762 | 0.885 |
| 2 | 0.614 | 0.71 | 0.772 | 0.578 | 0.807 | 0.706 | 0.785 | 0.669 | 0.333 | 0.84 | 0.835 | 0.839 | 0.839 | 0.755 | 0.859 |
| 3 | 0.604 | 0.697 | 0.744 | 0.575 | 0.778 | 0.694 | 0.765 | 0.641 | 0.319 | 0.813 | 0.816 | 0.817 | 0.818 | 0.747 | 0.834 |

### Macro F1

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B No RECoT (think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B No RECoT (think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B No RECoT (think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.473 | 0.56 | 0.594 | 0.279 | 0.632 | 0.629 | 0.611 | 0.532 | 0.524 | 0.663 | 0.71 | 0.741 | 0.723 | 0.575 | 0.766 |
| 2 | 0.327 | 0.412 | 0.405 | 0.211 | 0.449 | 0.47 | 0.437 | 0.364 | 0.423 | 0.502 | 0.553 | 0.581 | 0.534 | 0.429 | 0.599 |
| 3 | 0.243 | 0.254 | 0.272 | 0.141 | 0.371 | 0.365 | 0.335 | 0.264 | 0.328 | 0.379 | 0.467 | 0.495 | 0.454 | 0.358 | 0.487 |

### Macro Precision

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B No RECoT (think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B No RECoT (think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B No RECoT (think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.564 | 0.554 | 0.613 | 0.498 | 0.704 | 0.735 | 0.614 | 0.617 | 0.65 | 0.707 | 0.683 | 0.719 | 0.784 | 0.759 | 0.759 |
| 2 | 0.402 | 0.366 | 0.46 | 0.345 | 0.55 | 0.529 | 0.423 | 0.481 | 0.536 | 0.614 | 0.595 | 0.611 | 0.557 | 0.594 | 0.644 |
| 3 | 0.285 | 0.241 | 0.304 | 0.224 | 0.46 | 0.427 | 0.324 | 0.337 | 0.409 | 0.475 | 0.506 | 0.495 | 0.479 | 0.524 | 0.504 |

### Macro Recall

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B No RECoT (think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B No RECoT (think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B No RECoT (think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.429 | 0.641 | 0.61 | 0.206 | 0.629 | 0.556 | 0.628 | 0.537 | 0.458 | 0.646 | 0.75 | 0.78 | 0.689 | 0.472 | 0.798 |
| 2 | 0.334 | 0.571 | 0.424 | 0.164 | 0.438 | 0.485 | 0.501 | 0.367 | 0.393 | 0.474 | 0.561 | 0.61 | 0.547 | 0.356 | 0.622 |
| 3 | 0.309 | 0.408 | 0.338 | 0.114 | 0.371 | 0.415 | 0.454 | 0.297 | 0.317 | 0.369 | 0.512 | 0.584 | 0.488 | 0.316 | 0.544 |

### Micro F1

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B No RECoT (think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B No RECoT (think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B No RECoT (think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.696 | 0.74 | 0.776 | 0.684 | 0.828 | 0.775 | 0.795 | 0.693 | 0.478 | 0.862 | 0.854 | 0.858 | 0.855 | 0.819 | 0.877 |
| 2 | 0.625 | 0.651 | 0.724 | 0.652 | 0.791 | 0.721 | 0.746 | 0.646 | 0.457 | 0.821 | 0.816 | 0.817 | 0.825 | 0.794 | 0.839 |
| 3 | 0.557 | 0.55 | 0.642 | 0.63 | 0.765 | 0.679 | 0.685 | 0.613 | 0.428 | 0.791 | 0.792 | 0.774 | 0.787 | 0.772 | 0.812 |

### Micro Precision

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B No RECoT (think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B No RECoT (think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B No RECoT (think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.829 | 0.78 | 0.771 | 0.911 | 0.84 | 0.866 | 0.804 | 0.7 | 0.79 | 0.875 | 0.87 | 0.866 | 0.87 | 0.932 | 0.879 |
| 2 | 0.681 | 0.62 | 0.706 | 0.854 | 0.808 | 0.771 | 0.732 | 0.649 | 0.726 | 0.84 | 0.826 | 0.814 | 0.843 | 0.909 | 0.843 |
| 3 | 0.553 | 0.472 | 0.595 | 0.807 | 0.799 | 0.705 | 0.647 | 0.621 | 0.677 | 0.825 | 0.806 | 0.761 | 0.799 | 0.881 | 0.829 |

### Micro Recall

| Level | Qwen3.5-4B Base (think) | Qwen3.5-4B Base (no-think) | CARDS-Qwen3.5-4B No RECoT (no-think) | CARDS-Qwen3.5-4B No RECoT (think) | CARDS-Qwen3.5-4B | Qwen3.5-9B Base (think) | Qwen3.5-9B Base (no-think) | CARDS-Qwen3.5-9B No RECoT (no-think) | CARDS-Qwen3.5-9B No RECoT (think) | CARDS-Qwen3.5-9B | Qwen3.5-27B Base (think) | Qwen3.5-27B Base (no-think) | CARDS-Qwen3.5-27B No RECoT (no-think) | CARDS-Qwen3.5-27B No RECoT (think) | CARDS-Qwen3.5-27B |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.6 | 0.704 | 0.781 | 0.547 | 0.816 | 0.701 | 0.786 | 0.685 | 0.343 | 0.849 | 0.838 | 0.85 | 0.841 | 0.731 | 0.874 |
| 2 | 0.579 | 0.685 | 0.743 | 0.528 | 0.775 | 0.677 | 0.759 | 0.643 | 0.334 | 0.802 | 0.807 | 0.82 | 0.808 | 0.705 | 0.836 |
| 3 | 0.561 | 0.66 | 0.698 | 0.516 | 0.734 | 0.654 | 0.728 | 0.605 | 0.313 | 0.76 | 0.778 | 0.788 | 0.775 | 0.687 | 0.796 |

