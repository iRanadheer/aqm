# Model comparison (auto-generated)

Auto-generated from `data/results/val/*.jsonl` and `data/results/test/*.jsonl`. All metrics computed over all codes (no min-support filter). Best value in each row is bolded. "Opus" = slim-prompt teacher; "Opus Turbo" = full-prompt teacher. Student models are all slim-trained; "aug" = retrained on augmented train.jsonl; "pre-aug" = original 1999-row train.jsonl.

## Val (development set)

330 rows total, 182 opposition-positive.

| Metric | Opus | Opus Turbo | 4B pre-aug | 4B aug | 9B pre-aug | 9B aug | 27B aug |
|---|---:|---:|---:|---:|---:|---:|---:|
| Detection F1 | 0.939 | **0.947** | 0.849 | 0.864 | 0.838 | 0.866 | 0.912 |
| Frames samples_F1 | 0.815 | **0.836** | 0.682 | 0.729 | 0.707 | 0.727 | 0.794 |
| Frames macro_F1 | 0.613 | **0.680** | 0.368 | 0.569 | 0.476 | 0.566 | 0.606 |
| Frames micro_F1 | 0.738 | **0.763** | 0.599 | 0.687 | 0.648 | 0.681 | 0.730 |
| Frames exact_match | 0.679 | **0.700** | 0.544 | 0.604 | 0.550 | 0.574 | 0.656 |
| Claims samples_F1 | 0.811 | **0.826** | 0.625 | 0.672 | 0.660 | 0.693 | 0.786 |
| Claims macro_F1 | 0.678 | **0.687** | 0.365 | 0.488 | 0.472 | 0.601 | 0.639 |
| Claims micro_F1 | 0.739 | **0.759** | 0.501 | 0.606 | 0.572 | 0.648 | 0.733 |
| Claims exact_match | **0.621** | **0.621** | 0.432 | 0.462 | 0.468 | 0.477 | 0.589 |
| Frames (opp-only) samples_F1 | 0.720 | **0.747** | 0.614 | 0.727 | 0.648 | 0.713 | **0.747** |
| Frames (opp-only) macro_F1 | 0.632 | **0.693** | 0.418 | 0.656 | 0.523 | 0.620 | 0.637 |
| Claims (opp-only) samples_F1 | 0.713 | 0.728 | 0.510 | 0.623 | 0.563 | 0.651 | **0.733** |
| Claims (opp-only) macro_F1 | 0.687 | **0.692** | 0.395 | 0.540 | 0.493 | 0.639 | 0.657 |

## Test (held-out)

772 rows total, 436 opposition-positive.

| Metric | Opus | 4B aug | 9B aug | 27B aug |
|---|---:|---:|---:|---:|
| Detection F1 | **0.894** | 0.851 | 0.853 | **0.894** |
| Frames samples_F1 | **0.793** | 0.697 | 0.696 | 0.781 |
| Frames macro_F1 | **0.664** | 0.573 | 0.552 | 0.622 |
| Frames micro_F1 | **0.757** | 0.679 | 0.674 | 0.752 |
| Frames exact_match | **0.685** | 0.554 | 0.561 | 0.668 |
| Claims samples_F1 | **0.755** | 0.654 | 0.676 | 0.741 |
| Claims macro_F1 | 0.559 | 0.490 | 0.517 | **0.578** |
| Claims micro_F1 | **0.698** | 0.606 | 0.643 | 0.686 |
| Claims exact_match | **0.578** | 0.462 | 0.489 | 0.577 |
| Frames (opp-only) samples_F1 | 0.734 | 0.699 | 0.695 | **0.747** |
| Frames (opp-only) macro_F1 | **0.692** | 0.631 | 0.606 | 0.659 |
| Claims (opp-only) samples_F1 | 0.667 | 0.623 | 0.660 | **0.675** |
| Claims (opp-only) macro_F1 | 0.583 | 0.531 | 0.559 | **0.610** |
