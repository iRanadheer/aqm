# aqm

PhD thesis monorepo. One folder per chapter; each chapter is self-contained
(its own data, models, scripts, docs).

| Chapter | Topic |
|---|---|
| [cards/](cards/) | Hierarchical climate-discourse claim classification |
| `wind/` | Opposition-to-wind-energy detection (TBD: migrate from sibling repo) |
| `debunk/` | TBD |
| [dynamic-fewshot-learning/](dynamic-fewshot-learning/) | Dynamic (retrieval) few-shot vs fine-tuning on cards + wind (no training) |
| [dynamic-fewshot-learning/](dynamic-fewshot-learning/) | Dynamic few-shot learning for computational social science (paper) |

## Layout

Each chapter folder owns everything it needs to reproduce its results — data
splits, training scripts, inference, eval, docs. Chapters do not import from
each other; duplication across chapters is intentional.

See each chapter's `README.md` for its pipeline.
