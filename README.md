# aqm

PhD thesis monorepo. One folder per chapter; each chapter is self-contained
(its own data, models, scripts, docs).

| Chapter | Topic |
|---|---|
| [cards/](cards/) | Hierarchical climate-discourse claim classification |
| [wind/](wind/) | Wind-energy opposition detection (detection / frames / claims) |
| [debunk/](debunk/) | Climate-claim fact-checking benchmark (Leippold et al. veracity dataset) |
| [icl/](icl/) | In-context (static/dynamic few-shot) vs fine-tuning on cards + wind (no training) |

## Layout

Each chapter folder owns everything it needs to reproduce its results — data
splits, training scripts, inference, eval, docs. Chapters do not import from
each other; duplication across chapters is intentional.

See each chapter's `README.md` for its pipeline.
