# Reports & plots

Three scripts turn the `data/results/test/*.jsonl` inference outputs into
scored tables, figures, and significance reports:

| Script | Output | What it answers |
|---|---|---|
| `generate_report.py` | `data/results/test/metrics_summary.{json,md}` | breadth — every model, both benchmarks, point metrics |
| `plot_hierarchy.py` | `data/results/test/debunk_benchmark.png` (+ online/offline) | the headline figure |
| `evals/significance.py` | `data/significance/summary_<variant>.md` | depth — 6 core models, bootstrap CIs, pairwise deltas |

All three score with the same parser (`prompts.extract_raw_label`) and the
same gold columns, so the numbers are consistent across them.

---

## `generate_report.py`

Scores each `(model, variant)` result file and writes one combined summary
— models as rows, metrics grouped by prompt variant.

- **`MODELS`** — the headline lineup: `(display label, slug)` pairs. The
  slug is the inference filename prefix (`data/results/test/<slug>-<variant>.jsonl`).
- **`REPORT_VARIANTS`** — the only variants scored, mapped to clean display
  names so the version suffix is dropped in the rendered tables:

  | internal id | shown as |
  |---|---|
  | `veracityV2` | `veracity` |
  | `climinator_v4` | `climinator` |

  (The JSON keeps the internal ids for traceability; only the `.md` and the
  plots use the clean names.) Earlier variants (`veracityV1`,
  `climinator` v1/v2/v3/v5) are defined in `prompts.py` but intentionally
  **not** reported — change `REPORT_VARIANTS` to widen the scope.

### Metrics

Per `(model, variant)`: `accuracy`, `macro_f1_present` (macro-F1 over
classes with non-zero gold support), `mcc`, per-class precision/recall/F1,
a majority-class `baseline_acc`, raw prediction counts, a confusion map, and
counts of `api_errors` / `parse_failures` / `oov`. API errors, unparseable
responses, and out-of-vocabulary labels all collapse to a `__PARSE_FAIL__`
prediction so they're penalised, not silently dropped.

### Climinator hierarchy

For the `climinator*` variants, the same predictions are also scored at the
4 credibility levels (`prompts.climinator_rollup`): L1=12 → L2=5 → L3=3 →
L4=2 classes. The `.md` gets a second table showing how the metrics improve
as the taxonomy is coarsened.

### Paper CLIM baseline

The special slug **`__paper__`** isn't a result file — `main()` synthesises
rows from `test.jsonl`'s `true_climinator` column (the paper's own CLIM
predictions) and scores them against `true_cfb_label` with the **same code,
same rows, same denominator** as every model. Rows where the paper emitted
`NEI` are passed through as synthetic errors (it abstains there). This makes
"Paper CLIM (recomputed)" a strictly fair baseline rather than a number
trusted from the paper. Only meaningful for the `climinator*` variants.

```bash
python generate_report.py
```

## `plot_hierarchy.py`

`plot_combined()` → `debunk_benchmark.png`: a two-row figure —

- **top** — climinator across the 4 hierarchy levels (Accuracy + Macro-F1),
  paired offline/RAG bars per model family, with the Paper CLIM value as a
  black reference tick per level;
- **bottom** — veracity (flat 4-class), with the RAG-minus-offline delta
  annotated under each family.

`plot_online_vs_offline()` → `climinator_online_vs_offline.png` is only
emitted when both `*-online` and offline halves of a model pair exist;
otherwise it's skipped with a note. The shared `FAMILIES` /
`FAMILY_COLOURS` constants keep the two panels reading as a matched pair.

```bash
python plot_hierarchy.py
```

## `evals/significance.py`

The defensible-comparison layer (following Ulmer et al., LREC 2022). For
each benchmark (`veracityV2`, `climinator_v4`) it reports per-model accuracy
with a 95% BCa bootstrap CI (`scipy.stats.bootstrap`, 9999 resamples,
seed 42), plus MCC and macro-F1 as point estimates, and pairwise A−B deltas
with CIs — verdict `improves` / `lower` / `comparable` by whether the delta
CI clears zero. Lineup: the 6 core models × {offline, +RAG} vs the Paper
CLIM baseline.

```bash
python evals/significance.py                 # both benchmarks → data/significance/
python evals/significance.py pair climinator_v4 \
    qwen-qwen3-5-27b-rag-pplx-ctx __paper__  # one comparison
```
