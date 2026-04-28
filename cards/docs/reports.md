# Reports (`generate_report.py`)

One script, multiple reports. Re-running overwrites.

```
uv run generate_report.py
```

## Outputs

Per split (test, twitter), one `metrics_summary.{json,md}` with the
headline lineup. On the test split, two additional ablation reports:
- inner ablation (Base vs No-RECoT FT vs RECoT FT at one model size)
- scaling ablation (Base vs RECoT FT across all sizes)

## Strict parsing

Predictions are extracted from the `response` field at report time
(re-derived every run, never stored). A response counts as parsed only if
it contains a `</think>` tag followed by a `categories:` YAML block. No
fallback regex on free-form text. Pre-parsed list-shape responses (legacy
output format from earlier runs) are accepted as-is.

Anything else returns no predictions and is counted as a parse failure.
This makes "model can't reliably follow the structured output" itself a
visible signal in the report.

## Layout

Each report has a header table (model, N rows, parse failures), then
per-support sections (all labels, support ≥ N), with one sub-table per
F1 metric (samples / macro / micro). Models go in columns, hierarchy
levels in rows. Precision, recall, and exact-match are computed and
saved to JSON but omitted from the markdown.

## Configuration

The model lineup and ablation entries are flat lists at the top of the
script — `(display_label, slug)` pairs. Adding a model is one line plus
dropping a JSONL into the right split directory. The slug is the filename
stem (no `.jsonl`).

## Naming convention

- `cards-*` prefix → fine-tuned (we trained these)
- plain slug → zero-shot base, API model, or other

The twitter split uses `labels` instead of `true_claims` for ground truth;
the report copies it across automatically.
