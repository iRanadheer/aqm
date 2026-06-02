# debunk

Climate-claim fact-checking benchmark over the Leippold et al. (2024)
veracity dataset (169 claims; 160 after de-dup). A model assesses a single
claim and emits one label; we score against two independent gold taxonomies.
Per-stage docs live in [docs/](docs/).

| Doc | Covers |
|---|---|
| [docs/data.md](docs/data.md) | `prepare_splits.py`, the Leippold benchmark, `test.jsonl` schema, the RAG knowledge base |
| [docs/prompts.md](docs/prompts.md) | `prompts.py` — veracity + climinator taxonomies, the v1→v5 prompt lineage, credibility hierarchy, parser |
| [docs/inference.md](docs/inference.md) | `infer.py` — backends, RAG / Exa evidence, output slugs, resume |
| [docs/rag.md](docs/rag.md) | `rag/` — chunk → index → hybrid retrieve (pplx-ctx + qwen indices) |
| [docs/reports.md](docs/reports.md) | `generate_report.py`, `plot_hierarchy.py`, `evals/significance.py` |

## Two benchmarks

| Prompt (production) | Label space | Scored against |
|---|---|---|
| `veracityV2` | 4-class: TRUE / MISLEADING / FALSE / UNVERIFIABLE | `true_veracity` |
| `climinator_v4` | 12-class Climate Feedback / Climinator scheme | `true_cfb_label` |

The two taxonomies are scored independently — no cross-mapping. Earlier
prompt versions (`veracityV1`, `climinator` v1/v2/v3/v5) are kept in
`prompts.py` for the ablation trail but are not in the headline report; see
[docs/prompts.md](docs/prompts.md).

## Pipeline

```
data/raw/Leippold2024_veracity.csv
    │ prepare_splits.py            (dedup → 160 rows, normalise)
    ▼
data/test/test.jsonl
    │ infer.py                     (any chat model; ± RAG / Exa evidence)
    ▼
data/results/test/<slug>-<prompt>.jsonl
    │ generate_report.py / plot_hierarchy.py / evals/significance.py
    ▼
metrics_summary.{json,md} · debunk_benchmark.png · significance/summary_*.md
```

Optional RAG branch (only when `infer.py --rag-index` is used):

```
data/raw/kb_combined.jsonl ──rag/chunk.py──> data/rag/chunks.jsonl ──rag/index.py──> data/rag/<embedder>/
```

## Data record format

Test rows (`data/test/test.jsonl`) — metadata first, gold labels last so the
`pred_*` fields `infer.py` appends line up after them:

```json
{
  "itemId":          "leippold_001",
  "claim":           "<verbatim claim>",
  "source":          "<source attribution>",
  "date":            "DD-MMM-YY",
  "true_veracity":   "FALSE",
  "true_cfb_label":  "Inaccurate",
  "true_climinator": "incorrect"
}
```

Result rows add `response` (raw model output, or `"ERROR: ..."` on a backend
failure), `citations`, and the parsed `pred_label`. See
[docs/data.md](docs/data.md).

## Setup

```bash
cp .env.example .env   # OPENAI_API_KEY / OPENROUTER_API_KEY / EXA_API_KEY / PERPLEXITY_API_KEY
pip install -r requirements.txt
```

## Quick run

```bash
python prepare_splits.py
python infer.py --backend openrouter --model anthropic/claude-opus-4.7 --prompt veracityV2
python infer.py --backend openrouter --model anthropic/claude-opus-4.7 --prompt climinator_v4
python generate_report.py
python plot_hierarchy.py
```
