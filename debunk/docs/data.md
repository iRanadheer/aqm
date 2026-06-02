# Data

Two artifact groups live in `data/`:

1. **Benchmark** — the Leippold et al. (2024) climate-veracity claims, the
   only thing models are scored against.
2. **RAG knowledge base** — a climate-news corpus chunked + indexed for
   retrieval, used only when a model is run with `--rag-index` (see
   [rag.md](rag.md)).

```
data/raw/Leippold2024_veracity.csv ──prepare_splits.py──> data/test/test.jsonl
                                       (dedup, normalise)

data/raw/kb_combined.jsonl ──rag/chunk.py──> data/rag/chunks.jsonl ──rag/index.py──> data/rag/<embedder>/
```

`prepare_splits.py` is deterministic — rerunning it reproduces the
checked-in `test.jsonl` byte-for-byte.

---

## 1. Benchmark — `data/test/test.jsonl`

### Source

`data/raw/Leippold2024_veracity.csv` — 169 climate claims (170 lines incl.
header), each hand-labelled by the authors of:

> Leippold, M. et al. *Climinator: A fact-checking AI for climate science.*
> (2024). The paper PDF and the credibility-hierarchy figure are checked in
> at `data/raw/climinator_paper.pdf` and `data/raw/climinator_hierarchy.webp`.

The CSV carries three parallel gold columns plus a de-dup flag:

| CSV column | Meaning |
|---|---|
| `Veracity` | 4-class veracity (`TRUE` / `MISLEADING` / `FALSE` / `UNVERIFIABLE`) |
| `Climate_Feedback` | 12-class Climate Feedback / Climinator label (`Correct` … `Flawed Reasoning`) |
| `Climinator` | the paper's own CLIM model prediction (used as a baseline, **not** gold) |
| `Duplicated` | `True` for near-duplicate claims dropped from the published benchmark |

### `prepare_splits.py`

Reads the CSV, drops `Duplicated == True` rows (10 of them → **160 rows**),
uppercases `Veracity`, and writes `{metadata, gold}` records. Gold columns
land **last** so the `pred_*` fields `infer.py` appends line up after them.

```bash
python prepare_splits.py                    # de-duped, 160 rows (default)
python prepare_splits.py --keep-duplicates  # all 169 rows
```

Output record (`data/test/test.jsonl`):

```json
{
  "itemId":          "leippold_001",
  "claim":           "<verbatim claim text>",
  "source":          "<source attribution>",
  "date":            "DD-MMM-YY",
  "true_veracity":   "FALSE",
  "true_cfb_label":  "Inaccurate",
  "true_climinator": "incorrect"
}
```

| Field | From CSV | Notes |
|---|---|---|
| `itemId` | `ID` | zero-padded `leippold_NNN` |
| `claim` / `source` / `date` | `Claim` / `Source` / `Date` | stripped verbatim |
| `true_veracity` | `Veracity` | uppercased; scored by the `veracity*` prompts |
| `true_cfb_label` | `Climate_Feedback` | raw case (`Mostly_Accurate`, …); scored by the `climinator*` prompts |
| `true_climinator` | `Climinator` | the paper's CLIM prediction — feeds the "Paper CLIM" baseline, never used as gold |

The two benchmarks are **independent** — `veracity*` prompts score against
`true_veracity`, `climinator*` prompts against `true_cfb_label`. There is no
cross-mapping between the taxonomies. Gold strings are normalised at scoring
time (`prompts.normalise_gold`: upper-case, `_`/`-` → space), so
`Flawed_Reasoning` matches the codebook's `FLAWED REASONING`.

---

## 2. RAG knowledge base

The retrieval corpus is a static, checked-in climate-news dump. It is
**only** read when chunking/indexing — inference reads the *index*, not
these raw files.

| File | Rows | Role |
|---|---:|---|
| `data/raw/kb_combined.jsonl` | 51,035 | combined climate-news / fact-check KB — the chunk source |
| `data/raw/the_ccc_org_crawl_20052026.jsonl` | 1,020 | climate-disinfo crawl folded into the KB |
| `data/rag/chunks.jsonl` | 65,953 | `kb_combined` split into ~500-token overlapping windows |

Indices built from the chunks live under `data/rag/<embedder-slug>/`:

| Index dir | Embedder | Chunks | Built |
|---|---|---:|---|
| `perplexity-ai-pplx-embed-context-v1-0-6b/` | `perplexity-ai/pplx-embed-context-v1-0.6b` | 65,953 | 2026-05-21 |
| `qwen-qwen3-embedding-0-6b/` | `Qwen/Qwen3-Embedding-0.6B` | 171,668 | 2026-05-20 |

Each index dir holds `dense.faiss`, `bm25.pkl`, `embeddings.npy`,
`chunks.jsonl`, and `meta.json`. See [rag.md](rag.md) for how they're built
and queried. The `kb_combined.jsonl` / FAISS / `embeddings.npy` artifacts
are large and git-ignored.

---

## Result files — `data/results/test/`

`infer.py` writes one JSONL per `(model, prompt)` at
`data/results/test/<slug>-<prompt>.jsonl`. Each row is the `test.jsonl`
record plus:

- `response` — raw model output (`<think>…</think>` + a `assessment:` YAML
  block), or `"ERROR: …"` on a backend failure.
- `citations` — URLs extracted from the model's response annotations (RAG /
  online runs).
- `pred_label` — the parsed assessment (via `prompts.extract_raw_label`).

Earlier prompt-version runs (`climinator_v5`, the `v4`-vs-`v5` comparison
plots) are archived under `.archives/` and excluded from tracking. The
`data/results/test_baseline/` dir holds the first-pass `veracityV1` /
`climinator` (v1) runs kept for reference; the headline numbers come from
the `climinator_v4` + `veracityV2` files (see [reports.md](reports.md)).
