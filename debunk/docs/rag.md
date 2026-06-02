# RAG (`rag/`)

A self-contained hybrid retriever over a climate-news knowledge base.
Three stages — chunk, index, retrieve — all CPU-servable at query time.

```
data/raw/kb_combined.jsonl ──chunk.py──> data/rag/chunks.jsonl ──index.py──> data/rag/<embedder>/
                                                                               │ dense.faiss + bm25.pkl
                                                                               ▼
                                                          retrieve.py (HybridRetriever) ──> top-k chunks
                                                                               ▲
                                                          infer.py --rag-index ┘
```

| File | Role |
|---|---|
| `chunk.py` | split KB docs into ~500-token overlapping windows with `[Source: …]` headers |
| `index.py` | build a FAISS dense index + a BM25 sparse index over the chunks |
| `retrieve.py` | hybrid dense+BM25 retrieval, RRF merge, optional cross-encoder rerank |

## 1. Chunking — `chunk.py`

Splits `kb_combined.jsonl` (51,035 docs) into ~500-token windows
(`CHUNK_TOKENS=500`, `OVERLAP_TOKENS=100`) using the `o200k_base` tiktoken
tokenizer. Each chunk gets a `[Source: source | title | date]` header so a
retrieved snippet is self-describing. Output `data/rag/chunks.jsonl`
(65,953 chunks), one record per chunk:

```json
{"chunk_id": "source#doc.chunk", "doc_index": 0, "url": "...",
 "source": "berkeleyearth_org", "title": "...", "published_date": "...",
 "text": "[Source: ...]\n<chunk text>"}
```

Some sources (e.g. `science_feedback_org`, `snopes_com`) are excluded by
default to avoid leaking fact-check verdicts into the evidence.

## 2. Indexing — `index.py`

Builds two complementary indices per embedder under `data/rag/<slug>/`:

- **dense** — chunk embeddings → `embeddings.npy` (L2-normalised) + a FAISS
  `IndexFlatIP` (`dense.faiss`).
- **sparse** — a pickled `BM25Okapi` over the same tokenised chunks
  (`bm25.pkl`).

Plus `chunks.jsonl` and `meta.json` (embedder, dim, count, build time). Two
embedding backends:

| Path | Embedder | Notes |
|---|---|---|
| **local (vLLM)** | any sentence-transformers model, default `Qwen/Qwen3-Embedding-0.6B` | encodes chunks independently |
| **Perplexity contextual** (`--with-context`) | `perplexity-ai/pplx-embed-context-v1-0.6b` via `/v1/contextualizedembeddings` | groups chunks by document so each chunk is embedded *aware of its siblings*; needs `PERPLEXITY_API_KEY` |

```bash
# local dense+BM25 index
python3 -m debunk.rag.index --embedder Qwen/Qwen3-Embedding-0.6B
# Perplexity context-aware index (output dir gets a -ctx suffix)
python3 -m debunk.rag.index --embedder perplexity-ai/pplx-embed-context-v1-0.6b --with-context
```

Two indices are checked in (`meta.json` counts): the context-aware
`perplexity-ai-pplx-embed-context-v1-0-6b/` (65,953 chunks) and the local
`qwen-qwen3-embedding-0-6b/` (171,668 chunks).

## 3. Retrieval — `retrieve.py`

`HybridRetriever(index_dir).retrieve(query, k, candidate_pool=30,
use_reranker=True)` returns the top-k chunks:

1. **dense top-k** — embed the query (locally, or via the Perplexity
   contextual API for a `-ctx` index) and search FAISS.
2. **sparse top-k** — BM25 over the tokenised query.
3. **RRF merge** — reciprocal rank fusion (`k=60`) combines the two rankings.
4. **rerank** (optional) — a cross-encoder (`BAAI/bge-reranker-v2-m3`)
   re-scores the merged candidate pool; loaded lazily and GPU-thread-locked.

```bash
# smoke-test against the first index found in data/rag/
python3 -m debunk.rag.retrieve "Is the Amazon rainforest a net carbon source?"
```

## Wiring into inference

`infer.py --rag-index data/rag/<dir>` instantiates a `HybridRetriever`; each
claim is retrieved (`--rag-k` chunks, `--rag-no-rerank` to skip the
cross-encoder) and the chunks are formatted into the `USER_TEMPLATE_RAG`
evidence block. The output slug encodes which index was used
(`-rag-pplx-ctx` / `-rag-qwen`) so runs are traceable — see
[inference.md](inference.md). The `-rag-pplx-ctx` vs `-rag-qwen` results are
the offline-with-retrieval rows in the headline report.

> `kb_combined.jsonl`, `embeddings.npy`, and the FAISS files are large and
> git-ignored — rebuild them with `chunk.py` + `index.py` from the raw KB.
