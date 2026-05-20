"""Local RAG pipeline over the climate KB for the debunk benchmark.

  chunk.py    — split kb_combined.jsonl into ~800-token overlapping chunks
                with a `[Source: …]` metadata header.
  index.py    — build a dense FAISS index (Jina v5 / Qwen3 / BGE) and a BM25
                pickle over the same chunks.
  retrieve.py — hybrid retrieve (dense + BM25, merged via RRF) + optional
                cross-encoder rerank.

Designed to run one-time on a GPU box for indexing; serving is CPU-friendly.
"""
