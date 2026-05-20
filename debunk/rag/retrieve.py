"""Hybrid retrieval over the indexed climate KB.

  HybridRetriever(index_dir).retrieve(query, k=5) -> list[dict]

Pipeline (per query):
  1. Encode the query with the same dense model used at indexing time.
  2. FAISS top-N + BM25 top-N (default N=30 each).
  3. Reciprocal Rank Fusion to merge into a single ranked list.
  4. Optional cross-encoder rerank over the top-M (default 30) → return top-k.

The reranker is opt-in: cheap GPU-aided cross-encoder pass, or skip entirely
to keep serving on CPU. RRF alone usually gets 90%+ of the recall benefit.

CLI smoke-test:
  python3 -m debunk.rag.retrieve "Amazon rainforest savanna tipping point"
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import threading
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from .index import tokenize  # same whitespace tokeniser as at index time

ROOT = Path(__file__).resolve().parents[1]


class HybridRetriever:
    """Dense + BM25 hybrid retriever with optional cross-encoder rerank.

    Cheap to instantiate (lazy-loads the reranker on first use)."""

    def __init__(
        self,
        index_dir: Path | str,
        *,
        device: str = "cuda",
        reranker: str | None = "BAAI/bge-reranker-v2-m3",
    ):
        self.dir = Path(index_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.embedder = SentenceTransformer(meta["embedder"], device=device,
                                              trust_remote_code=True)
        if hasattr(self.embedder, "max_seq_length"):
            self.embedder.max_seq_length = meta.get("max_seq", 1024)
        self.faiss = faiss.read_index(str(self.dir / "dense.faiss"))
        with open(self.dir / "bm25.pkl", "rb") as f:
            bm = pickle.load(f)
        self.bm25 = bm["bm25"]
        self.chunks: list[dict] = []
        with open(self.dir / "chunks.jsonl") as f:
            for l in f:
                self.chunks.append(json.loads(l))
        if len(self.chunks) != meta["count"]:
            print(f"[warn] chunk count mismatch: meta={meta['count']} "
                  f"file={len(self.chunks)}", file=sys.stderr)

        self._reranker_name = reranker
        self._reranker_device = device
        self._reranker: CrossEncoder | None = None
        # Serialize GPU-bound work so a multi-threaded caller (e.g. infer.py
        # with --max-workers > 1) shares one device without OOMing. The API
        # call is still parallel — only embed+rerank serializes.
        self._gpu_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy reranker
    # ------------------------------------------------------------------

    @property
    def reranker(self) -> CrossEncoder | None:
        if self._reranker_name is None:
            return None
        if self._reranker is None:
            self._reranker = CrossEncoder(self._reranker_name,
                                            device=self._reranker_device,
                                            trust_remote_code=True)
        return self._reranker

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _dense_topk(self, query: str, n: int) -> list[tuple[int, float]]:
        q = self.embedder.encode([query], normalize_embeddings=True,
                                  convert_to_numpy=True).astype(np.float32)
        scores, idxs = self.faiss.search(q, n)
        return list(zip(idxs[0].tolist(), scores[0].tolist()))

    def _bm25_topk(self, query: str, n: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        # argpartition is O(n) vs full sort
        if n >= len(scores):
            order = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, n)[:n]
            order = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in order[:n]]

    @staticmethod
    def _rrf_merge(rankings: list[list[tuple[int, float]]], k: int = 60
                   ) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion. `k` is the RRF damping constant (60 is the
        canonical default from the original paper)."""
        agg: dict[int, float] = {}
        for ranked in rankings:
            for rank, (idx, _) in enumerate(ranked):
                agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(agg.items(), key=lambda x: -x[1])

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        candidate_pool: int = 30,
        use_reranker: bool = True,
    ) -> list[dict]:
        """Return top-k chunks for `query`. Each result carries the chunk
        record plus a `score` field (RRF score, or reranker score if used)."""
        with self._gpu_lock:
            dense = self._dense_topk(query, candidate_pool)
        sparse = self._bm25_topk(query, candidate_pool)  # CPU; lock-free
        merged = self._rrf_merge([dense, sparse])

        if use_reranker and self.reranker is not None:
            pool_idx = [i for i, _ in merged[:candidate_pool]]
            pairs = [(query, self.chunks[i]["text"]) for i in pool_idx]
            with self._gpu_lock:
                scores = self.reranker.predict(pairs, show_progress_bar=False)
            order = np.argsort(-np.asarray(scores))[:k]
            return [
                {**self.chunks[pool_idx[o]], "score": float(scores[o])}
                for o in order
            ]

        return [
            {**self.chunks[i], "score": float(s)}
            for i, s in merged[:k]
        ]


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

def _default_index_dir() -> Path:
    rag_root = ROOT / "data" / "rag"
    candidates = [p for p in rag_root.iterdir() if (p / "meta.json").exists()]
    if not candidates:
        sys.exit(f"no index dirs found under {rag_root}")
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    idx = Path(args.index_dir) if args.index_dir else _default_index_dir()
    print(f"index: {idx}")
    r = HybridRetriever(idx, device=args.device)
    hits = r.retrieve(args.query, k=args.k, use_reranker=not args.no_rerank)
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] score={h['score']:.4f}  {h.get('source')}  {h.get('title','')[:80]}")
        print(f"    {h.get('url','')}")
        print(f"    {h['text'][:300].strip()}…")


if __name__ == "__main__":
    main()
