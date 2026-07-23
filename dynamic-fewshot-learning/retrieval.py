"""Few-shot example selection: static and dynamic.

Both selectors draw from the SAME corpus (train + train_eval) and return the
SAME number of demos (k), so the only variable between the two regimes is
*which* k examples are chosen:

  StaticSelector  — one fixed, stratified sample of k demos, reused for every
                    test item (the classic "hand-pick / random pool" baseline).
  DynamicSelector — the k corpus items whose embeddings are most cosine-similar
                    to the item being classified (this chapter's method).

Embeddings are produced by build_corpus.py and cached to disk; DynamicSelector
just loads the matrix. Zero-shot needs no selector (k=0).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from tasks import Example


class StaticSelector:
    """A single fixed set of k demos, stratified across strat buckets, seed-fixed."""

    def __init__(self, corpus: list[Example], k: int, strat_keys: list[str], seed: int = 42):
        self.demos = self._sample(corpus, k, strat_keys, seed)

    @staticmethod
    def _sample(corpus, k, strat_keys, seed):
        rng = random.Random(seed)
        # Round-robin across buckets for even coverage, then fill to k.
        buckets: dict[str, list[int]] = {}
        for i, key in enumerate(strat_keys):
            buckets.setdefault(key, []).append(i)
        for idxs in buckets.values():
            rng.shuffle(idxs)
        order = sorted(buckets)
        rng.shuffle(order)
        picked: list[int] = []
        pos = {key: 0 for key in buckets}
        while len(picked) < min(k, len(corpus)):
            progressed = False
            for key in order:
                if len(picked) >= k:
                    break
                if pos[key] < len(buckets[key]):
                    picked.append(buckets[key][pos[key]])
                    pos[key] += 1
                    progressed = True
            if not progressed:
                break
        return [corpus[i] for i in picked]

    def select(self, query_text: str) -> list[Example]:
        return self.demos


class DynamicSelector:
    """Per-item top-k by cosine similarity over cached corpus embeddings."""

    def __init__(self, corpus: list[Example], k: int, embeddings: np.ndarray, embedder):
        assert len(corpus) == embeddings.shape[0], "corpus/embeddings length mismatch"
        self.corpus = corpus
        self.k = k
        self.embedder = embedder
        # L2-normalize so dot product == cosine similarity.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.emb = embeddings / np.clip(norms, 1e-12, None)

    def select(self, query_text: str) -> list[Example]:
        q = self.embedder.encode([query_text], normalize=True)[0]
        sims = self.emb @ q
        top = np.argsort(-sims)[: self.k]
        # Return most-similar LAST so the closest example sits nearest the query
        # (recency tends to weigh more in-context).
        return [self.corpus[i] for i in reversed(top)]


def build_selector(regime, corpus, k, task, cache_dir, embed_model, embed_url):
    """Factory. regime in {static, dynamic}."""
    if regime == "static":
        strat = [task.strat_key(ex) for ex in corpus]
        return StaticSelector(corpus, k, strat)
    if regime == "dynamic":
        from embedder import Embedder, load_cache
        emb = load_cache(Path(cache_dir), task.name, embed_model, expected=len(corpus))
        return DynamicSelector(corpus, k, emb, Embedder(embed_model, embed_url))
    raise SystemExit(f"Unknown regime: {regime}")
