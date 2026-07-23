"""Embeddings via an OpenAI-compatible endpoint (Qwen3-Embedding-0.6B on vLLM).

Same backend convention as infer.py: an OpenAI client pointed at a base_url.
Embeddings are cached to disk keyed by (task, embed_model) so cards/wind and
different embedders never collide. build_corpus.py writes the cache;
retrieval.py reads it.

    vllm serve Qwen/Qwen3-Embedding-0.6B  # -> http://localhost:8000/v1
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from openai import OpenAI

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBED_URL = "http://localhost:8000/v1"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def cache_path(cache_dir: Path, task: str, embed_model: str) -> Path:
    return Path(cache_dir) / task / f"emb__{_slug(embed_model)}.npy"


class Embedder:
    """OpenAI-compatible embedding client (vLLM by default; no key needed)."""

    def __init__(self, model=DEFAULT_EMBED_MODEL, base_url=DEFAULT_EMBED_URL, api_key="dummy"):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "dummy")

    def encode(self, texts: list[str], normalize: bool = False, batch_size: int = 64) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            vecs.extend(d.embedding for d in resp.data)
        emb = np.asarray(vecs, dtype=np.float32)
        if normalize:
            emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        return emb


def load_cache(cache_dir: Path, task: str, embed_model: str, expected: int | None = None) -> np.ndarray:
    p = cache_path(cache_dir, task, embed_model)
    if not p.exists():
        raise SystemExit(
            f"No embedding cache at {p}.\n"
            f"Build it first:  uv run build_corpus.py --task {task} --embed-model {embed_model}"
        )
    emb = np.load(p)
    if expected is not None and emb.shape[0] != expected:
        raise SystemExit(f"Cache {p} has {emb.shape[0]} rows but corpus has {expected}. Rebuild it.")
    return emb


def save_cache(cache_dir: Path, task: str, embed_model: str, emb: np.ndarray) -> Path:
    p = cache_path(cache_dir, task, embed_model)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, emb)
    return p
