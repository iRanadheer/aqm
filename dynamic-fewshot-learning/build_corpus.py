# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "numpy", "tqdm", "python-dotenv"]
# ///
"""Embed a task's few-shot corpus (train + train_eval) and cache to disk.

The corpus is the exact labelled pool the fine-tuned models learned from, so
the few-shot comparison is fair. Embeddings come from an OpenAI-compatible
endpoint (Qwen3-Embedding-0.6B on vLLM by default).

  vllm serve Qwen/Qwen3-Embedding-0.6B
  uv run build_corpus.py --task cards
  uv run build_corpus.py --task wind --embed-model Qwen/Qwen3-Embedding-0.6B
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from embedder import DEFAULT_EMBED_MODEL, DEFAULT_EMBED_URL, Embedder, cache_path, save_cache
from tasks import get_task

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--task", required=True, choices=["cards", "wind"])
ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
ap.add_argument("--embed-url", default=DEFAULT_EMBED_URL)
ap.add_argument("--batch-size", type=int, default=64)
args = ap.parse_args()

task = get_task(args.task)
corpus = task.load_corpus()
print(f"Task:   {task.name}")
print(f"Corpus: {len(corpus)} examples (train + train_eval)")
print(f"Model:  {args.embed_model}  @ {args.embed_url}")

embedder = Embedder(args.embed_model, args.embed_url)
emb = embedder.encode([ex.text for ex in corpus], batch_size=args.batch_size)

out = save_cache(ROOT / "data" / "embeddings", task.name, args.embed_model, emb)
print(f"Cached: {out}  shape={emb.shape}")
