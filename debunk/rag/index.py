"""Build dense (FAISS) + sparse (BM25) indices over `chunks.jsonl`.

  Dense:  Sentence-Transformers model (default Jina v5-text-small).
          Embeddings stored as float32 NPY + faiss inner-product index.
  Sparse: rank_bm25 over the same chunk texts, pickled.

Outputs land under `debunk/data/rag/<model-slug>/`:
  - `chunks.jsonl`     symlink/copy of the source chunks for downstream lookup
  - `embeddings.npy`   N x d float32, L2-normalised (so IP == cosine)
  - `dense.faiss`      faiss IndexFlatIP (no quantisation — 500k chunks fits)
  - `bm25.pkl`         pickled BM25Okapi state + tokenised corpus
  - `meta.json`        model name, dim, count, version stamps

Usage:
  python3 -m debunk.rag.index            # default Jina v5-text-small
  python3 -m debunk.rag.index --embedder Qwen/Qwen3-Embedding-0.6B
  python3 -m debunk.rag.index --batch 256 --device cuda
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "rag" / "chunks.jsonl"
DEFAULT_OUT = ROOT / "data" / "rag"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# BM25 wants pre-tokenised input. Cheap whitespace + lower + strip-punct is
# fine for English fact-check / science corpora; we're not chasing the last
# few points of recall here — that's the reranker's job.
_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK_RE.findall(text.lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--embedder", default="Qwen/Qwen3-Embedding-0.6B",
                    help="Any sentence-transformers-compatible model. "
                         "Default: Qwen3-Embedding-0.6B (current MTEB leader in "
                         "its size class). Try jinaai/jina-embeddings-v5-text-small "
                         "or BAAI/bge-m3 as alternatives.")
    ap.add_argument("--batch", type=int, default=128,
                    help="Encoder batch size. Default 128 is safe for Qwen3-0.6B "
                         "fp16 on a 24GB GPU at 1024-token sequences. Bump to "
                         "256 if your sequences are shorter or VRAM allows.")
    ap.add_argument("--device", default="cuda",
                    help="cuda | cpu | mps")
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="Load embedder weights in float16 (default on). "
                         "~2x throughput on Ampere+, negligible quality loss "
                         "for this model size. Pass --no-fp16 to disable.")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--max-seq", type=int, default=2048,
                    help="Truncate inputs to this many tokens at the embedder. "
                         "Default 2048 leaves headroom for our 800-token "
                         "chunks + `[Source: …]` headers. Qwen3-Embedding "
                         "natively supports 32k; Jina v5 supports 8k.")
    args = ap.parse_args()

    src = Path(args.chunks)
    if not src.exists():
        sys.exit(f"chunks not found: {src} — run `python3 -m debunk.rag.chunk` first")

    out_dir = Path(args.out_dir) / _slug(args.embedder)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from {src} …")
    chunks: list[dict] = []
    texts: list[str] = []
    with open(src) as f:
        for line in f:
            r = json.loads(line)
            chunks.append(r)
            texts.append(r["text"])
    n = len(chunks)
    print(f"  {n} chunks")

    # ----- BM25 -----
    print("Tokenising for BM25 …")
    tokenised = [tokenize(t) for t in tqdm(texts)]
    print("Building BM25Okapi …")
    t0 = time.time()
    bm25 = BM25Okapi(tokenised)
    print(f"  built in {time.time()-t0:.1f}s")
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "tokenised": tokenised}, f)

    # ----- Pre-truncate to the embedder's tokenizer -----
    # The chunker uses tiktoken/o200k (gpt-4o family). Embedders like Qwen3
    # have a different tokenizer that may produce 1.5–2x more tokens for the
    # same text, so chunks that fit our 800-token budget can overflow the
    # embedder's max_model_len. Truncate per-chunk with the actual tokenizer.
    from transformers import AutoTokenizer  # noqa: E402
    print(f"Truncating chunks to fit {args.embedder}'s tokenizer "
          f"(max_seq={args.max_seq}) …")
    _emb_tok = AutoTokenizer.from_pretrained(args.embedder, trust_remote_code=True)
    safety = 8  # leave room for any special tokens vLLM adds
    n_truncated = 0
    for i, t in enumerate(texts):
        ids = _emb_tok.encode(t, add_special_tokens=False, truncation=False)
        if len(ids) > args.max_seq - safety:
            texts[i] = _emb_tok.decode(ids[: args.max_seq - safety],
                                        skip_special_tokens=True)
            n_truncated += 1
    print(f"  truncated {n_truncated}/{len(texts)} chunks")

    # ----- Dense (vLLM) -----
    # vLLM handles its own batching, continuous scheduling, fp16, and CUDA
    # graph capture — much faster than sentence-transformers for a one-shot
    # index of 100k+ chunks.
    from vllm import LLM  # noqa: E402 — heavy, kept local to this fn
    print(f"Loading embedder {args.embedder} via vLLM "
          f"({'fp16' if args.fp16 else 'fp32'}) …")
    llm = LLM(
        model=args.embedder,
        runner="pooling",  # vLLM 0.10+ name; older releases use task="embed"
        dtype="float16" if args.fp16 else "float32",
        trust_remote_code=True,
        max_model_len=args.max_seq,
        enforce_eager=False,
    )

    print(f"Embedding {n} chunks …")
    t0 = time.time()
    outputs = llm.embed(texts)
    emb = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32)
                    for o in outputs])
    # L2-normalise so inner-product == cosine.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    print(f"  embedded in {time.time()-t0:.1f}s  -> shape {emb.shape}")
    np.save(out_dir / "embeddings.npy", emb)

    # Inner-product index over L2-normalised vectors == cosine similarity.
    # IndexFlatIP keeps full precision; 500k * 512 floats ≈ 1 GB — fine.
    print("Building FAISS IndexFlatIP …")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out_dir / "dense.faiss"))

    # Also drop a copy of the chunks alongside the index so retrieve.py only
    # needs one directory.
    with open(out_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "embedder": args.embedder,
        "dim":      int(emb.shape[1]),
        "count":    n,
        "max_seq":  args.max_seq,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nIndex written to {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:>20}  {f.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
