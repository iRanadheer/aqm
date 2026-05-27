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
    ap.add_argument("--with-context", action="store_true", default=False,
                    help="Encode each chunk *with its source document* visible "
                         "to the embedder. Required for context-aware models "
                         "like pplx-embed-context-v1. Loads kb_combined.jsonl "
                         "to look up each chunk's parent document by url.")
    ap.add_argument("--kb", default="data/raw/kb_combined.jsonl",
                    help="Path to the raw KB (used only with --with-context to "
                         "look up parent documents).")
    ap.add_argument("--context-tokens", type=int, default=1024,
                    help="With --with-context, max tokens of document context "
                         "to prepend to each chunk. The remaining budget "
                         "(max_seq - this) is left for the chunk text.")
    args = ap.parse_args()

    src = Path(args.chunks)
    if not src.exists():
        sys.exit(f"chunks not found: {src} — run `python3 -m debunk.rag.chunk` first")

    # Distinct output dir per embedder so multiple indices can coexist; suffix
    # `-ctx` when context mode is on so context / non-context variants of the
    # same model don't clobber each other.
    slug = _slug(args.embedder)
    if args.with_context:
        slug = f"{slug}-ctx"
    out_dir = Path(args.out_dir) / slug
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

    # ----- Document-context lookup (--with-context only) -----
    # Build a url -> full-document-text map from the raw KB; each chunk's
    # `url` field points back to its parent. Truncate context per the embedder
    # tokenizer so we still leave budget for the chunk itself.
    # NOTE: pplx-embed-context handles document-aware encoding internally
    # (via per-document chunk batching with [SEP]); skip this manual splice.
    doc_lookup: dict[str, str] = {}
    is_pplx_ctx_early = "pplx-embed-context" in args.embedder.lower()
    if args.with_context and is_pplx_ctx_early:
        print("Note: --with-context is a no-op for pplx-embed-context "
              "(model handles document context internally).")
    if args.with_context and not is_pplx_ctx_early:
        kb_path = Path(args.kb)
        if not kb_path.is_absolute():
            kb_path = ROOT.parent / args.kb if not (ROOT / args.kb).exists() else (ROOT / args.kb)
            # Walk back to debunk/ root regardless of cwd.
            for cand in (ROOT / args.kb, Path.cwd() / args.kb, Path(args.kb)):
                if cand.exists():
                    kb_path = cand
                    break
        if not kb_path.exists():
            sys.exit(f"--with-context needs the raw KB; not found: {kb_path}")
        print(f"Loading raw KB from {kb_path} for context lookup …")
        with open(kb_path) as f:
            for line in f:
                r = json.loads(line)
                url = r.get("url")
                if url and r.get("content"):
                    doc_lookup[url] = r["content"]
        print(f"  {len(doc_lookup)} documents indexed by url")

    # ----- BM25 -----
    print("Tokenising for BM25 …")
    tokenised = [tokenize(t) for t in tqdm(texts)]
    print("Building BM25Okapi …")
    t0 = time.time()
    bm25 = BM25Okapi(tokenised)
    print(f"  built in {time.time()-t0:.1f}s")
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "tokenised": tokenised}, f)

    # ----- Dense embedding -----
    is_pplx_ctx = "pplx-embed-context" in args.embedder.lower()

    if is_pplx_ctx:
        # Perplexity's contextual embedder has a custom transformers-only API:
        #   model.encode(list[list[str]])
        # — outer list = documents, inner list = chunks within that document.
        # The model joins same-document chunks internally with [SEP] so each
        # chunk's vector is computed with neighbouring chunks visible. We
        # group our flat chunk list by `doc_index`, encode per-document, and
        # reassemble back into the original flat order.
        # ----- Perplexity hosted API path (no local model) -----
        # Uses /v1/contextualizedembeddings; embeddings are base64-encoded
        # int8 vectors that we decode and L2-normalise for FAISS IP.
        # Per-request limits: 512 docs, 16k total chunks, 120k total tokens,
        # 32k tokens/doc. We batch conservatively on the token side.
        import base64
        import os
        from collections import defaultdict
        import requests  # noqa: E402
        from dotenv import load_dotenv  # noqa: E402

        # debunk/.env (two levels up: rag/index.py -> rag -> debunk)
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            sys.exit("PERPLEXITY_API_KEY not set — add it to debunk/.env")
        ENDPOINT = "https://api.perplexity.ai/v1/contextualizedembeddings"
        print(f"Embedding via Perplexity API: {args.embedder}  -> {ENDPOINT}")

        # Group chunks by parent document while preserving order.
        per_doc: dict[int, list[str]] = defaultdict(list)
        per_doc_ix: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            per_doc[c["doc_index"]].append(c["text"])
            per_doc_ix[c["doc_index"]].append(i)
        doc_keys = list(per_doc.keys())
        print(f"  {len(doc_keys)} documents, {n} chunks total")

        # API limits: ≤512 docs, ≤16k chunks, ≤120k tokens per request, AND
        # ≤32k tokens per individual document. Use the actual Qwen3 tokenizer
        # (pplx-embed-context is built on Qwen3, so this matches their server
        # token count to within a few percent).
        TOKEN_BUDGET = 80_000        # per-request total (under 120k)
        PER_DOC_BUDGET = 30_000      # per-individual-doc (under 32k)
        CHUNK_LIMIT = 480
        DOC_LIMIT = 480

        from transformers import AutoTokenizer  # noqa: E402
        print("  loading Qwen3 tokenizer for accurate token counts …")
        _pplx_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

        # Pre-compute chunk token counts (one pass; reuse for both per-doc
        # split planning and per-request packing).
        print("  counting tokens per chunk …")
        chunk_tokens_cache: list[int] = []
        for c in tqdm(chunks):
            chunk_tokens_cache.append(
                len(_pplx_tok.encode(c["text"], add_special_tokens=False))
            )

        def _est_tokens_by_chunk_idx(i: int) -> int:
            return chunk_tokens_cache[i]

        # Split any document whose chunks together exceed the per-doc budget
        # into multiple "virtual documents". Each virtual doc preserves local
        # neighbour context for its slice; the chunks that land on boundary
        # lose context to the next slice — acceptable given the alternative
        # is dropping them entirely.
        virtual_docs: list[tuple[list[int], list[str]]] = []  # (indices, texts)
        n_split = 0
        for d in doc_keys:
            doc_chunks = per_doc[d]
            doc_ixs = per_doc_ix[d]
            total = sum(_est_tokens_by_chunk_idx(i) for i in doc_ixs)
            if total <= PER_DOC_BUDGET:
                virtual_docs.append((doc_ixs, doc_chunks))
                continue
            # Slice the doc into virtual docs that each fit the budget.
            n_split += 1
            cur_t, cur_c, cur_i = 0, [], []
            for ix, c in zip(doc_ixs, doc_chunks):
                t = _est_tokens_by_chunk_idx(ix)
                if cur_c and cur_t + t > PER_DOC_BUDGET:
                    virtual_docs.append((cur_i, cur_c))
                    cur_t, cur_c, cur_i = 0, [], []
                cur_c.append(c)
                cur_i.append(ix)
                cur_t += t
            if cur_c:
                virtual_docs.append((cur_i, cur_c))
        if n_split:
            print(f"  split {n_split} oversized doc(s) into "
                  f"{len(virtual_docs) - (len(doc_keys) - n_split)} virtual docs")

        all_embs: list = [None] * n
        sess = requests.Session()
        sess.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

        # Pack virtual docs into requests up to the budget.
        batches: list[list[int]] = []
        cur_batch: list[int] = []
        cur_chunks = cur_tokens = 0
        for v_idx, (ixs, texts_v) in enumerate(virtual_docs):
            v_tokens = sum(_est_tokens_by_chunk_idx(i) for i in ixs)
            v_chunks = len(texts_v)
            if cur_batch and (
                cur_chunks + v_chunks > CHUNK_LIMIT
                or cur_tokens + v_tokens > TOKEN_BUDGET
                or len(cur_batch) + 1 > DOC_LIMIT
            ):
                batches.append(cur_batch)
                cur_batch, cur_chunks, cur_tokens = [], 0, 0
            cur_batch.append(v_idx)
            cur_chunks += v_chunks
            cur_tokens += v_tokens
        if cur_batch:
            batches.append(cur_batch)
        print(f"  packed into {len(batches)} API requests")

        # Send requests in parallel — Perplexity allows concurrent calls and
        # the API loop is otherwise network-bound. 16 workers ≈ 10x faster
        # than sequential without hitting rate limits.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _send(batch):
            payload = {
                "model": args.embedder.split("/")[-1],
                "input": [virtual_docs[v_idx][1] for v_idx in batch],
            }
            r = sess.post(ENDPOINT, json=payload, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(
                    f"Perplexity API {r.status_code}: {r.text[:500]}")
            return batch, r.json()

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(_send, b): b for b in batches}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="pplx API"):
                batch, resp = fut.result()
                for doc_pos, doc_obj in enumerate(resp["data"]):
                    v_idx = batch[doc_pos]
                    target_ixs = virtual_docs[v_idx][0]
                    for chunk_pos, chunk_obj in enumerate(doc_obj["data"]):
                        b64 = chunk_obj["embedding"]
                        raw = base64.b64decode(b64)
                        vec = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
                        all_embs[target_ixs[chunk_pos]] = vec

        emb = np.stack(all_embs)
        # Perplexity embeddings are unnormalised — L2-norm for FAISS IP.
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        print(f"  embedded in {time.time()-t0:.1f}s  -> shape {emb.shape}")
        np.save(out_dir / "embeddings.npy", emb)

    else:
        # ----- Standard path (vLLM) -----
        # Truncate to the embedder's tokenizer first.
        from transformers import AutoTokenizer  # noqa: E402
        print(f"Loading embedder tokenizer ({args.embedder}) …")
        _emb_tok = AutoTokenizer.from_pretrained(args.embedder, trust_remote_code=True)
        safety = 8
        print(f"Truncating chunks to fit {args.embedder}'s tokenizer "
              f"(max_seq={args.max_seq}) …")
        n_truncated = 0
        for i, t in enumerate(texts):
            ids = _emb_tok.encode(t, add_special_tokens=False, truncation=False)
            if len(ids) > args.max_seq - safety:
                texts[i] = _emb_tok.decode(ids[: args.max_seq - safety],
                                            skip_special_tokens=True)
                n_truncated += 1
        print(f"  truncated {n_truncated}/{len(texts)} chunks")

        from vllm import LLM  # noqa: E402 — heavy, kept local to this fn
        print(f"Loading embedder {args.embedder} via vLLM "
              f"({'fp16' if args.fp16 else 'fp32'}) …")
        llm = LLM(
            model=args.embedder,
            runner="pooling",
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
