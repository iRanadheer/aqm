"""Chunk `kb_combined.jsonl` into RAG-ready chunks.

Splits each document body into ~800-token windows with 100-token overlap,
prepends a `[Source: <slug> | <title> | <date>]` header (helps the embedder
disambiguate near-duplicate text across fact-check sites and helps the
classifier emit citations later), and writes one chunk per line to
`debunk/data/rag/chunks.jsonl`.

The tokenizer is `o200k_base` (gpt-4o family) — same model that will end up
reading the retrieved evidence, so chunk size budgets line up with the
classifier's context.

Usage:
  python3 -m debunk.rag.chunk
  python3 -m debunk.rag.chunk --input data/raw/kb_combined.jsonl --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tiktoken
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "kb_combined.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "rag" / "chunks.jsonl"

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 100
# Minimum meaningful chunk — below this and the document tail is folded back
# into the previous chunk instead of becoming its own near-empty record.
MIN_CHUNK_TOKENS = 75

_ENC = tiktoken.get_encoding("o200k_base")


def _split_by_tokens(text: str, chunk: int = CHUNK_TOKENS,
                     overlap: int = OVERLAP_TOKENS) -> list[str]:
    ids = _ENC.encode(text)
    if not ids:
        return []
    step = chunk - overlap
    chunks: list[str] = []
    i = 0
    while i < len(ids):
        window = ids[i : i + chunk]
        chunks.append(_ENC.decode(window))
        if i + chunk >= len(ids):
            break
        i += step
    # If the last chunk is shorter than MIN, merge it into the previous one.
    if len(chunks) >= 2 and len(_ENC.encode(chunks[-1])) < MIN_CHUNK_TOKENS:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + " " + tail
    return chunks


def _header(row: dict) -> str:
    parts = [
        row.get("source") or "?",
        (row.get("title") or "").strip(),
        (row.get("published_date") or "").split(" ")[0] or "?",
    ]
    return f"[Source: {parts[0]} | {parts[1]} | {parts[2]}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N documents (debugging).")
    ap.add_argument("--exclude-source", action="append",
                    default=["science_feedback_org", "snopes_com"],
                    help="Drop documents whose `source` field matches. "
                         "Repeatable. Defaults: science_feedback_org (gold-"
                         "label leakage) and snopes_com (general fact-check, "
                         "most non-climate, ~63%% of the corpus by count).")
    args = ap.parse_args()
    excluded = set(args.exclude_source)

    src = Path(args.input)
    out = Path(args.output)
    if not src.exists():
        sys.exit(f"input not found: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_docs = n_chunks = n_skipped = 0
    with open(src) as fin, open(out, "w") as fout:
        for line in tqdm(fin, desc="chunking"):
            r = json.loads(line)
            if r.get("source") in excluded:
                n_skipped += 1
                continue
            text = (r.get("content") or "").strip()
            if not text:
                continue
            if args.limit and n_docs >= args.limit:
                break
            header = _header(r)
            pieces = _split_by_tokens(text)
            for j, piece in enumerate(pieces):
                chunk = {
                    "chunk_id":       f"{r.get('source','?')}#{n_docs}.{j}",
                    "doc_index":      n_docs,
                    "url":            r.get("url"),
                    "source":         r.get("source"),
                    "title":          r.get("title"),
                    "published_date": (r.get("published_date") or "").split(" ")[0] or None,
                    "text":           f"{header}\n{piece}",
                }
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n_chunks += 1
            n_docs += 1

    print(f"docs:    {n_docs}")
    print(f"skipped: {n_skipped} (excluded sources: {sorted(excluded)})")
    print(f"chunks:  {n_chunks}  -> {out}")


if __name__ == "__main__":
    main()
