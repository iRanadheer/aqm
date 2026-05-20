# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "tenacity", "tqdm", "pandas", "python-dotenv", "transformers"]
# ///
"""Run a chat-completion model on the Leippold2024 fact-checking benchmark.

Talks to any OpenAI-compatible endpoint. Three backends: vllm (default,
local), openai, openrouter. API key is read from the matching env var.

  uv run infer.py --backend openrouter --model anthropic/claude-opus-4.7
  uv run infer.py --backend openai --model gpt-5 --prompt climinator
  uv run infer.py --model C3DS/Debunky-Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prompts import PROMPT_VARIANTS, extract_raw_label, output_schema  # noqa: E402

BACKENDS = {
    "vllm":       ("http://localhost:8000/v1",     None),
    "openai":     ("https://api.openai.com/v1",    "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    # Exa Answer API — OpenAI-compatible /chat/completions wrapper around
    # their search-grounded answer endpoint. Single model name: "exa".
    "exa":        ("https://api.exa.ai",           "EXA_API_KEY"),
}

USER_TEMPLATE = "### Claim:\n{claim}\n\n### Source:\n{source}"
USER_TEMPLATE_RAG = (
    "### Claim:\n{claim}\n\n### Source:\n{source}\n\n"
    "### Evidence (retrieved from a vetted climate-science knowledge base):\n"
    "{evidence}\n\n"
    "Use the evidence above to ground your assessment. Cite chunks by their "
    "[id] when relevant. Evidence may be incomplete or off-topic — apply the "
    "force-fit guard from the codebook if so."
)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--backend", choices=list(BACKENDS), default="vllm")
ap.add_argument("--model", required=True)
ap.add_argument("--input", default="data/test/test.jsonl")
ap.add_argument("--output", default=None,
                help="Default: data/results/<split>/<model-slug>-<prompt>.jsonl")
ap.add_argument("--prompt", choices=list(PROMPT_VARIANTS), default="veracityV1")
ap.add_argument("--max-workers", type=int, default=20)
ap.add_argument("--max-tokens", type=int, default=4000,
                help="Default 4000 — high enough for reasoning models (GPT-5.x, "
                     "Gemini 2.5+) that burn hidden reasoning tokens before "
                     "emitting visible content.")
ap.add_argument("--limit", type=int, default=None, help="Only process first N rows")
ap.add_argument("--max-input-tokens", type=int, default=None,
                help="Truncate user content to N tokens (using the model's "
                     "own HuggingFace tokenizer). Off by default.")
ap.add_argument("--no-think", dest="think", action="store_false", default=True,
                help="Disable Qwen3 thinking-mode at inference "
                     "(chat_template_kwargs.enable_thinking=False).")
ap.add_argument("--rag-index", default=None,
                help="Path to a debunk/rag index dir (built via "
                     "`python -m debunk.rag.index`). When set, each user "
                     "message is augmented with top-k retrieved evidence "
                     "chunks. The output slug gets a `-rag` suffix.")
ap.add_argument("--rag-k", type=int, default=5,
                help="Number of evidence chunks to inject per claim.")
ap.add_argument("--rag-no-rerank", dest="rag_rerank", action="store_false",
                default=True,
                help="Skip the cross-encoder rerank step (faster, CPU-only "
                     "friendly).")
ap.add_argument("--exa-evidence", action="store_true", default=False,
                help="Use Exa Answer API as the evidence source — fetch Exa's "
                     "answer + citations for each claim, inject as evidence, "
                     "then classify with the main --model. Adds `-exa` to the "
                     "output slug. Mutex with --rag-index.")
ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                help="Don't call the LLM — print the fully assembled system "
                     "+ user message that would be sent for the first N rows, "
                     "then exit. Useful for auditing RAG retrieval.")
ap.add_argument("--dump-prompt-md", default=None, metavar="PATH",
                help="Write the fully assembled system + user prompt for one "
                     "claim to PATH as markdown, then exit. By default picks "
                     "the first row; override with --dump-itemId.")
ap.add_argument("--dump-itemId", default=None,
                help="When set with --dump-prompt-md, dump that specific row.")
args = ap.parse_args()
if args.rag_index and args.exa_evidence:
    sys.exit("--rag-index and --exa-evidence are mutually exclusive")

load_dotenv(ROOT / ".env")
base_url, key_env = BACKENDS[args.backend]
if key_env:
    api_key = os.environ.get(key_env)
    if not api_key:
        sys.exit(f"{key_env} not set")
    print(f"{key_env}: set ({api_key[:7]}…{api_key[-4:]})")
else:
    api_key = "dummy"
    print(f"{args.backend}: no API key required")

input_path = Path(args.input) if Path(args.input).is_absolute() else ROOT / args.input
if not input_path.exists():
    sys.exit(f"Input not found: {input_path}")

slug = re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")
if args.rag_index:
    slug = f"{slug}-rag"
elif args.exa_evidence:
    slug = f"{slug}-exa"
split = input_path.stem  # test -> test, val -> val
output_path = (
    Path(args.output) if args.output
    else ROOT / "data" / "results" / split / f"{slug}-{args.prompt}.jsonl"
)
if not output_path.is_absolute():
    output_path = ROOT / output_path

client = OpenAI(base_url=base_url, api_key=api_key)
# Ephemeral caching for all backends — providers that don't honor it ignore the field.
system_content = [{
    "type": "text",
    "text": PROMPT_VARIANTS[args.prompt],
    "cache_control": {"type": "ephemeral"},
}]

# Lazy-load the RAG retriever so non-RAG runs don't pull torch + sentence-
# transformers into memory. Module-level so worker threads share one
# retriever instance (the embedder/reranker calls are thread-safe).
_retriever = None
if args.rag_index:
    from rag.retrieve import HybridRetriever  # noqa: E402
    print(f"Loading RAG index: {args.rag_index}")
    _retriever = HybridRetriever(args.rag_index)
    print(f"  retriever ready: {len(_retriever.chunks)} chunks, "
          f"reranker={'on' if args.rag_rerank else 'off'}")


def _build_evidence(claim: str) -> str:
    """Retrieve top-k chunks for the claim and format them for the prompt."""
    hits = _retriever.retrieve(claim, k=args.rag_k, use_reranker=args.rag_rerank)
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(
            f"[{i}] {h.get('source','?')} · {h.get('title','')} · {h.get('published_date','?')}\n"
            f"URL: {h.get('url','')}\n"
            f"{h['text']}"
        )
    return "\n\n".join(blocks) if blocks else "(no evidence retrieved)"


# Lazy Exa-evidence client. Each claim → one Exa /answer call → injected as
# evidence into the main classifier's prompt. Used in addition to (not instead
# of) the local RAG path so we can compare both head-to-head.
_exa_client = None
if args.exa_evidence:
    _exa_key = os.environ.get("EXA_API_KEY")
    if not _exa_key:
        sys.exit("EXA_API_KEY not set")
    _exa_client = OpenAI(base_url="https://api.exa.ai", api_key=_exa_key)
    print("Exa evidence: enabled (https://api.exa.ai)")


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=3, min=3, max=30),
       retry=retry_if_exception_type(Exception))
def _exa_search(claim: str) -> tuple[str, list[str]]:
    """Hit Exa's /answer with the claim as a search query; return (answer,
    citation URLs). Retries are tight because each failed Exa call still
    blocks the downstream classifier call."""
    resp = _exa_client.chat.completions.create(
        model="exa",
        messages=[
            {"role": "system",
             "content": "Find authoritative climate-science sources for the user's claim."},
            {"role": "user",
             "content": f"Fact-check this climate claim against authoritative sources "
                        f"(IPCC, NASA, NOAA, peer-reviewed work). Quote relevant evidence "
                        f"verbatim where possible.\n\nClaim: {claim}"},
        ],
        extra_body={"text": True},  # include full source text in citations
    )
    msg = resp.choices[0].message
    urls: list[str] = []
    for c in (getattr(msg, "citations", None) or []):
        url = getattr(c, "url", None) or (c.get("url") if isinstance(c, dict) else None)
        if url:
            urls.append(url)
    return msg.content or "", urls


def _build_exa_evidence(claim: str) -> str:
    answer, urls = _exa_search(claim)
    cites = "\n".join(f"- {u}" for u in urls) or "(no citations returned)"
    return (
        f"Exa search summary:\n{answer}\n\n"
        f"Cited URLs (Exa retrieved these as the supporting sources):\n{cites}"
    )


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=5, min=5, max=60),
       retry=retry_if_exception_type(Exception))
def query(claim: str, source: str):
    """Return the full assistant `message` object (content + annotations)."""
    if _retriever is not None:
        user_text = USER_TEMPLATE_RAG.format(
            claim=claim, source=source, evidence=_build_evidence(claim),
        )
    elif _exa_client is not None:
        user_text = USER_TEMPLATE_RAG.format(
            claim=claim, source=source, evidence=_build_exa_evidence(claim),
        )
    else:
        user_text = USER_TEMPLATE.format(claim=claim, source=source)
    kwargs = dict(
        model=args.model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_text},
        ],
    )
    # Exa's /answer endpoint ignores temperature/max_tokens and 400s on some
    # unrecognized fields. It also ignores the system prompt — we constrain
    # the output via a JSON schema derived from our pydantic assessment model.
    if args.backend == "exa":
        kwargs["extra_body"] = {"output_schema": output_schema(args.prompt)}
    else:
        kwargs["temperature"] = 0
        kwargs["max_tokens"] = args.max_tokens
        if not args.think:
            kwargs.setdefault("extra_body", {})["chat_template_kwargs"] = {"enable_thinking": False}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message


def _extract_citations(msg) -> list[str]:
    """Pull URL citations from search-grounded models (e.g. perplexity/sonar,
    exa).

    Two shapes handled:
      - OpenAI-style `annotations` with `type: url_citation` (perplexity).
      - Exa's `citations` field — a list of objects with a `url` attribute.
    Returns [] for models that don't emit either. Safe to call on any
    OpenAI-compatible message object.
    """
    urls: list[str] = []
    for a in (getattr(msg, "annotations", None) or []):
        if getattr(a, "type", None) != "url_citation":
            continue
        uc = getattr(a, "url_citation", None)
        url = getattr(uc, "url", None) if uc is not None else None
        if url:
            urls.append(url)
    for c in (getattr(msg, "citations", None) or []):
        url = getattr(c, "url", None) or (c.get("url") if isinstance(c, dict) else None)
        if url:
            urls.append(url)
    return urls


if args.max_input_tokens:
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def _truncate(text: str) -> str:
        ids = _tok.encode(text, add_special_tokens=False)
        if len(ids) <= args.max_input_tokens:
            return text
        return _tok.decode(ids[: args.max_input_tokens], skip_special_tokens=True)
else:
    def _truncate(text: str) -> str:
        return text


def process(row: dict) -> dict:
    # Output layout: metadata first, then response/citations, then all
    # true_*/pred_* label fields collected at the end for easy visual diff.
    true_fields = {k: v for k, v in row.items() if k.startswith("true_")}
    out = {k: v for k, v in row.items() if not k.startswith("true_")}

    pred_label: str | None = None
    try:
        msg = query(_truncate(row["claim"]), row.get("source", ""))
        out["response"] = msg.content
        out["citations"] = _extract_citations(msg)
        pred_label = extract_raw_label(msg.content)
    except Exception as e:
        out["response"] = f"ERROR: {e}"
        out["citations"] = []

    out.update(true_fields)
    out["pred_label"] = pred_label
    return out


df = pd.read_json(input_path, lines=True).reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)

if args.dump_prompt_md:
    # Single-row markdown dump for auditing the assembled prompt + retrieved
    # evidence. Keeps system / user / evidence in separate sections so you
    # can eyeball what 4o-mini actually sees.
    records = df.to_dict("records")
    if args.dump_itemId:
        records = [r for r in records if r.get("itemId") == args.dump_itemId]
        if not records:
            sys.exit(f"itemId {args.dump_itemId!r} not found in {input_path}")
    row = records[0]
    claim = _truncate(row["claim"])
    src = row.get("source", "")
    if _retriever is not None:
        evidence_block = _build_evidence(claim)
        evidence_label = "Local hybrid RAG (Qwen3 dense + BM25 + BGE rerank)"
    elif _exa_client is not None:
        evidence_block = _build_exa_evidence(claim)
        evidence_label = "Exa Answer API"
    else:
        evidence_block = None
        evidence_label = None

    out_path = Path(args.dump_prompt_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# Prompt dump · itemId `{row.get('itemId')}`\n\n")
        f.write(f"- **Model:** `{args.model}` (backend: `{args.backend}`)\n")
        f.write(f"- **Prompt variant:** `{args.prompt}`\n")
        if evidence_label:
            f.write(f"- **Evidence source:** {evidence_label}\n")
        f.write(f"- **Gold `true_cfb_label`:** `{row.get('true_cfb_label')}`\n")
        f.write(f"- **Gold `true_veracity`:** `{row.get('true_veracity')}`\n\n")
        f.write(f"## Claim\n\n> {claim}\n\n")
        f.write(f"**Reported source:** {src}\n\n")
        f.write("---\n\n## System prompt (codebook + decision rules)\n\n")
        f.write("```text\n" + PROMPT_VARIANTS[args.prompt] + "\n```\n\n")
        if evidence_block is not None:
            f.write("---\n\n## Retrieved evidence\n\n")
            f.write("```text\n" + evidence_block + "\n```\n\n")
        f.write("---\n\n## Assembled user message (verbatim, what the model sees)\n\n")
        if evidence_block is not None:
            user_text = USER_TEMPLATE_RAG.format(claim=claim, source=src,
                                                  evidence=evidence_block)
        else:
            user_text = USER_TEMPLATE.format(claim=claim, source=src)
        f.write("```text\n" + user_text + "\n```\n")
    print(f"Wrote prompt dump → {out_path}")
    sys.exit(0)

if args.dry_run:
    # Audit mode: print the assembled system+user message for the first N rows
    # without touching the LLM API. Lets you eyeball what RAG actually feeds
    # the classifier.
    rows = df.head(args.dry_run).to_dict("records")
    print(f"\n=== DRY RUN: {len(rows)} rows ===")
    print(f"System prompt: {len(PROMPT_VARIANTS[args.prompt])} chars "
          f"({args.prompt})")
    for i, row in enumerate(rows, 1):
        claim = _truncate(row["claim"])
        if _retriever is not None:
            user_text = USER_TEMPLATE_RAG.format(
                claim=claim, source=row.get("source", ""),
                evidence=_build_evidence(claim),
            )
        elif _exa_client is not None:
            user_text = USER_TEMPLATE_RAG.format(
                claim=claim, source=row.get("source", ""),
                evidence=_build_exa_evidence(claim),
            )
        else:
            user_text = USER_TEMPLATE.format(claim=claim, source=row.get("source", ""))
        print(f"\n----- ROW {i} (itemId={row.get('itemId')}) -----")
        print(f"gold true_cfb_label = {row.get('true_cfb_label')}")
        print(f"gold true_veracity  = {row.get('true_veracity')}")
        print(f"\n[user message]\n{user_text}")
    sys.exit(0)

# Resume support: stable per-row key = itemId.
KEY = "itemId"
def row_key(row: dict, idx: int):
    return row[KEY] if KEY in row and row.get(KEY) else f"#{idx}"

output_path.parent.mkdir(parents=True, exist_ok=True)
done_keys: set = set()
if output_path.exists():
    with open(output_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get(KEY):
                done_keys.add(r[KEY])

print(f"Backend: {args.backend} ({base_url})")
print(f"Model:   {args.model}  prompt={args.prompt}")
print(f"Input:   {input_path}  ({len(df)} rows)")
print(f"Output:  {output_path} (resuming, {len(done_keys)} already done)")

todo = [(i, row.to_dict()) for i, row in df.iterrows()
        if row_key(row.to_dict(), i) not in done_keys]
print(f"Todo:    {len(todo)} rows")

write_lock = threading.Lock()
ok = err = 0
with open(output_path, "a", buffering=1) as f, \
     ThreadPoolExecutor(max_workers=args.max_workers) as pool:
    futs = {pool.submit(process, row): i for i, row in todo}
    for fut in tqdm(as_completed(futs), total=len(futs), desc=slug):
        r = fut.result()
        with write_lock:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        if not str(r["response"]).startswith("ERROR:"):
            ok += 1
        else:
            err += 1

print(f"\nDone this run. ok={ok}, errors={err}")
