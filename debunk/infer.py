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
args = ap.parse_args()

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


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=5, min=5, max=60),
       retry=retry_if_exception_type(Exception))
def query(claim: str, source: str):
    """Return the full assistant `message` object (content + annotations)."""
    kwargs = dict(
        model=args.model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": USER_TEMPLATE.format(claim=claim, source=source)},
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
