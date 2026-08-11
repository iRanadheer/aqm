# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "tenacity", "tqdm", "pandas", "python-dotenv"]
# ///
"""Run a chat model on the FLICC fallacy-detection benchmark.

Adapted from debunk/infer.py: same three OpenAI-compatible backends, same slug
and resume conventions. The RAG and Exa evidence paths are deliberately absent —
the baseline is a text-in/label-out classifier that saw no evidence, so this
study runs without retrieval (see the design spec).

  uv run infer.py --backend openrouter --model anthropic/claude-opus-4.7 \\
      --prompt deconstruct --input data/test/val.jsonl
  uv run infer.py --backend openai --model gpt-5.5 --prompt bare
  uv run infer.py --model Qwen/Qwen3.5-0.8B --prompt deconstruct

Writes data/results/<split>/<model-slug>-<rung>.jsonl, one JSON object per row,
appending so an interrupted run resumes on the next invocation.
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
from prompts import (  # noqa: E402
    USER_TEMPLATE, VARIANT_NAMES, extract_raw_label, get_prompt,
)

BACKENDS = {
    "vllm":       ("http://localhost:8000/v1",     None),
    "openai":     ("https://api.openai.com/v1",    "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--backend", choices=list(BACKENDS), default="vllm")
ap.add_argument("--base-url", default=None,
                help="Override the backend's URL — e.g. a vLLM server on "
                     "another host: --base-url http://10.0.0.5:8000/v1. Also "
                     "settable as VLLM_BASE_URL in .env.")
ap.add_argument("--model", required=True)
ap.add_argument("--input", default="data/test/val.jsonl",
                help="Default is VAL. Prompt development happens on val; test "
                     "is run once per (model, rung) when the prompts are settled.")
ap.add_argument("--output", default=None,
                help="Default: data/results/<split>/<model-slug>-<rung>.jsonl")
ap.add_argument("--prompt", choices=VARIANT_NAMES, default="deconstruct-revised")
ap.add_argument("--max-workers", type=int, default=20)
ap.add_argument("--max-tokens", type=int, default=4000,
                help="High enough for reasoning models that burn hidden "
                     "reasoning tokens before emitting visible content.")
ap.add_argument("--limit", type=int, default=None, help="Only process first N rows")
ap.add_argument("--no-think", dest="think", action="store_false", default=True,
                help="Disable Qwen3 thinking-mode "
                     "(chat_template_kwargs.enable_thinking=False).")
ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                help="Print the assembled system + user messages for the first "
                     "N rows and exit, without calling the model.")
args = ap.parse_args()

# Keys live in the monorepo root .env; the two nearer paths are honoured first
# so a chapter- or study-local override wins.
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")
load_dotenv(ROOT.parent.parent / ".env")
base_url, key_env = BACKENDS[args.backend]
base_url = args.base_url or os.environ.get("VLLM_BASE_URL") or base_url
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
    sys.exit(f"Input not found: {input_path} — run prepare_splits.py first")

slug = re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")
split = input_path.stem  # val.jsonl -> val, test.jsonl -> test
output_path = (
    Path(args.output) if args.output
    else ROOT / "data" / "results" / split / f"{slug}-{args.prompt}.jsonl"
)
if not output_path.is_absolute():
    output_path = ROOT / output_path

system_prompt = get_prompt(args.prompt)
client = OpenAI(base_url=base_url, api_key=api_key)
# Ephemeral caching — providers that don't honour it ignore the field. The
# system prompt is identical across every row, so this is most of the spend.
system_content = [{
    "type": "text",
    "text": system_prompt,
    "cache_control": {"type": "ephemeral"},
}]


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=5, min=5, max=60),
       retry=retry_if_exception_type(Exception))
def query(text: str):
    """Return the assistant message object.

    `cards_claim` and `source` are deliberately NOT shown to the model —
    the CARDS code would leak taxonomy information the baseline classifier
    never had, and the source corpus name would leak provenance.
    """
    kwargs = dict(
        model=args.model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user",   "content": USER_TEMPLATE.format(text=text)},
        ],
        temperature=0,
        max_tokens=args.max_tokens,
    )
    if not args.think:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message


def process(row: dict) -> dict:
    true_fields = {k: v for k, v in row.items() if k.startswith("true_")}
    out = {k: v for k, v in row.items() if not k.startswith("true_")}
    out["prompt_variant"] = args.prompt

    pred_label: str | None = None
    try:
        msg = query(row["text"])
        # Some reasoning models expose chain-of-thought on a non-standard
        # `reasoning` field rather than in `content`. Fold it into the response
        # as a <think> block so one field holds the whole output.
        content = msg.content or ""
        reasoning = (getattr(msg, "reasoning", None)
                     or (msg.model_extra or {}).get("reasoning")
                     or "")
        full = (f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"
                if reasoning and "<think>" not in content else content)
        out["response"] = full
        pred_label = extract_raw_label(full)
    except Exception as e:
        out["response"] = f"ERROR: {e}"

    out.update(true_fields)
    out["pred_label"] = pred_label
    return out


df = pd.read_json(input_path, lines=True).reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)

if args.dry_run:
    rows = df.head(args.dry_run).to_dict("records")
    print(f"\n=== DRY RUN: {len(rows)} rows ===")
    print(f"Rung: {args.prompt}  ({len(system_prompt)} chars)")
    print(f"\n[system message]\n{system_prompt}")
    for i, row in enumerate(rows, 1):
        print(f"\n----- ROW {i} (itemId={row.get('itemId')}) -----")
        print(f"gold true_fallacy = {row.get('true_fallacy')}")
        print(f"\n[user message]\n{USER_TEMPLATE.format(text=row['text'])}")
    sys.exit(0)

KEY = "itemId"
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
print(f"Model:   {args.model}  rung={args.prompt}")
print(f"Input:   {input_path}  ({len(df)} rows)")
print(f"Output:  {output_path} (resuming, {len(done_keys)} already done)")

todo = [row.to_dict() for _, row in df.iterrows()
        if row.to_dict()[KEY] not in done_keys]
print(f"Todo:    {len(todo)} rows")

write_lock = threading.Lock()
ok = err = 0
with open(output_path, "a", buffering=1) as f, \
     ThreadPoolExecutor(max_workers=args.max_workers) as pool:
    futs = [pool.submit(process, row) for row in todo]
    for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{slug}-{args.prompt}"):
        r = fut.result()
        with write_lock:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        if str(r["response"]).startswith("ERROR:"):
            err += 1
        else:
            ok += 1

print(f"\nDone: {ok} ok, {err} errors -> {output_path}")
