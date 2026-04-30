# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "tenacity", "tqdm", "pandas", "python-dotenv"]
# ///
"""Run a chat-completion model on a benchmark jsonl and save responses.

Talks to any OpenAI-compatible endpoint. Three backends: vllm (default,
local), openai, openrouter. API key is read from the matching env var.

  uv run infer.py --model C3DS/Windy-Qwen3.5-9B
  uv run infer.py --backend openrouter --model anthropic/claude-opus-4.7
  uv run infer.py --backend openai --model gpt-5 --prompt full
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prompts import full_system_instruction, slim_system_instruction  # noqa: E402

BACKENDS = {
    "vllm":       ("http://localhost:8000/v1",     None),
    "openai":     ("https://api.openai.com/v1",    "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

PROMPT_VARIANTS = {"slim": slim_system_instruction, "full": full_system_instruction}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--backend", choices=list(BACKENDS), default="vllm")
ap.add_argument("--model", required=True)
ap.add_argument("--input", default="data/test/test.jsonl")
ap.add_argument("--output", default=None,
                help="Default: data/results/<split>/<model-slug>[-<prompt>].jsonl")
ap.add_argument("--prompt", choices=list(PROMPT_VARIANTS), default="slim")
ap.add_argument("--max-workers", type=int, default=40)
ap.add_argument("--max-tokens", type=int, default=1500)
ap.add_argument("--limit", type=int, default=None, help="Only process first N rows")
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
suffix = "" if args.prompt == "slim" else f"-{args.prompt}"
split = input_path.stem  # test -> test, val -> val
output_path = Path(args.output) if args.output else ROOT / "data" / "results" / split / f"{slug}{suffix}.jsonl"
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
def query(text: str) -> str:
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"### Text:\n{text}"},
        ],
        temperature=0,
        max_tokens=args.max_tokens,
    )
    return resp.choices[0].message.content


def process(row: dict) -> dict:
    out = dict(row)
    try:
        out["response"] = query(row["content"])
    except Exception as e:
        out["response"] = f"ERROR: {e}"
    return out


df = pd.read_json(input_path, lines=True).reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)

print(f"Backend: {args.backend} ({base_url})")
print(f"Model:   {args.model}  prompt={args.prompt}")
print(f"Input:   {input_path}  ({len(df)} rows)")
print(f"Output:  {output_path}")

results: list[dict | None] = [None] * len(df)
with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
    futs = {pool.submit(process, row.to_dict()): i for i, row in df.iterrows()}
    for fut in tqdm(as_completed(futs), total=len(futs), desc=slug):
        results[futs[fut]] = fut.result()

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

ok = sum(1 for r in results if r and not str(r["response"]).startswith("ERROR:"))
err = len(results) - ok
print(f"\nDone. ok={ok}, errors={err}")
