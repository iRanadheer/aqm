# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "tenacity", "tqdm", "pandas", "python-dotenv", "pydantic>=2", "transformers"]
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
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm


class WindOutput(BaseModel):
    """Structured output for wind opposition classification.

    Frames are N_* codes (e.g. N_1, N_2). Claims are C_*_* codes (e.g. C_1_1).
    Both lists are empty when opposition_detected=False.
    """
    opposition_detected: bool
    frames: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prompts import full_system_instruction, slim_system_instruction, slim_chat_system_instruction  # noqa: E402

BACKENDS = {
    "vllm":       ("http://localhost:8000/v1",     None),
    "openai":     ("https://api.openai.com/v1",    "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

PROMPT_VARIANTS = {"slim": slim_system_instruction, "full": full_system_instruction,
                   "chat": slim_chat_system_instruction}

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
ap.add_argument("--max-input-tokens", type=int, default=None,
                help="Truncate user content to N tokens (using the model's "
                     "own HuggingFace tokenizer). Off by default.")
ap.add_argument("--no-think", dest="think", action="store_false", default=True,
                help="Disable Qwen3 thinking-mode at inference (chat_template_kwargs.enable_thinking=False).")
ap.add_argument("--structured", action="store_true",
                help="Constrain output to the WindOutput Pydantic schema via "
                     "client.beta.chat.completions.parse(). Bypasses YAML parsing.")
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
    kwargs = dict(
        model=args.model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"### Text:\n{text}"},
        ],
        temperature=0,
        max_tokens=args.max_tokens,
    )
    # Always pass enable_thinking explicitly: Qwen templates default it ON when
    # absent, but Gemma 4 defaults OFF (pre-fills an empty thought channel) —
    # leaving it implicit silently evals Gemma without reasoning.
    kwargs.setdefault("extra_body", {})["chat_template_kwargs"] = {"enable_thinking": bool(args.think)}
    if args.structured:
        # parse() returns the Pydantic instance via message.parsed; serialize
        # to JSON so the stored response stays a string for downstream tools.
        completion = client.chat.completions.parse(
            response_format=WindOutput,
            **kwargs,
        )
        msg = completion.choices[0].message
        if msg.parsed is not None:
            return msg.parsed.model_dump_json()
        return msg.refusal or msg.content or ""
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


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
    out = dict(row)
    try:
        out["response"] = query(_truncate(row["content"]))
    except Exception as e:
        out["response"] = f"ERROR: {e}"
    return out


df = pd.read_json(input_path, lines=True).reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)

# Resume support: stable per-row key. Prefer `url`, fall back to row index.
KEY = "url" if "url" in df.columns else None
def row_key(row: dict, idx: int):
    return row[KEY] if KEY and row.get(KEY) else f"#{idx}"

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
            if KEY and r.get(KEY):
                done_keys.add(r[KEY])

print(f"Backend: {args.backend} ({base_url})")
print(f"Model:   {args.model}  prompt={args.prompt}")
print(f"Input:   {input_path}  ({len(df)} rows)")
print(f"Output:  {output_path} (resuming, {len(done_keys)} already done)")

todo = [(i, row.to_dict()) for i, row in df.iterrows()
        if row_key(row.to_dict(), i) not in done_keys]
print(f"Todo:    {len(todo)} rows")

import threading
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
