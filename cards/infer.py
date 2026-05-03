# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "tenacity", "tqdm", "pandas"]
# ///
"""Run a chat-completion model on a CARDS jsonl and save responses.

Three backends: vllm (default, local), openai, openrouter. API key is read
from the matching env var (OPENAI_API_KEY / OPENROUTER_API_KEY); vllm needs none.

  uv run eval.py --model C3DS/CARDS-Wind-Qwen3.6-27B-FP8
  uv run eval.py --backend openrouter --model anthropic/claude-opus-4.7 \\
      --input data/cards_twitter.jsonl
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prompts import slim_system_instruction, slim_system_instruction_norecot, cot_trigger  # noqa: E402

BACKENDS = {
    "vllm":       ("http://localhost:8000/v1",        None),
    "openai":     ("https://api.openai.com/v1",       "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1",    "OPENROUTER_API_KEY"),
}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--backend", choices=list(BACKENDS), default="vllm")
ap.add_argument("--model", required=True)
ap.add_argument("--input", default="data/cards_test.jsonl")
ap.add_argument("--output", default=None,
                help="Default: data/results/<split>/<model-slug>.jsonl")
ap.add_argument("--max-workers", type=int, default=40)
ap.add_argument("--max-tokens", type=int, default=4000)
ap.add_argument("--limit", type=int, default=None, help="Only process first N rows")
ap.add_argument("--no-recot", dest="recot", action="store_false", default=True,
                help="Use the no-RECoT system prompt. Default thinking-mode is OFF.")
ap.add_argument("--thinking", dest="thinking", action="store_true", default=None,
                help="Force enable_thinking=True at inference (overrides --no-recot default).")
ap.add_argument("--no-thinking", dest="thinking", action="store_false",
                help="Force enable_thinking=False at inference (overrides --recot default).")
args = ap.parse_args()
# If --thinking/--no-thinking not passed, default to: True for RECoT, False for no-RECoT.
if args.thinking is None:
    args.thinking = args.recot

base_url, key_env = BACKENDS[args.backend]
if key_env:
    api_key = os.environ.get(key_env)
    if not api_key:
        sys.exit(f"{key_env} not set")
else:
    api_key = "dummy"

input_path = Path(args.input) if Path(args.input).is_absolute() else ROOT / args.input
if not input_path.exists():
    sys.exit(f"Input not found: {input_path}")

slug = re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")
split = input_path.stem.replace("cards_", "")  # cards_test -> test
output_path = Path(args.output) if args.output else ROOT / "data" / "results" / split / f"{slug}.jsonl"
if not output_path.is_absolute():
    output_path = ROOT / output_path

client = OpenAI(base_url=base_url, api_key=api_key)
# RECoT system prompt instructs <think>...</think>+YAML; no-RECoT variant
# instructs YAML-only (no thinking). Pick to match the inference setting.
_system_text = slim_system_instruction if args.recot else slim_system_instruction_norecot
# Ephemeral caching for all backends — providers that don't honor it ignore the field.
system_content = [{
    "type": "text",
    "text": _system_text,
    "cache_control": {"type": "ephemeral"},
}]


def _user_content(text: str) -> str:
    # cot_trigger dropped for all variants — uniform user prompt across
    # RECoT-FT, no-RECoT-FT, and base. RECoT-FT still gets thinking-mode
    # via the chat template (enable_thinking=True) and its training-matched
    # system prompt, just without the explicit "step by step" appendix.
    return f"### Text:\n{text}"


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=5, min=5, max=60),
       retry=retry_if_exception_type(Exception))
def query(text: str) -> str:
    kwargs = dict(
        model=args.model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": _user_content(text)},
        ],
        temperature=0,
        max_tokens=args.max_tokens,
    )
    # No-RECoT models were trained without <think>...</think>. Qwen3's chat
    # template auto-injects <think>\n at the start of the assistant turn
    # unless we disable thinking mode — without this, the model fills the
    # injected <think> with the system prompt's example template instead of
    # the YAML it actually learned to emit.
    # Pass enable_thinking to the chat template (vLLM extra_body). Independent
    # of --no-recot now: RECoT models can be evaluated no-think, no-RECoT models
    # can be evaluated with-think (matches their training-time chat template).
    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": args.thinking}}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def process(row: dict) -> dict:
    out = dict(row)
    try:
        out["response"] = query(row["text"])
    except Exception as e:
        out["response"] = f"ERROR: {e}"
    return out


df = pd.read_json(input_path, lines=True).reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)

print(f"Backend: {args.backend} ({base_url})")
print(f"Model:   {args.model}")
print(f"RECoT:   {args.recot}  (system prompt={'recot' if args.recot else 'norecot'}, "
      f"trigger=off, thinking={'on' if args.thinking else 'off'})")
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
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

ok = sum(1 for r in results if r and not str(r["response"]).startswith("ERROR:"))
err = len(results) - ok
print(f"\nDone. ok={ok}, errors={err}")
