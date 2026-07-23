# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "tenacity", "tqdm", "numpy", "python-dotenv"]
# ///
"""Run a chat model on cards/wind under static or dynamic few-shot.

No fine-tuning, and no zero-shot here — zero-shot == the existing base runs
(same slim RECoT prompt + thinking); the ONLY delta is the added text+label
demo turns. Demos are drawn from the train+train_eval corpus; test/val are
untouched. Output lands in the schema the sibling scorer (score.py) reads, so
metrics are identical to the base/FT table.

  # static few-shot, k=10
  uv run infer.py --task wind --model qwen/qwen3.5-9b --regime static --k 10
  # dynamic few-shot, k=10 (needs build_corpus.py first)
  uv run infer.py --task cards --model qwen/qwen3.5-9b --regime dynamic --k 10
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))
from embedder import DEFAULT_EMBED_MODEL, DEFAULT_EMBED_URL  # noqa: E402
from retrieval import build_selector  # noqa: E402
from tasks import get_task  # noqa: E402

BACKENDS = {
    "vllm":       ("http://localhost:8000/v1",     None),
    "openai":     ("https://api.openai.com/v1",    "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--task", required=True, choices=["cards", "wind"])
ap.add_argument("--regime", required=True, choices=["static", "dynamic"])
ap.add_argument("--backend", choices=list(BACKENDS), default="openrouter")
ap.add_argument("--model", required=True)
ap.add_argument("--split", default="test", choices=["test", "val"])
ap.add_argument("--k", type=int, default=10, help="Number of demos.")
ap.add_argument("--no-think", dest="think", action="store_false", default=True,
                help="Disable thinking-mode (default ON, matching the base zero-shot runs).")
ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Retrieval embedder (dynamic only).")
ap.add_argument("--embed-url", default=DEFAULT_EMBED_URL, help="Embedding endpoint (dynamic only).")
ap.add_argument("--output", default=None, help="Default: data/results/<task>/<split>/<slug>.jsonl")
ap.add_argument("--max-workers", type=int, default=40)
ap.add_argument("--max-tokens", type=int, default=4000)
ap.add_argument("--limit", type=int, default=None, help="Only process first N rows (smoke test).")
args = ap.parse_args()

base_url, key_env = BACKENDS[args.backend]
api_key = os.environ.get(key_env) if key_env else "dummy"
if key_env and not api_key:
    sys.exit(f"{key_env} not set")

task = get_task(args.task)
corpus = task.load_corpus()
selector = build_selector(
    args.regime, corpus, args.k, task,
    cache_dir=ROOT / "data" / "embeddings",
    embed_model=args.embed_model, embed_url=args.embed_url,
)

eval_examples = task.load_split(args.split)
if args.limit:
    eval_examples = eval_examples[: args.limit]

# Output slug: <model>-<regime>-k<k>. Mirrors sibling result naming.
model_slug = re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")
slug = f"{model_slug}-{args.regime}-k{args.k}"
output_path = Path(args.output) if args.output else ROOT / "data" / "results" / task.name / args.split / f"{slug}.jsonl"
if not output_path.is_absolute():
    output_path = ROOT / output_path

client = OpenAI(base_url=base_url, api_key=api_key)
# Ephemeral cache the (large) system prompt; backends that ignore it are fine.
system_content = [{"type": "text", "text": task.system_prompt, "cache_control": {"type": "ephemeral"}}]


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=5, min=5, max=60),
       retry=retry_if_exception_type(Exception))
def query(text: str) -> str:
    messages = [{"role": "system", "content": system_content}]
    if selector is not None:
        for demo in selector.select(text):
            messages.extend(task.demo_turns(demo))
    messages.append(task.query_turn(text))
    resp = client.chat.completions.create(
        model=args.model, messages=messages, temperature=0, max_tokens=args.max_tokens,
        # Explicit enable_thinking (Qwen defaults ON, Gemma-4 defaults OFF) so
        # the model reasons on the query — identical to the base zero-shot runs.
        extra_body={"chat_template_kwargs": {"enable_thinking": bool(args.think)}},
    )
    return resp.choices[0].message.content


def process(ex) -> dict:
    try:
        response = query(ex.text)
    except Exception as e:
        response = f"ERROR: {e}"
    return task.result_row(ex.raw, response)


print(f"Task:    {task.name}  ({args.split}, {len(eval_examples)} rows)")
print(f"Regime:  {args.regime}  (k={args.k}, corpus={len(corpus)}, think={args.think})")
print(f"Backend: {args.backend} ({base_url})")
print(f"Model:   {args.model}")
if args.regime == "dynamic":
    print(f"Embedder:{args.embed_model} @ {args.embed_url}")
print(f"Output:  {output_path}")

results: list[dict | None] = [None] * len(eval_examples)
with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
    futs = {pool.submit(process, ex): i for i, ex in enumerate(eval_examples)}
    for fut in tqdm(as_completed(futs), total=len(futs), desc=slug):
        results[futs[fut]] = fut.result()

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

ok = sum(1 for r in results if r and not str(r["response"]).startswith("ERROR:"))
print(f"\nDone. ok={ok}, errors={len(results) - ok}")
