# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "tenacity", "tqdm", "pandas", "python-dotenv"]
# ///
"""Generate RECoT training traces with a teacher LLM.

Calls a teacher on (text, true_label) pairs and asks for expert reasoning that
arrives at the given label — the trace the student is trained to imitate. The
system prompt is the ordinary `deconstruct-revised` prompt, unchanged; only the
user turn carries the gold label and the trigger, so the student learns exactly
the prompt it will meet at inference.

    uv run teacher.py --model anthropic/claude-opus-4.7 --limit 50
    uv run teacher.py --model anthropic/claude-opus-4.7

Reads  data/train/train.jsonl   (1,796 rows — NEVER val or test)
Writes data/train/recot-<model-slug>.jsonl, appending, resume-safe by itemId.

Traces whose FINAL label does not match gold are NOT written. They are counted,
reported, and retried on the next invocation, matching cards/teacher.py, where
an unvalidated trace teaches the student to reach a label its own reasoning
does not support.

Adapted from cards/teacher.py, with two deliberate departures: the OpenAI-
compatible backends of infer.py rather than LiteLLM (same three providers, same
.env loading, same resume convention), and validation by label match rather
than by YAML set equality, this task being single-label.
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
    FALLACY_LABELS, RECOT_TRIGGER, TEACHER_TEMPLATE, extract_raw_label, get_prompt,
)

BACKENDS = {
    "vllm":       ("http://localhost:8000/v1",     None),
    "openai":     ("https://api.openai.com/v1",    "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--backend", choices=list(BACKENDS), default="openrouter")
ap.add_argument("--base-url", default=None)
ap.add_argument("--model", required=True)
ap.add_argument("--prompt", default="deconstruct-revised",
                help="System prompt the student will also use at inference. "
                     "Changing this invalidates every trace already generated.")
ap.add_argument("--input", default="data/train/train.jsonl")
ap.add_argument("--output", default=None,
                help="Default: data/train/recot-<model-slug>.jsonl")
ap.add_argument("--max-workers", type=int, default=10)
ap.add_argument("--max-tokens", type=int, default=4000)
ap.add_argument("--limit", type=int, default=None,
                help="Only process the first N rows — use for a bake-off "
                     "between candidate teachers before committing to all 1,796.")
args = ap.parse_args()

load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")
load_dotenv(ROOT.parent.parent / ".env")
base_url, key_env = BACKENDS[args.backend]
base_url = args.base_url or os.environ.get("VLLM_BASE_URL") or base_url
if key_env:
    api_key = os.environ.get(key_env)
    if not api_key:
        sys.exit(f"{key_env} not set")
else:
    api_key = "dummy"

input_path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
if not input_path.exists():
    sys.exit(f"Input not found: {input_path}")
if input_path.stem in {"val", "test"}:
    sys.exit(f"Refusing to generate traces from {input_path.stem} — training split only")

slug = re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")
output_path = (Path(args.output) if args.output
               else ROOT / "data" / "train" / f"recot-{slug}.jsonl")
if not output_path.is_absolute():
    output_path = ROOT / output_path

system_prompt = get_prompt(args.prompt)
client = OpenAI(base_url=base_url, api_key=api_key)
system_content = [{"type": "text", "text": system_prompt,
                   "cache_control": {"type": "ephemeral"}}]


def supports_temperature(model: str) -> bool:
    """Opus 4.7+ removed sampling params — sending `temperature` returns a 400.

    Ported from cards/teacher.py. Handles both `4-7` and `4.7` style ids;
    date-suffixed ids like `claude-opus-4-20250514` (Opus 4.0) keep it.
    """
    return re.search(r"opus-4[.-](?:[7-9]|[1-9]\d)(?!\d)", model) is None


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=5, min=5, max=60),
       retry=retry_if_exception_type(Exception))
def call_teacher(text: str, label: str):
    kwargs = dict(
        model=args.model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": TEACHER_TEMPLATE.format(
                text=text, label=label, trigger=RECOT_TRIGGER)},
        ],
        max_tokens=args.max_tokens,
    )
    if supports_temperature(args.model):
        kwargs["temperature"] = 0
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = (getattr(msg, "reasoning", None)
                 or (msg.model_extra or {}).get("reasoning") or "")
    full = (f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"
            if reasoning and "<think>" not in content else content)
    return full, resp.usage.model_dump() if resp.usage else {}


def process(row: dict) -> dict:
    gold = row["true_fallacy"]
    try:
        response, usage = call_teacher(row["text"], gold)
    except Exception as e:
        return {"itemId": row["itemId"], "ok": False, "why": f"ERROR: {e}"}
    pred = extract_raw_label(response)
    if pred != gold:
        return {"itemId": row["itemId"], "ok": False,
                "why": f"trace reached {pred!r}, gold {gold!r}"}
    return {"itemId": row["itemId"], "ok": True, "text": row["text"],
            "true_fallacy": gold, "model": args.model,
            "prompt_variant": args.prompt, "response": response, "usage": usage}


df = pd.read_json(input_path, lines=True).reset_index(drop=True)
if args.limit:
    df = df.head(args.limit)

bad = set(df["true_fallacy"]) - set(FALLACY_LABELS)
if bad:
    sys.exit(f"Input has labels outside the codebook: {sorted(bad)}")

output_path.parent.mkdir(parents=True, exist_ok=True)
done: set = set()
if output_path.exists():
    for ln in output_path.read_text().splitlines():
        if ln.strip():
            try:
                done.add(json.loads(ln)["itemId"])
            except json.JSONDecodeError:
                continue

todo = [r.to_dict() for _, r in df.iterrows() if r["itemId"] not in done]
print(f"Teacher: {args.model}  prompt={args.prompt}")
print(f"Input:   {input_path} ({len(df)} rows)")
print(f"Output:  {output_path} ({len(done)} already validated)")
print(f"Todo:    {len(todo)} rows")
if not todo:
    sys.exit(0)

lock = threading.Lock()
kept = rejected = errored = 0
reasons: list[str] = []
with open(output_path, "a", buffering=1) as f, \
     ThreadPoolExecutor(max_workers=args.max_workers) as pool:
    futs = [pool.submit(process, r) for r in todo]
    for fut in tqdm(as_completed(futs), total=len(futs), desc=f"recot-{slug}"):
        r = fut.result()
        if r["ok"]:
            with lock:
                f.write(json.dumps({k: v for k, v in r.items() if k != "ok"},
                                   ensure_ascii=False, default=str) + "\n")
            kept += 1
        elif r["why"].startswith("ERROR:"):
            errored += 1
        else:
            rejected += 1
            if len(reasons) < 10:
                reasons.append(f"  {r['itemId']}: {r['why']}")

total = kept + rejected + errored
print(f"\nKept {kept}/{total} ({100 * kept / total:.0f}%)  "
      f"rejected {rejected}  errored {errored}")
if reasons:
    print("Sample rejections (retried on rerun):")
    print("\n".join(reasons))
print(f"-> {output_path}")
