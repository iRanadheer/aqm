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
    FALLACY_LABELS, USER_TEMPLATE, VARIANT_NAMES, extract_raw_label, get_prompt,
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
ap.add_argument("--structured", action="store_true",
                help="Constrain the answer to a JSON schema of {reasoning, "
                     "fallacy} via the standard OpenAI response_format. For "
                     "BASE models, which cannot produce the output contract "
                     "unaided — they invent <label> tags, echo the prompt's "
                     "placeholder, or never stop reasoning — so strict parsing "
                     "measures format failure rather than classification. The "
                     "schema keeps reasoning free-form AND guarantees an "
                     "in-vocabulary label, which closes both fairness "
                     "objections at once: the baseline is neither denied "
                     "reasoning nor penalised for formatting. Being generous to "
                     "the baseline shrinks the fine-tune's margin, not grows it.")
ap.add_argument("--reasoning-max-chars", type=int, default=1200,
                help="maxLength on the schema's reasoning field (--structured "
                     "only). 0 removes the cap. The cap exists to stop weak "
                     "models rambling until they hit max_tokens without ever "
                     "emitting the label — one 4B item ran to 20,972 chars "
                     "unterminated. Capable models do not need it and it can "
                     "truncate genuine reasoning, so pass 0 for 9B and up.")
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
# The decoding condition goes in the filename. Without it a --structured run
# resumes the strict run's file by itemId and silently blends two different
# measurement protocols into one result set.
suffix = f"{args.prompt}-structured" if args.structured else args.prompt
output_path = (
    Path(args.output) if args.output
    else ROOT / "data" / "results" / split / f"{slug}-{suffix}.jsonl"
)
if not output_path.is_absolute():
    output_path = ROOT / output_path

# Schema for --structured. Mirrors a pydantic model:
#     class FallacyAnswer(BaseModel):
#         reasoning: str
#         fallacy: Literal[<the twelve labels>]
# `reasoning` first so the model reasons BEFORE committing — JSON fields are
# generated in order, so putting the label first would force it to answer and
# then rationalise.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        # maxLength is load-bearing for weak models, not cosmetic: without it
        # they ramble inside `reasoning` until max_tokens, leaving the JSON
        # unterminated — one 4B item produced 20,972 chars and never reached the
        # `fallacy` key. The grammar backend enforces the cap (verified). But it
        # also truncates genuine reasoning, so --reasoning-max-chars 0 removes
        # it for models that do not need the brake.
        "reasoning": ({"type": "string", "maxLength": args.reasoning_max_chars}
                      if args.reasoning_max_chars > 0 else {"type": "string"}),
        "fallacy": {"type": "string", "enum": list(FALLACY_LABELS)},
    },
    "required": ["reasoning", "fallacy"],
    "additionalProperties": False,
}

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
    # Send enable_thinking EXPLICITLY on vLLM rather than trusting the chat
    # template's default. Qwen3.5's template defaults it off, which pre-closes
    # the <think> block: the FT 0.8B then emitted only the YAML answer with no
    # deconstruction at all (0/457 traces, macro F1 0.238) despite having been
    # trained on full traces. Passing True restores them.
    if args.backend == "vllm":
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": args.think}}
    elif not args.think:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if args.structured:
        if args.backend == "vllm":
            # disable_any_whitespace is the point of taking vLLM's extra_body
            # over the portable response_format here. A JSON grammar permits
            # unlimited whitespace after `{`, and Qwen3.5-0.8B exploits it: it
            # emitted 4,000 characters of spaces and hit the token cap on 36%
            # of val items, never reaching the `fallacy` key. Dropping
            # concurrency 64 -> 16 barely moved that (36% -> 31%), so it is the
            # grammar, not batching. Portability is no loss: --structured is
            # only ever used against base models on our own vLLM.
            kwargs["extra_body"]["structured_outputs"] = {
                "json": ANSWER_SCHEMA,
                "disable_any_whitespace": True,
            }
        else:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "fallacy_answer", "schema": ANSWER_SCHEMA},
            }
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
