"""Generate Reverse Engineered Chain-of-Thought (RECoT) training data.

Calls a teacher LLM (e.g. Claude Opus) on (text, true_labels) pairs and asks
it to produce expert-level reasoning that arrives at the given labels. Output
is JSONL ready to feed prepare_splits.py.

Usage:
    python teacher.py --input data/training.csv \\
        --model anthropic/claude-opus-4-20250514 \\
        --output data/training_recot_opus.jsonl

Input is the training pool (text, true_claims). Output adds the teacher's
<think>+YAML reasoning per row and feeds prepare_splits.py. Never run on
eval splits.
"""

import argparse
import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
import pandas as pd
import tenacity
from dotenv import load_dotenv
from litellm.exceptions import RateLimitError
from tqdm import tqdm

from prompts import system_instruction, recot_trigger

load_dotenv()


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(10),
    retry=tenacity.retry_if_exception_type(RateLimitError),
    reraise=False,
)
def call_teacher(model: str, text: str, true_labels, max_tokens: int) -> dict:
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_instruction,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": f"### Text:\n{text}\n\n### True Labels:\n{true_labels}\n\n{recot_trigger}",
        },
    ]
    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        timeout=120,
    )
    return {
        "response": response.choices[0].message.content,
        "usage": response.usage.model_dump(),
    }


def parse_labels(val):
    if not isinstance(val, str):
        return val if isinstance(val, list) else []
    val = val.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    try:
        return ast.literal_eval(val)
    except Exception:
        return [val]


def load_data(path: str, labels_col: str) -> list[dict]:
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
        df[labels_col] = df[labels_col].apply(parse_labels)
    return df.to_dict("records")


def main():
    ap = argparse.ArgumentParser(description="Generate RECoT training data with a teacher LLM")
    ap.add_argument("--input", required=True, help="Input CSV or JSONL")
    ap.add_argument("--output", required=True, help="Output JSONL (appends; resume-safe)")
    ap.add_argument("--model", required=True, help="LiteLLM model id, e.g. anthropic/claude-opus-4-20250514")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--max", type=int, help="Limit number of input examples")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--labels-col", default="true_claims")
    args = ap.parse_args()

    data = load_data(args.input, args.labels_col)
    if args.max:
        data = data[: args.max]

    already_done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["text"][:200])
                except Exception:
                    pass
    if already_done:
        print(f"Resuming — skipping {len(already_done)} already processed")

    remaining = [r for r in data if r[args.text_col][:200] not in already_done]
    if not remaining:
        print("Nothing to do.")
        return

    print(f"Processing {len(remaining)} examples with {args.model} -> {args.output}")
    with open(args.output, "a") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(call_teacher, args.model, r[args.text_col], r[args.labels_col], args.max_tokens): r
            for r in remaining
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="RECoT"):
            row = futures[fut]
            try:
                result = fut.result()
                out = {
                    "text": row[args.text_col],
                    "true_claims": row[args.labels_col],
                    "model": args.model,
                    "response": result["response"],
                    "usage": result["usage"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
