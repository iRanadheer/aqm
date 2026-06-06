"""Generate Reverse Engineered Chain-of-Thought (RECoT) training data.

Calls a teacher LLM (e.g. Claude Opus) on (text, true_labels) pairs and asks
it to produce expert-level reasoning that arrives at the given labels. Output
is JSONL ready to feed prepare_splits.py.

Usage:
    python teacher.py --input data/training.csv \\
        --model anthropic/claude-opus-4-20250514 \\
        --output data/training_recot_opus.jsonl

    # Chat variant (claim → nested categories + reasons). Same rows, same
    # trigger; only the system prompt differs. Responses are validated
    # against gold before writing; failures are skipped and picked up on
    # rerun via resume.
    python teacher.py --chat --input data/training.csv \\
        --model anthropic/claude-opus-4-8 \\
        --output data/training_recot_opus_chat.jsonl

Input is the training pool (text, true_claims). Output adds the teacher's
<think>+YAML reasoning per row and feeds prepare_splits.py. Never run on
eval splits.
"""

import argparse
import ast
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
import pandas as pd
import tenacity
import yaml
from dotenv import load_dotenv
from litellm.exceptions import RateLimitError
from tqdm import tqdm

from prompts import chat_system_instruction, system_instruction, recot_trigger

load_dotenv()


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(10),
    retry=tenacity.retry_if_exception_type(RateLimitError),
    reraise=False,
)
def supports_temperature(model: str) -> bool:
    """Opus 4.7+ removed sampling params — sending `temperature` returns a 400.

    Older models (Opus 4.6 and earlier, Sonnet, etc.) still accept it; we keep
    temperature=0 there to match how the published data was generated.
    Handles both `4-7` and `4.7` style ids (litellm/OpenRouter slugs vary).
    Date-suffixed ids like `claude-opus-4-20250514` (Opus 4.0) keep it.
    """
    return re.search(r"opus-4[.-](?:[7-9]|[1-9]\d)(?!\d)", model) is None


def call_teacher(model: str, text: str, true_labels, max_tokens: int, chat: bool = False) -> dict:
    system_text = chat_system_instruction if chat else system_instruction
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": f"### Text:\n{text}\n\n### True Labels:\n{true_labels}\n\n{recot_trigger}",
        },
    ]
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        timeout=120,
    )
    if supports_temperature(model):
        kwargs["temperature"] = 0
    response = litellm.completion(**kwargs)
    return {
        "response": response.choices[0].message.content,
        "usage": response.usage.model_dump(),
    }


def validate_chat_response(response: str, true_labels) -> str | None:
    """Validate a chat-format teacher response against gold. None = OK.

    Checks: a parseable YAML block after </think>, at least one claim entry,
    and the union of emitted codes across all entries equal to the gold
    label set. Multi-claim entries are allowed — the model may legitimately
    segment a multi-claim text; gold is text-level, so the check is
    union-based (same standard as the wind chat validator demos).
    """
    after = response.split("</think>")[-1] if "</think>" in response else response
    m = re.search(r"```yaml\s*\n(.*?)```", after, re.DOTALL)
    if not m:
        return "no YAML block after </think>"
    try:
        doc = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        return f"YAML parse error: {e}"
    claims = doc.get("claims") if isinstance(doc, dict) else None
    if not isinstance(claims, list) or len(claims) < 1:
        return "expected at least 1 claim entry"
    # Codes via regex on the raw YAML text — yaml.safe_load would parse the
    # unquoted code 4_1_1 as the integer 411 (YAML 1.1 digit separators).
    codes = set(re.findall(r"code:\s*['\"]?([\d_]+)", m.group(1)))
    # Gold may arrive as a list or as its string repr (jsonl input skips
    # parse_labels), with stray trailing commas — extract code tokens the
    # same way prepare_splits.parse_true_claims does.
    gold = set(re.findall(r"[\d_]+", str(true_labels)))
    if codes != gold:
        return f"codes {sorted(codes)} != gold {sorted(gold)}"
    return None


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
    ap.add_argument("--chat", action="store_true",
                    help="Generate chat-format responses (chat_system_instruction) with gold validation.")
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
            pool.submit(call_teacher, args.model, r[args.text_col], r[args.labels_col], args.max_tokens, args.chat): r
            for r in remaining
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="RECoT"):
            row = futures[fut]
            try:
                result = fut.result()
                if args.chat:
                    err = validate_chat_response(result["response"], row[args.labels_col])
                    if err:
                        tqdm.write(f"Validation failed (skipped, will retry on rerun): {err}")
                        continue
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
