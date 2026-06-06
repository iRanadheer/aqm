"""Generate Reverse-Engineered Chain-of-Thought (RECoT) SFT training data.

For each input labels file, sends every example through a teacher model
together with its gold labels, asks the teacher to produce expert
reasoning (using the reasoning chain from the system prompt) that arrives
at exactly those labels, and writes the result directly in SFT messages
format ready for fine-tuning.

Input/output mapping:
  data/train/train_labels.jsonl        → data/train/train.jsonl
  data/train/train_eval_labels.jsonl   → data/train/train_eval.jsonl

With --chat (chat-format SFT data, same rows/partition, validated against gold):
  data/train/train_labels.jsonl        → data/train/train_chat.jsonl
  data/train/train_eval_labels.jsonl   → data/train/train_eval_chat.jsonl

Each output row (OpenAI SFT chat format):
    {"messages": [
        {"role": "system", "content": <system prompt>},
        {"role": "user",   "content": "### Text:\\n<content>"},
        {"role": "assistant", "content": <teacher response>}
    ]}

Resumption: rows whose input content is already present in the output
file (as the trailing user message) are skipped.

Requires OPENROUTER_API_KEY in env or repo .env.

Usage:
    python teacher.py
    python teacher.py --model anthropic/claude-sonnet-4.6
    python teacher.py --concurrency 20 --max 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from prompts import (  # noqa: E402
    full_chat_system_instruction,
    full_system_instruction,
    slim_chat_system_instruction,
    slim_system_instruction,
)
from clean_recot import clean_file as clean_recot  # noqa: E402

DEFAULT_MODEL = "anthropic/claude-opus-4.7"
DEFAULT_INPUTS = [
    "data/train/train_labels.jsonl",
    "data/train/train_eval_labels.jsonl",
]

RECOT_TRIGGER = """The true labels above are the correct classification. Your task: produce an expert reasoning chain that arrives at EXACTLY these labels and nothing else.

Rules:
- Write as a confident expert who has never seen the true labels. Do not reference them, quote them back, or hint that they were provided.
- Follow every step of the reasoning chain from the system prompt: CONTEXT → DETECTION → FRAMES → CLAIMS (shortlist → adjudicate → granularity → force-fit check) → SPECIAL CONSIDERATIONS.
- In each shortlist step, surface all plausible candidates including the correct ones; in each adjudicate step, eliminate the wrong ones decisively with a short reason.
- Single pass. No second-guessing, no repetition.
- Final YAML must exactly match the true labels (same opposition_detected, same frames set, same claims set) and be the only content after </think>."""

USER_TEMPLATE = "### Text:\n{content}"


# ---------------------------------------------------------------------------
# Client + LLM call
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set (env or repo .env).")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def supports_temperature(model: str) -> bool:
    """Opus 4.7+ removed sampling params — sending `temperature` returns a 400.

    Older models (Opus 4.6 and earlier, Sonnet, etc.) still accept it; we keep
    temperature=0 there to match how the published data was generated.
    Handles both `4-7` and `4.7` style ids (OpenRouter slugs vary).
    Date-suffixed ids like `claude-opus-4-20250514` (Opus 4.0) keep it.
    """
    return re.search(r"opus-4[.-](?:[7-9]|[1-9]\d)(?!\d)", model) is None


def make_query(client: OpenAI, model: str, max_tokens: int, chat: bool = False):
    # cache_control on the system prompt → ~10x cheaper after warmup.
    system_content = [{
        "type": "text",
        "text": full_chat_system_instruction if chat else full_system_instruction,
        "cache_control": {"type": "ephemeral"},
    }]
    sampling = {"temperature": 0} if supports_temperature(model) else {}

    def _call(user_text: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_text},
            ],
            max_tokens=max_tokens,
            **sampling,
        )
        return resp.choices[0].message.content

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(Exception),
    )
    def call(user_text: str) -> str:
        return _call(user_text)

    return call


# ---------------------------------------------------------------------------
# Messages construction
# ---------------------------------------------------------------------------

def build_teacher_user_message(row: dict) -> str:
    """User message sent to the teacher — includes gold labels + trigger."""
    opposition = bool(row["true_opposition_detected"])
    frames = list(row.get("true_frames") or [])
    claims = list(row.get("true_claims") or [])
    labels_block = (
        f"opposition_detected: {'true' if opposition else 'false'}\n"
        f"frames: {frames}\n"
        f"claims: {claims}"
    )
    return (
        f"### Text:\n{row['content']}\n\n"
        f"### True Labels:\n{labels_block}\n\n"
        f"{RECOT_TRIGGER}"
    )


def build_sft_record(row: dict, assistant_response: str, chat: bool = False) -> dict:
    """Final SFT record (OpenAI chat format) — the student sees only the text.

    System prompt is SLIM (matches the rest of data/train/train.jsonl).
    Teacher reasoning was generated with full_system_instruction for quality,
    but the SFT record uses slim_system_instruction so appending to train.jsonl
    produces a consistent dataset.
    """
    return {
        "messages": [
            {"role": "system", "content": slim_chat_system_instruction if chat else slim_system_instruction},
            {"role": "user", "content": USER_TEMPLATE.format(content=row["content"])},
            {"role": "assistant", "content": assistant_response},
        ],
    }


def validate_chat_response(response: str, row: dict) -> str | None:
    """Validate a chat-format teacher response against gold. None = OK.

    Checks: a parseable YAML block after </think>, exactly one claim entry,
    frames set == gold frames, codes set == gold claims. No-opposition rows
    must have empty frames and categories.
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
    if not isinstance(claims, list) or len(claims) != 1:
        return f"expected exactly 1 claim entry, got {len(claims) if isinstance(claims, list) else 'none'}"
    entry = claims[0]
    frames = {str(x).strip() for x in (entry.get("frames") or [])}
    cats = entry.get("categories") or []
    codes = {str(c.get("code")).strip() for c in cats if isinstance(c, dict)}
    gold_frames = {str(x).strip() for x in (row.get("true_frames") or [])}
    gold_claims = {str(x).strip() for x in (row.get("true_claims") or [])}
    if frames != gold_frames:
        return f"frames {sorted(frames)} != gold {sorted(gold_frames)}"
    if codes != gold_claims:
        return f"codes {sorted(codes)} != gold {sorted(gold_claims)}"
    return None


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def output_path_for(input_path: Path, chat: bool = False) -> Path:
    """Strip the `_labels` suffix from the input stem to get the SFT output path.

    chat=True appends `_chat`: train_labels.jsonl → train_chat.jsonl.
    """
    stem = input_path.stem
    if stem.endswith("_labels"):
        stem = stem[: -len("_labels")]
    if chat:
        stem += "_chat"
    return input_path.with_name(stem + input_path.suffix)


def load_done_contents(path: Path) -> set[str]:
    """Return the set of content strings already in the output file (from the user turn)."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path) as f:
        for line in f:
            try:
                msgs = json.loads(line)["messages"]
                user = next(m for m in msgs if m.get("role") == "user")
                text = user["content"]
                if text.startswith("### Text:\n"):
                    text = text[len("### Text:\n"):]
                done.add(text.strip())
            except Exception:
                continue
    return done


def process_row(row: dict, call, chat: bool = False) -> tuple[dict | None, str | None]:
    try:
        response = call(build_teacher_user_message(row))
        if chat:
            err = validate_chat_response(response, row)
            if err:
                return None, f"[{row.get('itemId', '?')}] VALIDATION (will retry on rerun): {err}"
        return build_sft_record(row, response, chat=chat), None
    except Exception as e:
        return None, f"[{row.get('itemId', '?')}] ERROR: {e}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one_file(in_path: Path, client: OpenAI, args) -> None:
    out_path = output_path_for(in_path, chat=args.chat)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path) as f:
        all_rows = [json.loads(line) for line in f]
    done = load_done_contents(out_path)
    remaining = [r for r in all_rows if r["content"].strip() not in done]
    if args.max:
        remaining = remaining[: args.max]

    print(f"\n{in_path.name} → {out_path.name}")
    print(f"  Total: {len(all_rows)}, already done: {len(done)}, to process: {len(remaining)}")
    if not remaining:
        return

    call = make_query(client, args.model, args.max_tokens, chat=args.chat)

    n_ok = n_err = 0
    with open(out_path, "a") as fout, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process_row, r, call, args.chat): r for r in remaining}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=in_path.stem):
            rec, err = fut.result()
            if err is not None:
                n_err += 1
                tqdm.write(err)
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_ok += 1
    print(f"  Done: ok={n_ok}, errors={n_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="Labels jsonl paths to process.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model slug for the teacher.")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--max", type=int, default=None, help="Process only the first N remaining rows per file.")
    parser.add_argument("--chat", action="store_true",
                        help="Generate chat-format SFT data (claim → frames + nested categories + reasons). "
                             "Outputs *_chat.jsonl; responses validated against gold instead of clean_recot.")
    args = parser.parse_args()

    client = build_client()
    output_paths: list[Path] = []
    for rel in args.inputs:
        p = Path(rel) if Path(rel).is_absolute() else REPO_ROOT / rel
        if not p.exists():
            print(f"Skip (not found): {p}")
            continue
        run_one_file(p, client, args)
        output_paths.append(output_path_for(p, chat=args.chat))

    # Post-generation filter: drop any rows where the teacher second-guessed itself.
    # Rerunning teacher.py picks them back up via resume-by-content.
    # Chat mode skips this — per-row YAML validation against gold replaces it.
    if output_paths and not args.chat:
        print("\n=== Filtering second-guessing ===")
        for p in output_paths:
            clean_recot(p)


if __name__ == "__main__":
    main()
