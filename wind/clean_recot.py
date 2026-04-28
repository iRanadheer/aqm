"""Remove rows whose assistant `<think>` block shows second-guessing.

Second-guessing is detected by a short list of regex patterns that indicate
the model changed its mind mid-reasoning (e.g., "Wait —", "let me
reconsider", "actually, no"). The RECoT trigger instructs the teacher to
make a single pass; rows that slip through get stripped here so they don't
contaminate SFT training.

Usage (standalone):
    python clean_recot.py data/train/train.jsonl
    python clean_recot.py data/train/train.jsonl data/train/train_eval.jsonl

Usage (imported):
    from clean_recot import clean_file
    kept, dropped = clean_file(Path("data/train/train.jsonl"))
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Patterns that mark the teacher changing its mind mid-reasoning.
# Scoped to the assistant's <think> block.
SECOND_GUESS_PATTERNS = [
    r"\bwait\b[,.—-]",
    r"\breconsider(?:ing)?\b",
    r"\blet me rethink\b",
    r"\bchange(d)? my mind\b",
    r"\bscratch that\b",
    r"\bhold on\b",
    r"\bactually[,]?\s+(?:no|wait|let|i)\b",
    r"\bon (?:second thought|reflection|reread)\b",
    r"\bre-?read(?:ing)?\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in SECOND_GUESS_PATTERNS]


def has_second_guess(assistant_content: str) -> bool:
    """True if the <think> block in the assistant response contains a reconsideration."""
    m = re.search(r"<think>(.*?)</think>", assistant_content, re.DOTALL)
    think = m.group(1) if m else assistant_content
    return any(rx.search(think) for rx in _COMPILED)


def clean_file(path: Path, quiet: bool = False) -> tuple[int, int]:
    """Drop rows with second-guessing. Rewrites the file in place.

    Returns (kept_count, dropped_count). File must be an OpenAI chat-format
    jsonl (each row has {"messages": [...system, user, assistant]}).
    """
    if not path.exists():
        if not quiet:
            print(f"{path}: not found — skipping")
        return 0, 0

    rows = [json.loads(line) for line in open(path)]
    kept = []
    dropped = []
    for r in rows:
        assistant = next((m for m in r.get("messages", []) if m.get("role") == "assistant"), None)
        if assistant and has_second_guess(assistant.get("content", "")):
            dropped.append(r)
        else:
            kept.append(r)

    if dropped:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        shutil.move(tmp, path)

    if not quiet:
        print(f"{path.name}: {len(rows)} → {len(kept)} rows  ({len(dropped)} dropped)")
        for d in dropped[:10]:
            user = next((m for m in d["messages"] if m.get("role") == "user"), None)
            text = (user["content"] if user else "").replace("### Text:\n", "").strip().replace("\n", " ")
            print(f"    - {text[:100]}...")
        if len(dropped) > 10:
            print(f"    ... and {len(dropped) - 10} more")

    return len(kept), len(dropped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="jsonl file(s) to clean (repo-relative if not absolute).")
    args = parser.parse_args()

    for rel in args.paths:
        p = Path(rel)
        if not p.is_absolute():
            p = REPO_ROOT / p
        clean_file(p)


if __name__ == "__main__":
    main()
