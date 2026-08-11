# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas"]
# ///
"""Convert the vendored FLICC CSVs into house-schema JSONL.

    uv run prepare_splits.py

Reads  data/raw/fallacy_{train,val,test}.csv
Writes data/test/val.jsonl   (457)
       data/test/test.jsonl  (256)
       data/train/train.jsonl (1796, Arm B only)

Upstream: github.com/fzanart/FLICC @ 2060d2deb48c81810d75d8af4f5a181bf6be98a6.
The CSVs are vendored verbatim; this script never edits them.

Output schema matches the debunk convention — gold fields prefixed `true_`, a
stable `itemId` for resume support:

    {"itemId": "test-0001", "text": "...", "cards_claim": "2.3",
     "source": "CARDS", "true_fallacy": "misrepresentation"}

Also verifies the per-class counts against the paper's Table 8 and fails loudly
on any mismatch, so a bad vendor copy cannot silently reach the benchmark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prompts import FALLACY_LABELS  # noqa: E402

RAW = ROOT / "data" / "raw"

# Zanartu et al. 2024, Table 8 — the published split sizes.
TABLE_8 = {
    "ad hominem":              (264, 67, 37),
    "anecdote":                (170, 43, 24),
    "cherry picking":          (222, 56, 31),
    "conspiracy theory":       (154, 39, 22),
    "fake experts":            (44, 12, 7),
    "false choice":            (48, 13, 7),
    "false equivalence":       (52, 14, 8),
    "impossible expectations": (144, 37, 21),
    "misrepresentation":       (151, 38, 22),
    "oversimplification":      (143, 36, 20),
    "single cause":            (226, 57, 32),
    "slothful induction":      (178, 45, 25),
}
SPLIT_TOTALS = {"train": 1796, "val": 457, "test": 256}
DEST = {"train": ROOT / "data" / "train", "val": ROOT / "data" / "test",
        "test": ROOT / "data" / "test"}


def load(split: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"fallacy_{split}.csv")
    missing = {"text", "label", "Claim", "Source"} - set(df.columns)
    if missing:
        sys.exit(f"{split}: missing columns {sorted(missing)}")
    df["label"] = df["label"].str.strip().str.lower()
    return df


def check(split: str, df: pd.DataFrame, col: int) -> list[str]:
    """Return a list of problems; empty means the split matches the paper."""
    problems = []
    if len(df) != SPLIT_TOTALS[split]:
        problems.append(f"row count {len(df)} != {SPLIT_TOTALS[split]}")
    unknown = set(df["label"]) - set(FALLACY_LABELS)
    if unknown:
        problems.append(f"labels outside the 12-class space: {sorted(unknown)}")
    counts = df["label"].value_counts().to_dict()
    for label, expected in TABLE_8.items():
        got = counts.get(label, 0)
        if got != expected[col]:
            problems.append(f"{label}: {got} != Table 8's {expected[col]}")
    return problems


def main() -> None:
    frames = {s: load(s) for s in ("train", "val", "test")}

    failures = []
    for col, split in enumerate(("train", "val", "test")):
        problems = check(split, frames[split], col)
        status = "ok" if not problems else "MISMATCH"
        print(f"[{status}] {split}: {len(frames[split])} rows")
        for p in problems:
            print(f"         - {p}")
        failures += problems

    # Cross-split leakage. The paper de-duplicated with exact matching plus a
    # BERT-similarity review; exact matching is what we can reproduce cheaply,
    # and it is the check that matters for scoring validity.
    def norm(s: pd.Series) -> set[str]:
        return set(s.str.lower().str.split().str.join(" "))

    tr, va, te = (norm(frames[s]["text"]) for s in ("train", "val", "test"))
    for name, overlap in (("train∩test", tr & te), ("train∩val", tr & va),
                          ("val∩test", va & te)):
        if overlap:
            failures.append(f"{name}: {len(overlap)} duplicate texts")
            print(f"[MISMATCH] {name}: {len(overlap)} duplicate texts")
        else:
            print(f"[ok] {name}: no exact duplicates")

    if failures:
        sys.exit(f"\n{len(failures)} problem(s) — refusing to write splits.")

    for split, df in frames.items():
        out_dir = DEST[split]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for i, row in enumerate(df.itertuples(index=False), 1):
                f.write(json.dumps({
                    "itemId": f"{split}-{i:04d}",
                    "text": row.text,
                    "cards_claim": str(row.Claim),
                    "source": row.Source,
                    "true_fallacy": row.label,
                }, ensure_ascii=False) + "\n")
        print(f"wrote {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
