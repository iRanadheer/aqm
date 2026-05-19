# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas"]
# ///
"""Convert the Leippold2024 veracity CSV into a benchmark JSONL.

Input:
  data/raw/Leippold2024_veracity.csv     (170 rows)

Output:
  data/test/test.jsonl

Each output record (metadata first, true labels at the end so they line
up with `pred_*` fields appended by infer.py):
  {
    "itemId":             "leippold_001",
    "claim":              "<verbatim claim>",
    "source":             "<source attribution>",
    "date":               "<DD-MMM-YY>",
    "true_veracity":      "TRUE" | "MISLEADING" | "FALSE",
    "true_cfb_label":     "Inaccurate" | ... (Climate_Feedback raw label),
    "true_climinator":    "incorrect"  | ... (Climinator raw label)
  }

By default, rows where `Duplicated == True` are dropped (Leippold's published
benchmark de-dupes them). Use `--keep-duplicates` to keep all 170 rows.

Usage:
  python prepare_splits.py
  python prepare_splits.py --keep-duplicates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
INPUT_CSV = REPO_ROOT / "data/raw/Leippold2024_veracity.csv"
OUTPUT_TEST = REPO_ROOT / "data/test/test.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input", default=str(INPUT_CSV),
        help=f"Source CSV (default: {INPUT_CSV.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--output", default=str(OUTPUT_TEST),
        help=f"Output JSONL (default: {OUTPUT_TEST.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--keep-duplicates", action="store_true",
        help="Keep rows where Duplicated == True (default: drop them).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = REPO_ROOT / in_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} rows from {in_path}")

    expected = {"ID", "Claim", "Source", "Date", "Climate_Feedback",
                "Climinator", "Duplicated", "Veracity"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

    before = len(df)
    if not args.keep_duplicates:
        df = df[~df["Duplicated"].astype(bool)].copy()
    print(f"After de-dup: {len(df)} rows ({before - len(df)} duplicates dropped)")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"TRUE": 0, "MISLEADING": 0, "FALSE": 0, "other": 0}
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            veracity = str(row["Veracity"]).strip().upper()
            if veracity not in {"TRUE", "MISLEADING", "FALSE"}:
                counts["other"] += 1
            else:
                counts[veracity] += 1
            rec = {
                "itemId":          f"leippold_{int(row['ID']):03d}",
                "claim":           str(row["Claim"]).strip(),
                "source":          str(row["Source"]).strip(),
                "date":            str(row["Date"]).strip(),
                "true_veracity":   veracity,
                "true_cfb_label":  str(row["Climate_Feedback"]).strip(),
                "true_climinator": str(row["Climinator"]).strip(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")
    print(f"  Veracity distribution: {counts}")


if __name__ == "__main__":
    main()
