"""Create all splits (benchmark + training).

Deterministic, single-path pipeline:

  Benchmark (val / test from aerows_full.jsonl):
    1. Dedup by content, remove overlap with the fine-tuning dataset.
    2. Extract raw claim / narrative codes from annotations.
    3. Stratified 30/70 split on claims (seed=42).
    4. Convert raw labels to the three-level schema
       (true_opposition_detected / true_frames / true_claims).
    5. Apply annotation patches.
    6. Write val.jsonl and test.jsonl.

  Training (train_labels / train_eval_labels from real + synthetic):
    1. Load fine_tuning_data_final_fixed.csv (real) +
       synthetic_zeros.jsonl (synthetic).
    2. Assign ids: real_NNN by CSV row position, synth_NNN by jsonl row.
    3. Stratified 90/10 split on claims (seed=42) — matches the split
       originally published to Hugging Face.
    4. Convert raw labels to the three-level schema.
    5. Write train_labels.jsonl and train_eval_labels.jsonl
       (text + labels only; RECoT reasoning is a separate step).

Same seed, same inputs, same output.

Inputs:
  data/raw/aerows_full.jsonl
  data/raw/fine_tuning_data_final_fixed.csv
  data/raw/synthetic_zeros.jsonl
  data/annotation_patches.json               — optional (--patches)

Outputs:
  data/test/val.jsonl                  (~30% of benchmark pool)
  data/test/test.jsonl                 (~70% of benchmark pool)
  data/train/train_labels.jsonl        (~90% of training pool)
  data/train/train_eval_labels.jsonl   (~10% of training pool)

Usage:
  python prepare_splits.py --patches data/annotation_patches.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

AEROWS_FULL = REPO_ROOT / "data/raw/aerows_full.jsonl"
TRAINING_DATA = REPO_ROOT / "data/raw/fine_tuning_data_final_fixed.csv"
SYNTHETIC_DATA = REPO_ROOT / "data/raw/synthetic_zeros.jsonl"
OUTPUT_VAL = REPO_ROOT / "data/test/val.jsonl"
OUTPUT_TEST = REPO_ROOT / "data/test/test.jsonl"
OUTPUT_TRAIN_LABELS = REPO_ROOT / "data/train/train_labels.jsonl"
OUTPUT_TRAIN_EVAL_LABELS = REPO_ROOT / "data/train/train_eval_labels.jsonl"

C_32_COLLAPSE = {"C_32_1", "C_32_2", "C_32_3", "C_32_4", "C_32_5"}


# ---------------------------------------------------------------------------
# Raw label extraction + mapping to the three-level label schema
# ---------------------------------------------------------------------------

def extract_raw_labels(annotations) -> tuple[list[str], list[str]]:
    """Pull claim + narrative codes from an aerows annotations field."""
    if not annotations:
        return ["C_0_0"], ["N_0"]
    ann = annotations[0]
    claims = sorted(l["code"] for l in ann.get("labels", [])) or ["C_0_0"]
    narratives = sorted(t["code"] for t in ann.get("topics", [])) or ["N_0"]
    return claims, narratives


def _parse_list_col(s):
    """Parse stringified list column from CSV, handling smart quotes."""
    if isinstance(s, list):
        return s
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    return sorted(ast.literal_eval(s))


def load_training_pool() -> pd.DataFrame:
    """Load real + synthetic training examples into a single DataFrame.

    Columns: itemId, content, raw_claims, raw_narratives.
    itemId = "real_NNN" by CSV row position, "synth_NNN" by jsonl row.
    """
    df_real = pd.read_csv(TRAINING_DATA)
    df_real["raw_claims"] = df_real["true_claims"].apply(_parse_list_col)
    df_real["raw_narratives"] = df_real["true_narratives"].apply(_parse_list_col)
    df_real["raw_claims"] = df_real["raw_claims"].apply(
        lambda cs: ["C_16_0" if c == "C_16" else c for c in cs]
    )
    df_real = df_real.rename(columns={"text": "content"}).reset_index(drop=True)
    df_real["itemId"] = [f"real_{i:03d}" for i in range(len(df_real))]

    rows = []
    with open(SYNTHETIC_DATA) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            rows.append({
                "itemId": f"synth_{i:03d}",
                "content": r["text"],
                "raw_claims": r.get("true_claims") or ["C_0_0"],
                "raw_narratives": r.get("true_narratives") or ["N_0"],
            })
    df_synth = pd.DataFrame(rows)

    cols = ["itemId", "content", "raw_claims", "raw_narratives"]
    return pd.concat([df_real[cols], df_synth[cols]], ignore_index=True)


def to_labels(narratives_raw: list[str], claims_raw: list[str]) -> dict:
    """Map raw annotator codes to the three-level label schema."""
    if "C_0_0" in claims_raw:
        return {"true_opposition_detected": False, "true_frames": [], "true_claims": []}

    seen: set[str] = set()
    claims_out: list[str] = []
    for c in claims_raw:
        mapped = "C_32_0" if c in C_32_COLLAPSE else c
        if mapped not in seen:
            seen.add(mapped)
            claims_out.append(mapped)
    if not claims_out:
        claims_out = ["C_0_1"]

    frames_out = list(narratives_raw) if narratives_raw else ["N_0"]
    return {
        "true_opposition_detected": True,
        "true_frames": frames_out,
        "true_claims": claims_out,
    }


# ---------------------------------------------------------------------------
# Annotation patches
# ---------------------------------------------------------------------------

def load_patches(path: Path) -> dict[str, dict]:
    with open(path) as f:
        patches = json.load(f)
    by_id: dict[str, dict] = {}
    for p in patches:
        iid = p["itemId"]
        if iid in by_id:
            raise ValueError(f"Duplicate patch for itemId {iid}")
        by_id[iid] = p
    return by_id


def apply_patch(row: dict, patch: dict) -> dict:
    row["true_opposition_detected"] = bool(patch["opposition_detected"])
    row["true_frames"] = list(patch.get("frames") or [])
    row["true_claims"] = list(patch.get("claims") or [])
    row["patch_applied"] = {"note": patch.get("note", ""), "itemId": patch["itemId"]}
    return row


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def bucket_rare_labels(label_lists: list[list[str]], min_count: int = 2) -> list[list[str]]:
    counts = Counter(l for labels in label_lists for l in labels)
    rare = {l for l, c in counts.items() if c < min_count}
    if rare:
        print(f"  Rare labels bucketed as _rare_ (< {min_count}): {sorted(rare)}")
    return [[("_rare_" if l in rare else l) for l in labels] for labels in label_lists]


def stratified_split(df: pd.DataFrame, label_col: str, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = df[label_col].tolist()
    labels = bucket_rare_labels(labels)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    x = np.arange(len(df)).reshape(-1, 1)

    np.random.seed(seed)
    x_train, _y_train, x_test, _y_test = iterative_train_test_split(x, y, test_size=test_size)
    return df.iloc[x_train.flatten()].copy(), df.iloc[x_test.flatten()].copy()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="Random state (default 42).")
    parser.add_argument("--val-frac", type=float, default=0.3, help="Validation fraction (default 0.3).")
    parser.add_argument(
        "--patches",
        default="data/annotation_patches.json",
        help="Annotation patches JSON path (default: data/annotation_patches.json). "
             "Pass --no-patches to skip.",
    )
    parser.add_argument(
        "--no-patches",
        dest="patches",
        action="store_const",
        const=None,
        help="Skip applying annotation patches.",
    )
    args = parser.parse_args()

    patches: dict[str, dict] = {}
    if args.patches:
        patches_path = Path(args.patches)
        if not patches_path.is_absolute():
            patches_path = REPO_ROOT / patches_path
        patches = load_patches(patches_path)
        print(f"Loaded {len(patches)} annotation patches from {patches_path}")
    else:
        print("Skipping annotation patches (--no-patches).")

    # ── Load and clean benchmark pool ──
    df = pd.read_json(AEROWS_FULL, lines=True)
    df_ft = pd.read_csv(TRAINING_DATA)
    print(f"Aerows raw: {len(df)} rows")

    before = len(df)
    df["_key"] = df["content"].str.strip()
    df = df.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    print(f"After dedup: {len(df)} ({before - len(df)} duplicates removed)")

    ft_texts = set(df_ft["text"].str.strip())
    overlap = df["content"].str.strip().isin(ft_texts)
    print(f"Training overlap removed: {overlap.sum()}")
    df = df[~overlap].copy().reset_index(drop=True)
    print(f"Benchmark pool: {len(df)} rows")

    # ── Extract raw labels (used for stratification) ──
    raw = df["annotations"].apply(extract_raw_labels)
    df["raw_claims"] = raw.apply(lambda x: x[0])
    df["raw_narratives"] = raw.apply(lambda x: x[1])
    df["annotator"] = df["annotations"].apply(lambda a: a[0]["userEmail"] if a else None)

    # ── Stratified split on raw claims (reproduces the HF split) ──
    print(f"\n=== Benchmark split (val {args.val_frac*100:.0f}% / test {(1-args.val_frac)*100:.0f}%) ===")
    df_test, df_val = stratified_split(df, "raw_claims", test_size=args.val_frac, seed=args.seed)
    print(f"  Val:  {len(df_val)} rows")
    print(f"  Test: {len(df_test)} rows")

    overlap_ct = len(set(df_val["content"].str.strip().str.lower()) & set(df_test["content"].str.strip().str.lower()))
    print(f"  Val <-> Test text overlap: {overlap_ct}")

    def finalize_benchmark(frame: pd.DataFrame, out_path: Path) -> None:
        stats: Counter = Counter()
        with open(out_path, "w") as f:
            for _, row in frame.iterrows():
                lab = to_labels(row["raw_narratives"], row["raw_claims"])
                rec = {
                    "itemId": row["itemId"],
                    "annotator": row["annotator"],
                    "content": row["content"],
                    "true_opposition_detected": lab["true_opposition_detected"],
                    "true_frames": lab["true_frames"],
                    "true_claims": lab["true_claims"],
                }
                if rec["itemId"] in patches:
                    rec = apply_patch(rec, patches[rec["itemId"]])
                    stats["patches_applied"] += 1
                if any(c in C_32_COLLAPSE for c in row["raw_claims"]):
                    stats["c32x_collapsed"] += 1
                if rec["true_opposition_detected"]:
                    stats["opposition_true"] += 1
                else:
                    stats["opposition_false"] += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {out_path}: {dict(stats)}")

    (REPO_ROOT / "data/test").mkdir(parents=True, exist_ok=True)
    print("\n=== Writing benchmark splits ===")
    finalize_benchmark(df_val, OUTPUT_VAL)
    finalize_benchmark(df_test, OUTPUT_TEST)

    # ── Training split (train_labels / train_eval_labels) ──
    print("\n=== Loading training pool ===")
    df_tr = load_training_pool()
    print(f"Training pool: {len(df_tr)} rows "
          f"({sum(df_tr['itemId'].str.startswith('real_'))} real, "
          f"{sum(df_tr['itemId'].str.startswith('synth_'))} synthetic)")

    eval_frac = 0.10
    print(f"\n=== Training split (train {100-eval_frac*100:.0f}% / eval {eval_frac*100:.0f}%) ===")
    df_train, df_train_eval = stratified_split(df_tr, "raw_claims", test_size=eval_frac, seed=args.seed)
    print(f"  Train:      {len(df_train)} rows")
    print(f"  Train eval: {len(df_train_eval)} rows")

    # Benchmark texts (normalized) — train rows matching val/test are dropped
    # AFTER the split, so the partition of all other rows is unchanged. The
    # benchmark-side dedup above uses exact strip-matching and misses
    # whitespace/quote near-dupes (e.g. real_106 ≈ heartland_171).
    def _norm(t: str) -> str:
        import re as _re
        return _re.sub(r"\s+", " ", str(t)).strip().lower()

    bench_norm = {_norm(c) for c in df_val["content"]} | {_norm(c) for c in df_test["content"]}

    def finalize_train(frame: pd.DataFrame, out_path: Path) -> None:
        stats: Counter = Counter()
        with open(out_path, "w") as f:
            for _, row in frame.iterrows():
                if _norm(row["content"]) in bench_norm:
                    stats["dropped_in_benchmark"] += 1
                    continue
                lab = to_labels(row["raw_narratives"], row["raw_claims"])
                rec = {
                    "itemId": row["itemId"],
                    "content": row["content"],
                    "true_opposition_detected": lab["true_opposition_detected"],
                    "true_frames": lab["true_frames"],
                    "true_claims": lab["true_claims"],
                }
                if any(c in C_32_COLLAPSE for c in row["raw_claims"]):
                    stats["c32x_collapsed"] += 1
                if rec["true_opposition_detected"]:
                    stats["opposition_true"] += 1
                else:
                    stats["opposition_false"] += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {out_path}: {dict(stats)}")

    (REPO_ROOT / "data/train").mkdir(parents=True, exist_ok=True)
    print("\n=== Writing training splits ===")
    finalize_train(df_train, OUTPUT_TRAIN_LABELS)
    finalize_train(df_train_eval, OUTPUT_TRAIN_EVAL_LABELS)


if __name__ == "__main__":
    main()
