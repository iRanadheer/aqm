# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "scikit-learn", "pyyaml"]
# ///
"""Score a dynamic-fewshot result file with the SIBLING chapter's own metrics.

This chapter does not reimplement scoring — it imports cards/generate_report.py
and wind/generate_report.py so the numbers are byte-identical to how the
base/fine-tuned models were reported. Few-shot output is RECoT (thinking on),
identical in shape to the base zero-shot runs, so the default cards parser
(require_think=True) applies unchanged.

  uv run score.py --task cards data/results/cards/test/qwen-qwen3.5-9b-dynamic-k10.jsonl
  uv run score.py --task wind  data/results/wind/test/qwen-qwen3.5-9b-static-k10.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--task", required=True, choices=["cards", "wind"])
ap.add_argument("paths", nargs="+", help="Result jsonl file(s) to score.")
args = ap.parse_args()


def score_cards(path: Path) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "cards"))
    import pandas as pd
    import generate_report as gr  # cards/generate_report.py

    df = pd.read_json(path, lines=True)
    # Few-shot output is RECoT (thinking on) like the base runs -> default parser.
    df["pred_claims"] = df["response"].apply(lambda r: gr.parse_response(r, require_think=True))
    parse_fail = int((df["pred_claims"].map(len) == 0).sum())
    out = {"n_samples": len(df), "parse_failures": parse_fail}
    for ms in gr.MIN_SUPPORTS:
        for lvl in gr.LEVELS:
            out[f"L{lvl}_minsup{ms}"] = gr.compute_metrics(df, lvl, ms)
    return out


def score_wind(path: Path) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "wind"))
    import generate_report as gr  # wind/generate_report.py — parses YAML, no <think> needed
    return gr.score_one(path)


scorer = {"cards": score_cards, "wind": score_wind}[args.task]

for p in args.paths:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    print(f"\n=== {path.name} ===")
    result = scorer(path)
    print(json.dumps(result, indent=2, default=str))
    summary_path = path.with_suffix(".score.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"(saved {summary_path.name})")
