# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "scikit-learn", "numpy"]
# ///
"""Score FLICC result files and emit metrics_summary.{json,md}.

    uv run generate_report.py                    # score every run in data/results/val
    uv run generate_report.py --split test
    uv run generate_report.py --split val --perclass

Headline metric is macro F1 over the twelve classes, matching the paper.

STRICT PARSING. A prediction that is missing, unparseable, or outside the
twelve-label vocabulary counts as WRONG — never repaired, never dropped. This is
deliberately harsher than Zanartu et al., who hand-assigned "the most likely
label" to 44 of 256 GPT-4 outputs and 8 Gemini outputs. Parse and vocabulary
failures are reported as their own columns so the cost of the stricter protocol
is visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prompts import FALLACY_LABELS, extract_raw_label  # noqa: E402

LABELS = list(FALLACY_LABELS)

# Zanartu et al. 2024, Table 4 — the published DeBERTa-v2-xlarge test scores.
# Point estimates from the paper; their model is not re-run. Val column is
# Table 4's validation half, usable as a development yardstick.
BASELINE = {
    "test": {"accuracy": 0.74, "macro_f1": 0.73, "weighted_f1": 0.74,
             "per_class": {"ad hominem": 0.79, "anecdote": 0.90,
                           "cherry picking": 0.77, "conspiracy theory": 0.80,
                           "fake experts": 1.00, "false choice": 0.67,
                           "false equivalence": 0.43,
                           "impossible expectations": 0.77,
                           "misrepresentation": 0.68, "oversimplification": 0.74,
                           "single cause": 0.72, "slothful induction": 0.53}},
    "val":  {"accuracy": 0.73, "macro_f1": 0.73, "weighted_f1": 0.73,
             "per_class": {"ad hominem": 0.75, "anecdote": 0.90,
                           "cherry picking": 0.67, "conspiracy theory": 0.80,
                           "fake experts": 0.96, "false choice": 0.80,
                           "false equivalence": 0.46,
                           "impossible expectations": 0.71,
                           "misrepresentation": 0.63, "oversimplification": 0.70,
                           "single cause": 0.77, "slothful induction": 0.65}},
}

SENTINEL = "__unparsed__"  # never equals a gold label, so it always scores wrong


def score_file(path: Path) -> dict:
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    df = pd.DataFrame(rows)

    y_true = df["true_fallacy"].tolist()
    # Re-parse from the stored response rather than trusting the `pred_label`
    # infer.py wrote at generation time, so parser fixes apply to runs already
    # on disk without paying to re-run inference.
    raw = [extract_raw_label(str(r)) for r in df["response"]]

    n_error = sum(1 for r in df["response"].astype(str) if r.startswith("ERROR:"))
    n_unparsed = sum(1 for p in raw if p is None or (isinstance(p, float)))
    n_oov = sum(1 for p in raw
                if p is not None and not isinstance(p, float) and p not in LABELS)

    y_pred = [p if (p in LABELS) else SENTINEL for p in raw]

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "file": path.name,
        "n": len(df),
        "errors": n_error,
        "unparsed": n_unparsed,
        "out_of_vocab": n_oov,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS,
                                   average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=LABELS,
                                      average="weighted", zero_division=0), 4),
        "per_class": {
            lab: {"precision": round(pp, 3), "recall": round(rr, 3),
                  "f1": round(ff, 3), "support": int(ss)}
            for lab, pp, rr, ff, ss in zip(LABELS, p, r, f1, sup)
        },
        "confusion": confusion_matrix(
            y_true, y_pred, labels=LABELS + [SENTINEL]).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--perclass", action="store_true",
                    help="Include the per-class table in the markdown output.")
    a = ap.parse_args()

    results_dir = ROOT / "data" / "results" / a.split
    files = sorted(results_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"No result files in {results_dir}")

    scored = {}
    for f in files:
        try:
            scored[f.stem] = score_file(f)
        except Exception as e:  # a partial or malformed run should not kill the report
            print(f"[skip] {f.name}: {e}")

    base = BASELINE[a.split]
    out_json = results_dir / "metrics_summary.json"
    out_json.write_text(json.dumps({"baseline_flicc_deberta": base, **scored}, indent=1))

    lines = [f"# FLICC results — {a.split} split", "",
             f"Baseline: Zanartu et al. 2024 DeBERTa-v2-xlarge (887M), "
             f"macro F1 **{base['macro_f1']}**, accuracy {base['accuracy']} "
             f"(published, not re-run).", "",
             "Parse and vocabulary failures count as wrong. No hand-repair.", "",
             "| Run | N | Macro F1 | vs 0.73 | Accuracy | Weighted F1 | Unparsed | OOV | Errors |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in sorted(scored.items(), key=lambda kv: -kv[1]["macro_f1"]):
        delta = m["macro_f1"] - base["macro_f1"]
        lines.append(
            f"| {name} | {m['n']} | **{m['macro_f1']:.3f}** | {delta:+.3f} | "
            f"{m['accuracy']:.3f} | {m['weighted_f1']:.3f} | "
            f"{m['unparsed']} | {m['out_of_vocab']} | {m['errors']} |")

    if a.perclass:
        lines += ["", "## Per-class F1", "",
                  "| Fallacy | n | baseline | " + " | ".join(scored) + " |",
                  "|---|---:|---:|" + "---:|" * len(scored)]
        for lab in LABELS:
            sup = next(iter(scored.values()))["per_class"][lab]["support"]
            cells = " | ".join(f"{m['per_class'][lab]['f1']:.2f}" for m in scored.values())
            lines.append(f"| {lab} | {sup} | {base['per_class'][lab]:.2f} | {cells} |")

    out_md = results_dir / "metrics_summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_json}\nwrote {out_md}")


if __name__ == "__main__":
    main()
