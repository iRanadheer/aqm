# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "scikit-learn"]
# ///
"""Bootstrap confidence intervals for FLICC results.

Follows debunk/evals/significance.py: 95% BCa bootstrap via scipy, 9,999
resamples, seed 42, reporting the difference and its interval rather than a
significance verdict (Ulmer et al., LREC 2022; Koehn 2004).

Two comparison types, and the distinction matters:

  vs BASELINE — one-sample. Zanartu et al.'s 0.73 is a published point
      estimate; their model is not re-run here, so there are no paired
      predictions and no interval on their side. We bootstrap OUR macro F1 and
      ask whether its interval excludes their fixed 0.73. This is weaker than a
      paired test and is labelled as such.

  vs EACH OTHER — paired. Our own runs share the same items, so the bootstrap
      is over the per-item difference, which is the stronger comparison.

Parse failures, out-of-vocabulary labels, and API errors all count as WRONG,
matching generate_report.py. No hand-repair.

    uv run evals/significance.py --split test
    uv run evals/significance.py --split test --pair anthropic-claude-opus-4-7-bare openai-gpt-5-5-bare

Writes data/significance/summary_<split>.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prompts import FALLACY_LABELS, extract_raw_label  # noqa: E402

LABELS = list(FALLACY_LABELS)
WRONG = "__wrong__"  # never equals a gold label

# Zanartu et al. 2024 Table 4 macro F1 — published point estimates.
BASELINE_MACRO_F1 = {"test": 0.73, "val": 0.73}


def load(path: Path) -> dict[str, tuple[str, str]]:
    """itemId -> (gold, pred). Keyed rather than positional: infer.py appends in
    completion order, so two runs over the same split are not row-aligned."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # Re-parsed from the response, matching generate_report.py, so the two
        # never disagree about what a run predicted.
        p = extract_raw_label(str(r.get("response", "")))
        out[r["itemId"]] = (r["true_fallacy"], p if p in LABELS else WRONG)
    return out


def as_arrays(data: dict, keys: list[str]) -> tuple[list[str], list[str]]:
    return [data[k][0] for k in keys], [data[k][1] for k in keys]


def macro_f1(gold: list[str], pred: list[str], idx: np.ndarray) -> float:
    g = [gold[i] for i in idx]
    p = [pred[i] for i in idx]
    return float(f1_score(g, p, labels=LABELS, average="macro", zero_division=0))


def ci(fn, n: int, cfg: dict) -> tuple[float, float, float]:
    """Bootstrap over item indices. Falls back to percentile if BCa degenerates
    — with 12 classes and small support, BCa's acceleration term can blow up."""
    point = fn(np.arange(n))
    for method in ("BCa", "percentile"):
        res = bootstrap(
            (np.arange(n),), lambda ix: fn(ix.astype(np.intp)),
            n_resamples=cfg["n_resamples"], confidence_level=cfg["confidence"],
            method=method, rng=np.random.default_rng(cfg["seed"]), vectorized=False)
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
        if np.isfinite(lo) and np.isfinite(hi):
            return float(point), float(lo), float(hi)
    return float(point), float("nan"), float("nan")


def verdict(lo: float, hi: float, ref: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/a"
    if lo > ref:
        return "**improves**"
    if hi < ref:
        return "lower"
    return "comparable"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"),
                    help="Paired comparison between two run slugs.")
    ap.add_argument("--n-resamples", type=int, default=9999)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = {"n_resamples": a.n_resamples, "confidence": a.confidence, "seed": a.seed}

    results_dir = ROOT / "data" / "results" / a.split
    runs = {p.stem: load(p) for p in sorted(results_dir.glob("*.jsonl"))}
    if not runs:
        sys.exit(f"No result files in {results_dir}")

    base = BASELINE_MACRO_F1[a.split]
    lines = [f"# FLICC significance — {a.split} split", "",
             f"95% BCa bootstrap, {a.n_resamples} resamples, seed {a.seed}. "
             f"Parse failures and out-of-vocabulary labels count as wrong.", "",
             "## vs published baseline (one-sample)", "",
             f"Zanartu et al. DeBERTa-v2-xlarge macro F1 = **{base}**, a published "
             f"point estimate. Their model is not re-run, so this is a one-sample "
             f"comparison against a fixed value — weaker than a paired test.", "",
             "| Run | n | Macro F1 | 95% CI | vs baseline |",
             "|---|---:|---:|---|---|"]

    for name, data in runs.items():
        gold, pred = as_arrays(data, sorted(data))
        n = len(gold)
        pt, lo, hi = ci(lambda ix: macro_f1(gold, pred, ix), n, cfg)
        lines.append(f"| {name} | {n} | {pt:.3f} | [{lo:.3f}, {hi:.3f}] | "
                     f"{verdict(lo, hi, base)} |")

    if a.pair:
        na, nb = a.pair
        for s in (na, nb):
            if s not in runs:
                sys.exit(f"Unknown run {s!r}. Available: {sorted(runs)}")
        keys = sorted(set(runs[na]) & set(runs[nb]))
        ga, pa = as_arrays(runs[na], keys)
        gb, pb = as_arrays(runs[nb], keys)
        if ga != gb:
            sys.exit(f"Gold labels differ between {na} and {nb} — alignment off.")
        n = len(keys)
        d, lo, hi = ci(
            lambda ix: macro_f1(ga, pa, ix) - macro_f1(gb, pb, ix), n, cfg)
        lines += ["", "## Paired comparison", "",
                  "| A | B | Delta (A-B) | 95% CI | verdict |",
                  "|---|---|---:|---|---|",
                  f"| {na} | {nb} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                  f"{verdict(lo, hi, 0.0)} |"]

    out_dir = ROOT / "data" / "significance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"summary_{a.split}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
