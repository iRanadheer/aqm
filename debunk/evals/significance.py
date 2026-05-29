"""Bootstrap confidence intervals for Debunk results — the simple, defensible view.

Debunk is single-label fact-checking: each climate claim gets one verdict,
scored against Leippold gold on two benchmarks —
    veracityV2     (4-class, vs true_veracity)
    climinator_v4  (12-class, vs true_cfb_label)

Headline metric is **accuracy** (per-claim correctness), reported with a 95%
**BCa bootstrap confidence interval** (`scipy.stats.bootstrap`). MCC and
macro-F1 (over present classes) are shown as point estimates, and the majority
baseline is the reference. Parse failures / API errors count as wrong. This
follows the recommendation to report the difference and its confidence interval
rather than significance-test verdicts (Ulmer et al., LREC 2022; Koehn 2004).

NOTE: the test set is small (~160 claims), so confidence intervals are WIDE and
many comparisons land on `comparable` simply from limited power — that is itself
a finding worth stating.

The Climinator file also scores the **paper baseline** (Leippold's CLIM
predictions baked into the test set), recomputed on our exact rows with the same
code, so the "vs paper" comparison is strictly fair.

Comparisons report the accuracy gap (A − B) and its CI, labelled:
    improves / lower  — CI clears 0      comparable — CI includes 0 (too close)

Usage:
    python evals/significance.py             # -> data/significance/summary_<variant>.md
    python evals/significance.py pair climinator_v4 qwen-qwen3-5-27b-rag-pplx-ctx __paper__

Flags: --n-resamples 9999  --confidence 0.95  --seed 42  --out-dir data/significance
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompts import LABEL_SETS, GOLD_FIELDS, extract_raw_label, normalise_gold  # noqa: E402

PARSE_FAIL = "__PARSE_FAIL__"
RAG = "-rag-pplx-ctx"
VARIANTS = ["veracityV2", "climinator_v4"]

# The six models present on disk for both live benchmarks, each run offline and
# with RAG (pplx-ctx). (display, base slug)
MODELS = [
    ("Claude Opus 4.7",  "anthropic-claude-opus-4-7"),
    ("DeepSeek V4 Flash", "deepseek-deepseek-v4-flash"),
    ("GPT-4o-mini",      "openai-gpt-4o-mini"),
    ("GPT-5.5",          "openai-gpt-5-5"),
    ("Qwen3.5-9B",       "qwen-qwen3-5-9b"),
    ("Qwen3.5-27B",      "qwen-qwen3-5-27b"),
]


# ---------------------------------------------------------------------------
# Loading — claim -> (gold, pred), using the report's parse/normalise pipeline
# ---------------------------------------------------------------------------

def _key(r):
    return r.get("claim") or str(r.get("itemId"))


def _pred_from_response(resp, base):
    if isinstance(resp, str) and resp.startswith("ERROR:"):
        return PARSE_FAIL
    raw = extract_raw_label(resp)
    return raw if (raw is not None and raw in base) else PARSE_FAIL


def load(slug, variant):
    path = ROOT / "data" / "results" / "test" / f"{slug}-{variant}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    base = set(LABEL_SETS[variant])
    gold_field = GOLD_FIELDS[variant]
    out = {}
    for line in open(path):
        r = json.loads(line)
        gold = normalise_gold(r.get(gold_field, ""))
        if gold not in base:
            continue                          # no usable gold for this variant
        out[_key(r)] = (gold, _pred_from_response(r.get("response", ""), base))
    return out


def load_paper(variant):
    """Leippold CLIM baseline: predictions baked into the test set as
    `true_climinator`, scored vs gold with the same parser (climinator only)."""
    base = set(LABEL_SETS[variant])
    gold_field = GOLD_FIELDS[variant]
    test_path = ROOT / "data" / "test" / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"missing: {test_path}")
    out = {}
    for line in open(test_path):
        r = json.loads(line)
        gold = normalise_gold(r.get(gold_field, ""))
        if gold not in base:
            continue
        clim = (r.get("true_climinator") or "").strip()
        if not clim or clim.upper() == "NEI":
            pred = PARSE_FAIL                  # paper abstains on these
        else:
            wrapped = clim.upper().replace("_", " ").replace("-", " ")
            pred = _pred_from_response(f"```yaml\nassessment: {wrapped}\n```", base)
        out[_key(r)] = (gold, pred)
    return out


_CACHE = {}


def get_data(slug, variant):
    if (slug, variant) not in _CACHE:
        _CACHE[(slug, variant)] = (load_paper(variant) if slug == "__paper__"
                                   else load(slug, variant))
    return _CACHE[(slug, variant)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def correct_array(data, keys):
    return np.array([1.0 if data[k][1] == data[k][0] else 0.0 for k in keys])


def mcc_macro(data, keys):
    yt = [data[k][0] for k in keys]
    yp = [data[k][1] for k in keys]
    present = sorted(set(yt))
    return (matthews_corrcoef(yt, yp),
            f1_score(yt, yp, labels=present, average="macro", zero_division=0))


def ci_over(rows, value_fn, cfg):
    rows = np.asarray(rows)
    m = len(rows)
    if m == 0:
        return float("nan"), float("nan"), float("nan")
    point = value_fn(rows)

    def statistic(pos):
        return value_fn(rows[pos.astype(np.intp)])

    for method in ("BCa", "percentile"):
        res = bootstrap((np.arange(m),), statistic, n_resamples=cfg["n_resamples"],
                        confidence_level=cfg["confidence"], method=method,
                        rng=np.random.default_rng(cfg["seed"]), vectorized=False)
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
        if np.isfinite(lo) and np.isfinite(hi):
            return float(point), float(lo), float(hi)
    return float(point), float("nan"), float("nan")


def model_scores(slug, variant, cfg):
    data = get_data(slug, variant)
    keys = sorted(data)
    c = correct_array(data, keys)
    acc, lo, hi = ci_over(np.arange(len(keys)), lambda ii: float(c[ii].mean()), cfg)
    mcc, macro = mcc_macro(data, keys)
    maj = Counter(data[k][0] for k in keys).most_common(1)[0][1] / len(keys)
    return {"n": len(keys), "acc": (acc, lo, hi), "mcc": mcc, "macro": macro,
            "baseline": maj}


def compare(slug_a, slug_b, variant, cfg):
    A, B = get_data(slug_a, variant), get_data(slug_b, variant)
    keys = sorted(set(A) & set(B))
    n_mis = sum(1 for k in keys if A[k][0] != B[k][0])
    if n_mis:
        print(f"  WARNING: {n_mis}/{len(keys)} gold mismatches ({slug_a} vs "
              f"{slug_b}) — alignment off.", file=sys.stderr)
    cA, cB = correct_array(A, keys), correct_array(B, keys)
    a, b = float(cA.mean()), float(cB.mean())
    _, lo, hi = ci_over(np.arange(len(keys)),
                        lambda ii: float(cA[ii].mean() - cB[ii].mean()), cfg)
    return {"a": a, "b": b, "delta": a - b, "lo": lo, "hi": hi, "n": len(keys)}


def verdict(M):
    if not (np.isfinite(M["lo"]) and np.isfinite(M["hi"])):
        return "n/a"
    if M["lo"] > 0:
        return "improves"
    if M["hi"] < 0:
        return "lower"
    return "comparable"


# ---------------------------------------------------------------------------
# Lineup + comparisons per benchmark
# ---------------------------------------------------------------------------

def lineup(variant):
    out = []
    for disp, slug in MODELS:
        out.append((f"{disp} (offline)", slug))
        out.append((f"{disp} + RAG", slug + RAG))
    if variant.startswith("climinator"):
        out.append(("Paper CLIM (recomputed)", "__paper__"))
    return out


def comparison_groups(variant):
    groups = [
        ("RAG helps", [(f"{disp}: +RAG vs offline", slug + RAG, slug)
                       for disp, slug in MODELS]),
        ("Does scale help? (Qwen)", [
            ("27B vs 9B (offline)", "qwen-qwen3-5-27b", "qwen-qwen3-5-9b"),
            ("27B vs 9B (+RAG)", "qwen-qwen3-5-27b" + RAG, "qwen-qwen3-5-9b" + RAG)]),
    ]
    if variant.startswith("climinator"):
        groups.append(("Vs paper baseline (best config = +RAG)",
                       [(f"{disp} +RAG vs Paper", slug + RAG, "__paper__")
                        for disp, slug in MODELS]))
    return groups


# ---------------------------------------------------------------------------
# Markdown report (one file per benchmark variant)
# ---------------------------------------------------------------------------

def build_variant(variant, cfg):
    pct = int(round(cfg["confidence"] * 100))
    nclass = len(LABEL_SETS[variant])
    gold_field = GOLD_FIELDS[variant]

    scores = {}
    for disp, slug in lineup(variant):
        try:
            scores[disp] = model_scores(slug, variant, cfg)
        except FileNotFoundError:
            continue
    if not scores:
        return None
    n = next(iter(scores.values()))["n"]
    baseline = next(iter(scores.values()))["baseline"]

    L = [f"# Debunk — {variant} ({nclass}-class)\n",
         f"Single-label fact-checking verdict vs `{gold_field}` on n={n} claims. "
         f"Headline metric: **accuracy** with a {pct}% BCa bootstrap CI "
         f"(`scipy.stats.bootstrap`, {cfg['n_resamples']} resamples). MCC and "
         f"macro-F1 (present classes) are point estimates. Parse failures / API "
         f"errors count as wrong. **Small n → wide CIs**, so many gaps are "
         f"`comparable` from limited power.\n",
         "## Model scores\n",
         f"| Model | Accuracy ({pct}% CI) | MCC | Macro-F1 |",
         "|---|---|---|---|",
         f"| *Majority baseline* | {baseline:.3f} | 0.000 | — |"]
    for disp, _ in lineup(variant):
        e = scores.get(disp)
        if e is None:
            continue
        a = e["acc"]
        L.append(f"| {disp} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | "
                 f"{e['mcc']:.3f} | {e['macro']:.3f} |")

    L += [f"\n## Comparisons (accuracy)\n",
          f"Gap = A − B with a {pct}% CI. `improves`/`lower` = CI clears 0; "
          f"`comparable` = CI includes 0 (too close to call).\n",
          f"| Comparison | A | B | Gap ({pct}% CI) | Verdict |",
          "|---|---|---|---|---|"]
    for group, pairs in comparison_groups(variant):
        rows = []
        for label, sa, sb in pairs:
            try:
                M = compare(sa, sb, variant, cfg)
            except FileNotFoundError:
                continue
            rows.append(f"| {label} | {M['a']:.3f} | {M['b']:.3f} | "
                        f"{M['delta']:+.3f} [{M['lo']:+.3f}, {M['hi']:+.3f}] | "
                        f"{verdict(M)} |")
        if rows:
            L.append(f"| **{group}** | | | | |")
            L += rows
    return "\n".join(L) + "\n"


def run_report(cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        print(f"  computing {variant} ...")
        text = build_variant(variant, cfg)
        if text is None:
            print(f"    (no result files for {variant})")
            continue
        (out_dir / f"summary_{variant}.md").write_text(text)
        print(f"  -> {out_dir}/summary_{variant}.md")


def run_pair(variant, slug_a, slug_b, cfg):
    M = compare(slug_a, slug_b, variant, cfg)
    pct = int(round(cfg["confidence"] * 100))
    print(f"\n{slug_a}  vs  {slug_b}  ({variant}, n={M['n']})\n")
    print(f"  accuracy: A={M['a']:.4f}  B={M['b']:.4f}  "
          f"gap={M['delta']:+.4f}  {pct}% CI [{M['lo']:+.4f}, {M['hi']:+.4f}]  "
          f"-> {verdict(M)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")
    sub.add_parser("report", help="write data/significance/summary_<variant>.md (default)")
    pr = sub.add_parser("pair", help="one A vs B comparison, console output")
    pr.add_argument("variant", choices=list(LABEL_SETS))
    pr.add_argument("slug_a")
    pr.add_argument("slug_b", help="result slug, or __paper__ for the Leippold baseline")

    for p in (ap, pr):
        p.add_argument("--n-resamples", type=int, default=9999)
        p.add_argument("--confidence", type=float, default=0.95)
        p.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "significance"))

    args = ap.parse_args()
    cfg = {"n_resamples": args.n_resamples, "confidence": args.confidence,
           "seed": args.seed}
    if args.mode == "pair":
        run_pair(args.variant, args.slug_a, args.slug_b, cfg)
    else:
        run_report(cfg, Path(args.out_dir))


if __name__ == "__main__":
    main()
