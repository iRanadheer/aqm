"""Bootstrap confidence intervals for Wind results — the simple, defensible view.

Wind has three tasks: binary **opposition detection**, and **frames**/**claims**
multi-label tagging (reported on the opposition-only rows, where they apply).
For each we report F1 with a 95% **BCa bootstrap confidence interval**
(`scipy.stats.bootstrap`) — no hand-rolled resampling, no p-values. This follows
the recommendation to report the difference and its confidence interval rather
than significance-test verdicts (Ulmer et al., LREC 2022; Koehn 2004 for the
paired bootstrap).

Parse failures / API errors are penalised (forced detection miss + a sentinel
label that can't match gold), exactly as in `generate_report.py`, so a model
can't get a free pass by failing on hard rows.

Model comparisons report the gap (A − B) and its CI, labelled:
    improves    — CI entirely above 0
    lower       — CI entirely below 0
    comparable  — CI includes 0 (too close to call)

Usage:
    python evals/significance.py             # -> data/significance/summary_test.md
    python evals/significance.py pair windy-qwen35-27b claude-opus-4-7

Flags: --split test  --n-resamples 9999  --confidence 0.95  --seed 42
       --out-dir data/significance
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generate_report import parse_response  # noqa: E402

SENTINEL = "__PARSE_FAIL__"
# Minimum gold support for a code to enter the macro vocabulary (matches the
# CARDS convention and generate_report.MIN_SUPPORT). Detection and samples-F1
# are support-independent; only macro is restricted.
MIN_SUPPORT = 3
METRICS = [("det", "Detection F1"),
           ("frames", "Frames Samples-F1 (opp)"),
           ("frames_macro", f"Frames Macro-F1 (opp, sup≥{MIN_SUPPORT})"),
           ("claims", "Claims Samples-F1 (opp)"),
           ("claims_macro", f"Claims Macro-F1 (opp, sup≥{MIN_SUPPORT})")]


# ---------------------------------------------------------------------------
# Final reported lineup and comparisons (slugs verified present on test)
# ---------------------------------------------------------------------------

LINEUP = [
    ("Qwen3.5-4B (base, zero-shot)",        "qwen35-4b-base"),
    ("Qwen3.5-9B (base, zero-shot)",        "qwen35-9b-base"),
    ("Qwen3.5-27B (base, zero-shot)",       "qwen35-27b-base"),
    ("Windy-Qwen3.5-4B (ours)",             "windy-qwen35-4b"),
    ("Windy-Qwen3.5-9B (ours)",             "windy-qwen35-9b"),
    ("Windy-Qwen3.5-27B (ours)",            "windy-qwen35-27b"),
    ("Windy-Qwen3.5-27B FP8 (ours)",        "windy-qwen35-27b-fp8"),
    ("CARDS-Wind-Qwen3.6-27B (ours, joint)", "cards-wind-qwen36-27b"),
    ("Claude Opus 4.7 (zero-shot)",         "claude-opus-4-7"),
    ("GPT-5.5 (zero-shot)",                 "openai-gpt-5-5"),
]

COMPARISONS = [
    ("Fine-tuning helps", [
        ("Windy-4B vs base",  "windy-qwen35-4b",  "qwen35-4b-base"),
        ("Windy-9B vs base",  "windy-qwen35-9b",  "qwen35-9b-base"),
        ("Windy-27B vs base", "windy-qwen35-27b", "qwen35-27b-base")]),
    ("Does scale help?", [
        ("Windy-9B vs 4B",  "windy-qwen35-9b",  "windy-qwen35-4b"),
        ("Windy-27B vs 9B", "windy-qwen35-27b", "windy-qwen35-9b")]),
    ("Vs frontier APIs", [
        ("Windy-27B vs Claude Opus 4.7",      "windy-qwen35-27b",     "claude-opus-4-7"),
        ("Windy-27B vs GPT-5.5",              "windy-qwen35-27b",     "openai-gpt-5-5"),
        ("CARDS-Wind-27B vs Claude Opus 4.7", "cards-wind-qwen36-27b", "claude-opus-4-7"),
        ("CARDS-Wind-27B vs GPT-5.5",         "cards-wind-qwen36-27b", "openai-gpt-5-5")]),
    ("Joint vs Wind-only", [
        ("CARDS-Wind-27B vs Windy-27B", "cards-wind-qwen36-27b", "windy-qwen35-27b")]),
    ("FP8 quantization", [
        ("Windy-27B FP8 vs full",      "windy-qwen35-27b-fp8",      "windy-qwen35-27b"),
        ("CARDS-Wind-27B FP8 vs full", "cards-wind-qwen36-27b-fp8", "cards-wind-qwen36-27b")]),
]


# ---------------------------------------------------------------------------
# Loader — applies the same parse-failure penalty the report does.
# ---------------------------------------------------------------------------

def load(slug, split):
    path = ROOT / "data" / "results" / split / f"{slug}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    rows = [json.loads(l) for l in open(path)]
    # Result files are written in completion order; sort by `content` to align
    # rows across models. content is unique per row; itemId is NOT a safe key —
    # its missing-value representation differs across files (float NaN vs None),
    # which would sort rows differently and silently misalign the pairing.
    rows.sort(key=lambda r: r.get("content") or "")
    yt_op, yp_op, yt_f, yp_f, yt_c, yp_c = [], [], [], [], [], []
    for r in rows:
        gold_op = bool(r.get("true_opposition_detected", False))
        gold_f = list(r.get("true_frames") or [])
        gold_c = list(r.get("true_claims") or [])
        resp = r.get("response", "")
        failed = isinstance(resp, str) and resp.startswith("ERROR:")
        if not failed:
            pred = parse_response(resp)
            failed = (pred["opposition_detected"] is None
                      or pred["frames"] is None or pred["claims"] is None)
        yt_op.append(gold_op); yt_f.append(gold_f); yt_c.append(gold_c)
        if failed:
            yp_op.append(not gold_op)        # forced detection miss
            yp_f.append([SENTINEL]); yp_c.append([SENTINEL])
        else:
            yp_op.append(bool(pred["opposition_detected"]))
            yp_f.append(list(pred["frames"])); yp_c.append(list(pred["claims"]))
    return {"op": (yt_op, yp_op), "frames": (yt_f, yp_f), "claims": (yt_c, yp_c)}


# ---------------------------------------------------------------------------
# Per-row metric pieces
# ---------------------------------------------------------------------------

def per_row_f1(t_set, p_set):
    """Empty/empty → 1.0 (an agreed-empty prediction is correct; matches
    wind's `_samples_f1`)."""
    if not t_set and not p_set:
        return 1.0
    inter = len(t_set & p_set)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(p_set), inter / len(t_set)
    return 2 * prec * rec / (prec + rec)


def det_stats(yt_op, yp_op):
    """Per-row TP/FP/FN indicators for binary detection."""
    yt, yp = np.array(yt_op, bool), np.array(yp_op, bool)
    return (yt & yp).astype(float), (~yt & yp).astype(float), (yt & ~yp).astype(float)


def f1_rows(yt_list, yp_list):
    return np.array([per_row_f1(set(t), set(p)) for t, p in zip(yt_list, yp_list)])


def binary_f1(tp, fp, fn):
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


# ---------------------------------------------------------------------------
# Bootstrap CI via scipy (resample the given rows; BCa, percentile fallback)
# ---------------------------------------------------------------------------

def ci_over(rows, value_fn, cfg):
    """(point, lo, hi) for value_fn evaluated over `rows` (absolute indices),
    with a CI from resampling those rows with replacement."""
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


# ---------------------------------------------------------------------------
# Per-model scores and paired comparisons
# ---------------------------------------------------------------------------

def support_vocab(yt_lists, min_support):
    """Gold codes with support >= min_support, computed over the given rows
    (here: the opposition-only rows, where frames/claims apply). The vocab is
    fixed up front so bootstrap resampling indexes into stable per-label
    arrays — matching aqm/cards/evals/significance.py."""
    sup = Counter(l for row in yt_lists for l in row)
    return sorted([l for l, c in sup.items() if c >= min_support])


def label_mats(yt_lists, yp_lists, vocab):
    """Per-row, per-label TP/FP/FN matrices (n_rows x |vocab|), gold-only vocab.

    Predictions outside the vocab (rare codes below threshold, hallucinations,
    the parse-fail sentinel) are dropped, so a failed row contributes FNs on its
    gold labels — the same penalty the report applies."""
    idx = {l: j for j, l in enumerate(vocab)}

    def mat(rows):
        M = np.zeros((len(rows), len(vocab)))
        for i, r in enumerate(rows):
            for l in r:
                j = idx.get(l)
                if j is not None:
                    M[i, j] = 1.0
        return M

    Yt, P = mat(yt_lists), mat(yp_lists)
    return Yt * P, P * (1 - Yt), Yt * (1 - P)          # TP, FP, FN per row/label


def macro_fn(TP, FP, FN):
    """Value function: macro-F1 over `vocab` for a subset of row indices."""
    def f(ii):
        tp, fp, fn = TP[ii].sum(0), FP[ii].sum(0), FN[ii].sum(0)
        den = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den > 0)
        return float(f1.mean()) if f1.size else float("nan")
    return f


def _rows(yt_op):
    opp = np.flatnonzero(np.array(yt_op, bool))
    return np.arange(len(yt_op)), opp


def model_scores(slug, split, cfg):
    D = load(slug, split)
    yt_op, yp_op = D["op"]
    yt_f, yp_f = D["frames"]
    yt_c, yp_c = D["claims"]
    tp, fp, fn = det_stats(yt_op, yp_op)
    f1f, f1c = f1_rows(yt_f, yp_f), f1_rows(yt_c, yp_c)
    allr, opp = _rows(yt_op)
    opp_list = opp.tolist()
    # Macro vocab from opposition-only gold (support >= MIN_SUPPORT).
    vf = support_vocab([yt_f[i] for i in opp_list], MIN_SUPPORT)
    vc = support_vocab([yt_c[i] for i in opp_list], MIN_SUPPORT)
    f_macro = macro_fn(*label_mats(yt_f, yp_f, vf))
    c_macro = macro_fn(*label_mats(yt_c, yp_c, vc))
    det = lambda ii: binary_f1(tp[ii].sum(), fp[ii].sum(), fn[ii].sum())
    return {"n": len(yt_op), "n_opp": len(opp),
            "det":          ci_over(allr, det, cfg),
            "frames":       ci_over(opp, lambda ii: float(f1f[ii].mean()), cfg),
            "frames_macro": ci_over(opp, f_macro, cfg),
            "claims":       ci_over(opp, lambda ii: float(f1c[ii].mean()), cfg),
            "claims_macro": ci_over(opp, c_macro, cfg)}


def compare(split, slug_a, slug_b, cfg):
    A, B = load(slug_a, split), load(slug_b, split)
    yt_op = A["op"][0]                                   # gold from slug_a
    # Guard against row misalignment: gold must match positionally across the
    # two files (same task, same rows). A nonzero count means the sort key
    # failed to align them — results would be garbage.
    n_mis = sum(1 for x, y in zip(yt_op, B["op"][0]) if x != y)
    if n_mis:
        print(f"  WARNING: {n_mis}/{len(yt_op)} rows misaligned ({slug_a} vs "
              f"{slug_b}) — comparison invalid.", file=sys.stderr)
    yt_f, yt_c = A["frames"][0], A["claims"][0]          # shared canonical gold
    tpA, fpA, fnA = det_stats(yt_op, A["op"][1])
    tpB, fpB, fnB = det_stats(yt_op, B["op"][1])
    f1fA, f1fB = f1_rows(yt_f, A["frames"][1]), f1_rows(yt_f, B["frames"][1])
    f1cA, f1cB = f1_rows(yt_c, A["claims"][1]), f1_rows(yt_c, B["claims"][1])
    allr, opp = _rows(yt_op)
    opp_list = opp.tolist()
    vf = support_vocab([yt_f[i] for i in opp_list], MIN_SUPPORT)
    vc = support_vocab([yt_c[i] for i in opp_list], MIN_SUPPORT)
    f_macroA = macro_fn(*label_mats(yt_f, A["frames"][1], vf))
    f_macroB = macro_fn(*label_mats(yt_f, B["frames"][1], vf))
    c_macroA = macro_fn(*label_mats(yt_c, A["claims"][1], vc))
    c_macroB = macro_fn(*label_mats(yt_c, B["claims"][1], vc))

    def pack(rows, fa, fb):
        a, b = fa(rows), fb(rows)
        _, lo, hi = ci_over(rows, lambda ii: fa(ii) - fb(ii), cfg)
        return {"a": a, "b": b, "delta": a - b, "lo": lo, "hi": hi}

    return {
        "det":    pack(allr, lambda ii: binary_f1(tpA[ii].sum(), fpA[ii].sum(), fnA[ii].sum()),
                             lambda ii: binary_f1(tpB[ii].sum(), fpB[ii].sum(), fnB[ii].sum())),
        "frames":       pack(opp, lambda ii: float(f1fA[ii].mean()), lambda ii: float(f1fB[ii].mean())),
        "frames_macro": pack(opp, f_macroA, f_macroB),
        "claims":       pack(opp, lambda ii: float(f1cA[ii].mean()), lambda ii: float(f1cB[ii].mean())),
        "claims_macro": pack(opp, c_macroA, c_macroB),
    }


def verdict(M):
    if not (np.isfinite(M["lo"]) and np.isfinite(M["hi"])):
        return "n/a"
    if M["lo"] > 0:
        return "improves"
    if M["hi"] < 0:
        return "lower"
    return "comparable"


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_split(split, cfg):
    pct = int(round(cfg["confidence"] * 100))
    L = [f"# Wind results — {split} set\n",
         f"Three tasks: binary **opposition detection** (all rows), and "
         f"**frames**/**claims** multi-label tagging on **opposition-only** rows "
         f"(where they apply). For frames/claims we report **Samples-F1** "
         f"(per-row, support-independent — agreed-empty/non-opposition rows count "
         f"as correct) and **Macro-F1** restricted to gold codes with support ≥ "
         f"{MIN_SUPPORT} (CARDS convention; hallucinated codes dropped). Each "
         f"value is F1 with a {pct}% BCa bootstrap CI (`scipy.stats.bootstrap`, "
         f"{cfg['n_resamples']} resamples). Parse failures / API errors are "
         f"penalised, matching `generate_report.py`.\n",
         "## Model scores\n",
         f"| Model | Detection F1 | Frames Samples-F1 (opp) | Frames Macro-F1 (opp, sup≥{MIN_SUPPORT}) "
         f"| Claims Samples-F1 (opp) | Claims Macro-F1 (opp, sup≥{MIN_SUPPORT}) |",
         "|---|---|---|---|---|---|"]
    for name, slug in LINEUP:
        try:
            r = model_scores(slug, split, cfg)
        except FileNotFoundError:
            continue
        cell = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"
        L.append(f"| {name} | {cell(r['det'])} | {cell(r['frames'])} | {cell(r['frames_macro'])} "
                 f"| {cell(r['claims'])} | {cell(r['claims_macro'])} |")

    # Compute each comparison once, then render one sub-table per task.
    results = {}
    for group, pairs in COMPARISONS:
        for label, sa, sb in pairs:
            try:
                results[(group, label)] = compare(split, sa, sb, cfg)
            except FileNotFoundError:
                results[(group, label)] = None

    for key, disp in METRICS:
        L += [f"\n## Comparisons — {disp}\n",
              f"Gap = A − B with a {pct}% CI. `improves`/`lower` = CI clears 0; "
              f"`comparable` = CI includes 0.\n",
              f"| Comparison | A | B | Gap ({pct}% CI) | Verdict |",
              "|---|---|---|---|---|"]
        for group, pairs in COMPARISONS:
            rows = []
            for label, sa, sb in pairs:
                res = results[(group, label)]
                if res is None:
                    continue
                M = res[key]
                rows.append(f"| {label} | {M['a']:.3f} | {M['b']:.3f} | "
                            f"{M['delta']:+.3f} [{M['lo']:+.3f}, {M['hi']:+.3f}] | "
                            f"{verdict(M)} |")
            if rows:
                L.append(f"| **{group}** | | | | |")
                L += rows
    return "\n".join(L) + "\n"


def run_report(cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    split = cfg["split"]
    print(f"  computing {split} ...")
    (out_dir / f"summary_{split}.md").write_text(build_split(split, cfg))
    print(f"  -> {out_dir}/summary_{split}.md")


def run_pair(slug_a, slug_b, cfg):
    res = compare(cfg["split"], slug_a, slug_b, cfg)
    pct = int(round(cfg["confidence"] * 100))
    print(f"\n{slug_a}  vs  {slug_b}  ({cfg['split']})\n")
    print(f"  {'task':18s}  {'A':>7}  {'B':>7}  {'gap':>8}  {f'{pct}% CI':>22}  verdict")
    print("  " + "-" * 74)
    for key, disp in METRICS:
        M = res[key]
        ci = f"[{M['lo']:+.4f}, {M['hi']:+.4f}]"
        print(f"  {disp:18s}  {M['a']:>7.4f}  {M['b']:>7.4f}  "
              f"{M['delta']:+8.4f}  {ci:>22}  {verdict(M)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")
    sub.add_parser("report", help="write data/significance/summary_<split>.md (default)")
    pr = sub.add_parser("pair", help="one A vs B comparison, console output")
    pr.add_argument("slug_a")
    pr.add_argument("slug_b")

    for p in (ap, pr):
        p.add_argument("--split", choices=["test", "val"], default="test")
        p.add_argument("--n-resamples", type=int, default=9999)
        p.add_argument("--confidence", type=float, default=0.95)
        p.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "significance"))

    args = ap.parse_args()
    cfg = {"split": args.split, "n_resamples": args.n_resamples,
           "confidence": args.confidence, "seed": args.seed}
    if args.mode == "pair":
        run_pair(args.slug_a, args.slug_b, cfg)
    else:
        run_report(cfg, Path(args.out_dir))


if __name__ == "__main__":
    main()
