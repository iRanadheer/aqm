"""Bootstrap confidence intervals for CARDS results — the simple, defensible view.

Headline metric is **samples-F1** (per-document tagging quality); micro- and
macro-F1 are reported alongside for completeness (macro is lower because rare
categories are harder). Everything is a 95% **BCa bootstrap confidence
interval**, computed with `scipy.stats.bootstrap` — no hand-rolled resampling,
no p-values, no corrections. This follows the recommendation to report the
difference and its confidence interval rather than significance-test verdicts
(Ulmer et al., LREC 2022; Koehn 2004 for the paired bootstrap).

Model comparisons report the gap (A − B) and its CI, labelled:
    improves    — CI entirely above 0
    lower       — CI entirely below 0
    comparable  — CI includes 0 (too close to call)

Usage:
    python evals/significance.py              # -> data/significance/summary_{test,twitter}.md
    python evals/significance.py pair test cards-qwen35-27b claude-opus-4-7

Flags: --level 3  --min-support 3  --n-resamples 9999  --confidence 0.95
       --seed 42  --out-dir data/significance

(We use scipy because it returns confidence intervals. `deep-significance`
[deepsig] is the NLP-specific alternative, but it returns p-values / ASO scores,
not CIs, so it doesn't fit this CI-based reporting.)
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

from generate_report import load_canonical_gold, parse_response  # noqa: E402

OUT_DIR = ROOT / "data" / "significance"
METRICS = [("samples", "Samples-F1"), ("micro", "Micro-F1"), ("macro", "Macro-F1")]


# ---------------------------------------------------------------------------
# Final reported lineup and the comparisons we care about
# ---------------------------------------------------------------------------

LINEUP = [
    ("Qwen3.5-4B (base, zero-shot)",  "qwen35-4b-base"),
    ("Qwen3.5-9B (base, zero-shot)",  "qwen35-9b-base"),
    ("Qwen3.5-27B (base, zero-shot)", "qwen35-27b-base"),
    ("CARDS-Qwen3.5-4B (ours)",       "cards-qwen35-4b"),
    ("CARDS-Qwen3.5-9B (ours)",       "cards-qwen35-9b"),
    ("CARDS-Qwen3.5-27B (ours)",      "cards-qwen35-27b"),
    ("GPT-4o-mini (zero-shot)",       "gpt-4o-mini"),
    ("CARDS-mini-opus (ours)",        "cards-mini-opus"),
    ("Claude Opus 4.7 (zero-shot)",   "claude-opus-4-7"),
    ("GPT-5.5 (zero-shot)",           "gpt-5-5"),
]

# Each comparison is gap = A - B on the chosen metric. On the test split the
# gold embedded in result files is ignored — both sides score against the
# canonical cards_test.jsonl gold (see load), so the order of slugs is free.
COMPARISONS = [
    ("Fine-tuning helps (open)", [
        ("CARDS-4B vs base",  "cards-qwen35-4b",  "qwen35-4b-base"),
        ("CARDS-9B vs base",  "cards-qwen35-9b",  "qwen35-9b-base"),
        ("CARDS-27B vs base", "cards-qwen35-27b", "qwen35-27b-base")]),
    ("Fine-tuning helps (closed)", [
        ("CARDS-mini-opus vs GPT-4o-mini", "cards-mini-opus", "gpt-4o-mini")]),
    ("Does scale help?", [
        ("CARDS-9B vs 4B",  "cards-qwen35-9b",  "cards-qwen35-4b"),
        ("CARDS-27B vs 9B", "cards-qwen35-27b", "cards-qwen35-9b")]),
    ("RECoT format helps", [
        ("CARDS-4B vs No-RECoT",  "cards-qwen35-4b",  "cards-qwen35-4b-norecot"),
        ("CARDS-9B vs No-RECoT",  "cards-qwen35-9b",  "cards-qwen35-9b-norecot"),
        ("CARDS-27B vs No-RECoT", "cards-qwen35-27b", "cards-qwen35-27b-norecot")]),
    ("Vs frontier APIs", [
        ("CARDS-27B vs Claude Opus 4.7",       "cards-qwen35-27b", "claude-opus-4-7"),
        ("CARDS-27B vs GPT-5.5",               "cards-qwen35-27b", "gpt-5-5"),
        ("CARDS-mini-opus vs Claude Opus 4.7", "cards-mini-opus",  "claude-opus-4-7"),
        ("CARDS-mini-opus vs GPT-5.5",         "cards-mini-opus",  "gpt-5-5")]),
    ("FP8 quantization", [
        ("CARDS-27B FP8 vs full", "cards-qwen35-27b-fp8", "cards-qwen35-27b")]),
]

VERDICT_TEX = {"improves": "higher", "lower": "lower", "comparable": "comparable"}

# Comparisons as reported in the chapter appendix (samples-F1, L3).
# RECoT rows use the no-think baseline — the no-RECoT models' natural
# inference mode — to match the chapter's ablation table.
TEX_COMPARISONS = {
    "test": [
        ("Fine-tuning (open-source)", [
            (r"CARDS-Qwen3.5-4B vs.\ base",  "cards-qwen35-4b",  "qwen35-4b-base"),
            (r"CARDS-Qwen3.5-9B vs.\ base",  "cards-qwen35-9b",  "qwen35-9b-base"),
            (r"CARDS-Qwen3.5-27B vs.\ base", "cards-qwen35-27b", "qwen35-27b-base")]),
        ("Fine-tuning (closed-source)", [
            (r"CARDS-mini-opus vs.\ GPT-4o-mini", "cards-mini-opus", "gpt-4o-mini")]),
        ("Scaling", [
            (r"CARDS-Qwen3.5-9B vs.\ 4B",  "cards-qwen35-9b",  "cards-qwen35-4b"),
            (r"CARDS-Qwen3.5-27B vs.\ 9B", "cards-qwen35-27b", "cards-qwen35-9b")]),
        ("RECoT ablation", [
            (r"CARDS-Qwen3.5-4B vs.\ no RECoT",  "cards-qwen35-4b",  "cards-qwen35-4b-norecot-nothink"),
            (r"CARDS-Qwen3.5-9B vs.\ no RECoT",  "cards-qwen35-9b",  "cards-qwen35-9b-norecot-nothink"),
            (r"CARDS-Qwen3.5-27B vs.\ no RECoT", "cards-qwen35-27b", "cards-qwen35-27b-norecot-nothink")]),
        ("Versus frontier models", [
            (r"CARDS-Qwen3.5-27B vs.\ Claude Opus 4.7", "cards-qwen35-27b", "claude-opus-4-7"),
            (r"CARDS-Qwen3.5-27B vs.\ GPT-5.5",         "cards-qwen35-27b", "gpt-5-5"),
            (r"CARDS-mini-opus vs.\ Claude Opus 4.7",   "cards-mini-opus",  "claude-opus-4-7"),
            (r"CARDS-mini-opus vs.\ GPT-5.5",           "cards-mini-opus",  "gpt-5-5")]),
        (r"Open vs.\ closed fine-tunes", [
            (r"CARDS-mini-opus vs.\ CARDS-Qwen3.5-27B", "cards-mini-opus", "cards-qwen35-27b")]),
        ("Quantization", [
            (r"CARDS-Qwen3.5-27B FP8 vs.\ BF16", "cards-qwen35-27b-fp8", "cards-qwen35-27b")]),
    ],
}
# twitter: same lineup — groups whose files are absent on that split are
# skipped by the FileNotFoundError handling below.
TEX_COMPARISONS["twitter"] = TEX_COMPARISONS["test"]


# ---------------------------------------------------------------------------
# Loader (rows aligned by unique input text; labels truncated + deduped)
# ---------------------------------------------------------------------------

_CANONICAL_GOLD = None


def canonical_gold():
    global _CANONICAL_GOLD
    if _CANONICAL_GOLD is None:
        _CANONICAL_GOLD = load_canonical_gold()
    return _CANONICAL_GOLD


def load(slug, split, level):
    path = ROOT / "data" / "results" / split / f"{slug}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    rows = [json.loads(l) for l in open(path)]
    # Test split: ignore the gold embedded in result files (stale for 67 rows
    # in several of them) and re-pair against canonical cards_test.jsonl by text.
    gold_map = canonical_gold() if split == "test" else None
    if gold_map is not None:
        rows = [r for r in rows if r["text"] in gold_map]
    rows.sort(key=lambda r: r.get("text") or "")          # align across files
    label_field = "labels" if split == "twitter" else "true_claims"
    require_think = "norecot" not in slug.lower() and "nothink" not in slug.lower()
    yt, yp = [], []
    for r in rows:
        gold = gold_map[r["text"]] if gold_map is not None else list(r.get(label_field) or [])
        pred = parse_response(r.get("response", ""), require_think=require_think)
        yt.append(list(dict.fromkeys("_".join(c.split("_")[:level]) for c in gold)))
        yp.append(list(dict.fromkeys("_".join(c.split("_")[:level]) for c in pred)))
    return yt, yp


# ---------------------------------------------------------------------------
# Metrics as functions of row indices (so scipy can resample by index)
#
# Each metric is a function of summed per-row pieces, so a metric over any
# subset of rows is computed by indexing precomputed arrays — which is exactly
# what scipy.stats.bootstrap needs to resample efficiently.
# ---------------------------------------------------------------------------

def row_stats(yt, yp, vocab):
    idx = {l: j for j, l in enumerate(vocab)}

    def mat(rows):
        M = np.zeros((len(rows), len(vocab)))
        for i, r in enumerate(rows):
            for l in r:
                if l in idx:
                    M[i, idx[l]] = 1.0
        return M

    Yt, P = mat(yt), mat(yp)
    inter = (Yt * P).sum(1)
    den = Yt.sum(1) + P.sum(1)
    return {
        "f1": np.divide(2 * inter, den, out=np.zeros_like(den), where=den > 0),
        "tp": inter, "fp": (P * (1 - Yt)).sum(1), "fn": (Yt * (1 - P)).sum(1),
        "TP": Yt * P, "FP": P * (1 - Yt), "FN": Yt * (1 - P),
    }


def _f1(tp, fp, fn):
    den = np.asarray(2 * tp + fp + fn, dtype=float)
    return np.divide(2 * np.asarray(tp, dtype=float), den,
                     out=np.zeros_like(den), where=den > 0)


def metric(s, ii, name):
    if name == "samples":
        return float(s["f1"][ii].mean())
    if name == "micro":
        return float(_f1(s["tp"][ii].sum(), s["fp"][ii].sum(), s["fn"][ii].sum()))
    return float(_f1(s["TP"][ii].sum(0), s["FP"][ii].sum(0), s["FN"][ii].sum(0)).mean())


# ---------------------------------------------------------------------------
# Bootstrap CI via scipy (BCa, percentile fallback if degenerate)
# ---------------------------------------------------------------------------

def boot_ci(stat_of_idx, n, n_resamples, confidence, seed):
    """95% (or `confidence`) BCa CI for a statistic that takes an index array."""
    idx = np.arange(n)

    def statistic(resampled):
        return stat_of_idx(resampled.astype(np.intp))

    for method in ("BCa", "percentile"):
        res = bootstrap((idx,), statistic, n_resamples=n_resamples,
                        confidence_level=confidence, method=method,
                        rng=np.random.default_rng(seed), vectorized=False)
        lo, hi = res.confidence_interval.low, res.confidence_interval.high
        if np.isfinite(lo) and np.isfinite(hi):
            return float(lo), float(hi)
    return float("nan"), float("nan")


def support_vocab(yt, min_support):
    sup = Counter(l for row in yt for l in row)
    return sorted([l for l, c in sup.items() if c >= min_support])


def filter_to(rows, vset):
    return [[l for l in r if l in vset] for r in rows]


# ---------------------------------------------------------------------------
# Per-model scores and paired comparisons
# ---------------------------------------------------------------------------

def model_scores(slug, split, level, min_support, n_resamples, confidence, seed):
    yt, yp = load(slug, split, level)
    vocab = support_vocab(yt, min_support)
    vset = set(vocab)
    s = row_stats(filter_to(yt, vset), filter_to(yp, vset), vocab)
    n = len(yt)
    idx = np.arange(n)
    out = {"n": n}
    for name, _ in METRICS:
        point = metric(s, idx, name)
        lo, hi = boot_ci(lambda ii, nm=name: metric(s, ii, nm),
                         n, n_resamples, confidence, seed)
        out[name] = (point, lo, hi)
    return out


def compare(split, slug_a, slug_b, level, min_support, n_resamples, confidence, seed):
    yt, ya = load(slug_a, split, level)
    _, yb = load(slug_b, split, level)              # same canonical gold both sides
    vocab = support_vocab(yt, min_support)
    vset = set(vocab)
    yt = filter_to(yt, vset)
    sA = row_stats(yt, filter_to(ya, vset), vocab)
    sB = row_stats(yt, filter_to(yb, vset), vocab)
    n = len(yt)
    idx = np.arange(n)
    out = {"n": n}
    for name, _ in METRICS:
        a, b = metric(sA, idx, name), metric(sB, idx, name)
        lo, hi = boot_ci(lambda ii, nm=name: metric(sA, ii, nm) - metric(sB, ii, nm),
                         n, n_resamples, confidence, seed)
        out[name] = {"a": a, "b": b, "delta": a - b, "lo": lo, "hi": hi}
    return out


def verdict(M):
    if M["lo"] > 0:
        return "improves"
    if M["hi"] < 0:
        return "lower"
    return "comparable"


def cached_compare(cache, split, slug_a, slug_b, cfg):
    """compare() memoized on (split, slug_a, slug_b) — the md summary and the
    .tex appendix share most pairs, so each BCa run happens once."""
    key = (split, slug_a, slug_b)
    if key not in cache:
        cache[key] = compare(split, slug_a, slug_b, cfg["level"], cfg["min_support"],
                             cfg["n_resamples"], cfg["confidence"], cfg["seed"])
    return cache[key]


# ---------------------------------------------------------------------------
# Markdown report (one file per split)
# ---------------------------------------------------------------------------

def build_split(split, cfg, cache):
    pct = int(round(cfg["confidence"] * 100))
    L = [f"# CARDS results — {split} set\n",
         f"Headline metric: **samples-F1** (how well each document is tagged), "
         f"with a {pct}% BCa bootstrap confidence interval "
         f"(`scipy.stats.bootstrap`, {cfg['n_resamples']} resamples). Micro/macro "
         f"shown for completeness; macro is lower because rare categories are "
         f"harder.\n",
         "## Model scores\n",
         f"| Model | Samples-F1 ({pct}% CI) | Micro-F1 | Macro-F1 |",
         "|---|---|---|---|"]
    for name, slug in LINEUP:
        try:
            r = model_scores(slug, split, cfg["level"], cfg["min_support"],
                             cfg["n_resamples"], cfg["confidence"], cfg["seed"])
        except FileNotFoundError:
            continue
        sp, mi, ma = r["samples"], r["micro"], r["macro"]
        L.append(f"| {name} | {sp[0]:.3f} [{sp[1]:.3f}, {sp[2]:.3f}] | "
                 f"{mi[0]:.3f} | {ma[0]:.3f} |")

    L += [f"\n## Comparisons (samples-F1)\n",
          f"Gap = A − B with a {pct}% CI. `improves`/`lower` = CI clears 0; "
          f"`comparable` = CI includes 0 (too close to call).\n",
          f"| Comparison | A | B | Gap ({pct}% CI) | Verdict |",
          "|---|---|---|---|---|"]
    for group, pairs in COMPARISONS:
        rows = []
        for label, sa, sb in pairs:
            try:
                res = cached_compare(cache, split, sa, sb, cfg)
            except FileNotFoundError:
                continue
            M = res["samples"]
            rows.append(f"| {label} | {M['a']:.3f} | {M['b']:.3f} | "
                        f"{M['delta']:+.3f} [{M['lo']:+.3f}, {M['hi']:+.3f}] | "
                        f"{verdict(M)} |")
        if rows:
            L.append(f"| **{group}** | | | | |")
            L += rows
    return "\n".join(L) + "\n"


def build_tex(split, cfg, cache):
    """LaTeX row fragment for the chapter appendix (samples-F1 gap + CI).

    Rows only — main.tex owns the table scaffolding and \\inputs this."""
    lines = []
    for group, pairs in TEX_COMPARISONS[split]:
        rows = []
        for label, sa, sb in pairs:
            try:
                res = cached_compare(cache, split, sa, sb, cfg)
            except FileNotFoundError:
                continue
            M = res["samples"]
            rows.append(f"{label} & ${M['delta']:+.3f}$ $[{M['lo']:+.3f}, {M['hi']:+.3f}]$"
                        f" & {VERDICT_TEX[verdict(M)]} \\\\")
        if rows:
            if lines:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{group}}}}} \\")
            lines += rows
    return "\n".join(lines) + "\n"


def run_report(cfg, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("test", "twitter"):
        print(f"  computing {split} ...")
        cache = {}
        (out_dir / f"summary_{split}.md").write_text(build_split(split, cfg, cache))
        print(f"  -> {out_dir}/summary_{split}.md")
        (out_dir / f"sig_{split}.tex").write_text(build_tex(split, cfg, cache))
        print(f"  -> {out_dir}/sig_{split}.tex")


def run_pair(split, slug_a, slug_b, cfg):
    res = compare(split, slug_a, slug_b, cfg["level"], cfg["min_support"],
                  cfg["n_resamples"], cfg["confidence"], cfg["seed"])
    pct = int(round(cfg["confidence"] * 100))
    print(f"\n{slug_a}  vs  {slug_b}  ({split}, n={res['n']})\n")
    print(f"  {'metric':12s}  {'A':>7}  {'B':>7}  {'gap':>8}  {f'{pct}% CI':>22}  verdict")
    print("  " + "-" * 70)
    for name, disp in METRICS:
        M = res[name]
        ci = f"[{M['lo']:+.4f}, {M['hi']:+.4f}]"
        print(f"  {disp:12s}  {M['a']:>7.4f}  {M['b']:>7.4f}  "
              f"{M['delta']:+8.4f}  {ci:>22}  {verdict(M)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")
    sub.add_parser("report", help="write data/significance/summary_{test,twitter}.md (default)")
    pr = sub.add_parser("pair", help="one A vs B comparison, console output")
    pr.add_argument("split", choices=["test", "twitter"])
    pr.add_argument("slug_a")
    pr.add_argument("slug_b")

    for p in (ap, pr):
        p.add_argument("--level", type=int, default=3)
        p.add_argument("--min-support", type=int, default=3)
        p.add_argument("--n-resamples", type=int, default=9999)
        p.add_argument("--confidence", type=float, default=0.95)
        p.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(OUT_DIR))

    args = ap.parse_args()
    cfg = {"level": args.level, "min_support": args.min_support,
           "n_resamples": args.n_resamples, "confidence": args.confidence,
           "seed": args.seed}
    if args.mode == "pair":
        run_pair(args.split, args.slug_a, args.slug_b, cfg)
    else:
        run_report(cfg, Path(args.out_dir))


if __name__ == "__main__":
    main()
