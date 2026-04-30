"""Paired bootstrap (BCa) + sign-flip permutation tests for CARDS results.

Reads two model result jsonls and reports samples/micro/macro F1 differences
together with a 95% BCa bootstrap CI on the difference and a sign-flip
permutation p-value (where applicable).

Per arXiv:2511.19794, claim a significant improvement only when the BCa CI
excludes 0 AND the permutation p-value is below alpha.

Usage:
    python evals/significance.py test cards-qwen35-27b-fp8 claude-opus-4-7
    python evals/significance.py twitter cards-qwen36-27b claude-opus-4-7

Optional flags: --level 3  --n-boot 2000  --n-perm 10000  --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generate_report import parse_response  # noqa: E402


# ---------------------------------------------------------------------------
# Per-row metric helpers
# ---------------------------------------------------------------------------

def per_row_f1(t_set: set, p_set: set) -> float:
    """sklearn samples-F1 convention: empty/empty → 0 (not 1).
    CARDS test/twitter have no empty-gold rows, so this is moot in practice."""
    if not t_set and not p_set:
        return 0.0
    inter = len(t_set & p_set)
    if inter == 0:
        return 0.0
    prec = inter / len(p_set)
    rec = inter / len(t_set)
    return 2 * prec * rec / (prec + rec)


def samples_f1(yt, yp) -> float:
    if not yt:
        return 0.0
    return float(np.mean([per_row_f1(set(t), set(p)) for t, p in zip(yt, yp)]))


def micro_f1(yt, yp, vocab: set[str]) -> float:
    tp = fp = fn = 0
    for t, p in zip(yt, yp):
        ts = set(t) & vocab
        ps = set(p) & vocab
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0


def macro_f1(yt, yp, vocab: list[str]) -> float:
    f1s = []
    for L in vocab:
        tp = sum(1 for t, p in zip(yt, yp) if L in t and L in p)
        fp = sum(1 for t, p in zip(yt, yp) if L not in t and L in p)
        fn = sum(1 for t, p in zip(yt, yp) if L in t and L not in p)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * P * R / (P + R) if P + R else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load(slug: str, split: str, level: int):
    path = ROOT / "data" / "results" / split / f"{slug}.jsonl"
    if not path.exists():
        sys.exit(f"missing: {path}")
    rows = [json.loads(l) for l in open(path)]
    # Align row order across files (threadpool writes in completion order).
    # `id` isn't unique on twitter (172/744 None, only 513 distinct), so sort
    # by the input text — which is unique per row in both splits.
    rows.sort(key=lambda r: r.get("text") or "")
    label_field = "labels" if split == "twitter" else "true_claims"
    yt, yp = [], []
    for r in rows:
        gold = list(r.get(label_field) or [])
        pred = parse_response(r.get("response", ""))
        yt.append(["_".join(c.split("_")[:level]) for c in gold])
        yp.append(["_".join(c.split("_")[:level]) for c in pred])
    return yt, yp


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------

def paired_bca_ci(yt, yp_a, yp_b, metric_fn, n_boot=2000, alpha=0.05, seed=42):
    n = len(yt)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)

    def stat(idx_arr):
        idx = np.asarray(idx_arr).astype(int)
        yt_s = [yt[i]  for i in idx]
        a_s  = [yp_a[i] for i in idx]
        b_s  = [yp_b[i] for i in idx]
        return metric_fn(yt_s, a_s) - metric_fn(yt_s, b_s)

    point = metric_fn(yt, yp_a) - metric_fn(yt, yp_b)
    res = bootstrap(
        (indices,), stat,
        n_resamples=n_boot, method="BCa",
        confidence_level=1 - alpha, random_state=rng,
        vectorized=False,
    )
    return point, float(res.confidence_interval.low), float(res.confidence_interval.high)


def sign_flip_pvalue(per_row_diffs: np.ndarray, n_perm=10000, seed=42) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(per_row_diffs.mean())
    n = len(per_row_diffs)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = np.abs((signs * per_row_diffs).mean(axis=1))
    return float((perm_means >= obs).mean())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def report(label_a, label_b, comps, n_boot, n_perm, alpha):
    print(f"\n{label_a}  vs  {label_b}")
    print(f"  n_boot={n_boot}  n_perm={n_perm}  alpha={alpha}\n")
    print(f"  {'metric':30s}  {'A':>7}  {'B':>7}  {'Δ':>8}  {'95% CI on Δ':>22}  {'p (sign-flip)':>14}  significant?")
    print(f"  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*22}  {'-'*14}  ------------")
    for name, a, b, delta, lo, hi, p in comps:
        ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
        ci_excludes_zero = (lo > 0) or (hi < 0)
        p_str = f"{p:.4f}" if p is not None else "  n/a"
        is_sig = ci_excludes_zero and (p is None or p < alpha)
        sig = "yes" if is_sig else ("border" if ci_excludes_zero else "no")
        print(f"  {name:30s}  {a:>7.4f}  {b:>7.4f}  {delta:+8.4f}  {ci_str:>22}  {p_str:>14}  {sig}")


def run(split, slug_a, slug_b, level, n_boot, n_perm, alpha):
    yt_a, ya = load(slug_a, split, level)
    yt_b, yb = load(slug_b, split, level)
    # Sanity: gold should match across the two files row-for-row.
    n_mismatch = sum(1 for x, y in zip(yt_a, yt_b) if sorted(x) != sorted(y))
    if n_mismatch:
        print(f"WARNING: {n_mismatch}/{len(yt_a)} gold mismatches across files —"
              f" results not directly comparable.", file=sys.stderr)
    yt = yt_a
    vocab = sorted({l for row in yt for l in row})

    comps = []
    pt, lo, hi = paired_bca_ci(yt, ya, yb, samples_f1, n_boot=n_boot)
    row_a = np.array([per_row_f1(set(t), set(p)) for t, p in zip(yt, ya)])
    row_b = np.array([per_row_f1(set(t), set(p)) for t, p in zip(yt, yb)])
    p = sign_flip_pvalue(row_a - row_b, n_perm=n_perm)
    comps.append(("Claims samples F1",
                  samples_f1(yt, ya), samples_f1(yt, yb), pt, lo, hi, p))

    mfn_micro = lambda t, q, V=set(vocab): micro_f1(t, q, V)
    pt, lo, hi = paired_bca_ci(yt, ya, yb, mfn_micro, n_boot=n_boot)
    comps.append(("Claims micro F1",
                  mfn_micro(yt, ya), mfn_micro(yt, yb), pt, lo, hi, None))

    mfn_macro = lambda t, q, V=vocab: macro_f1(t, q, V)
    pt, lo, hi = paired_bca_ci(yt, ya, yb, mfn_macro, n_boot=n_boot)
    comps.append(("Claims macro F1",
                  mfn_macro(yt, ya), mfn_macro(yt, yb), pt, lo, hi, None))

    report(f"{slug_a} ({split}, L{level})", slug_b, comps, n_boot, n_perm, alpha)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", choices=["test", "twitter"])
    ap.add_argument("slug_a")
    ap.add_argument("slug_b")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    run(args.split, args.slug_a, args.slug_b, args.level,
        args.n_boot, args.n_perm, args.alpha)


if __name__ == "__main__":
    main()
