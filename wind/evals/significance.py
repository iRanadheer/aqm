"""Paired bootstrap (BCa) + sign-flip permutation tests for wind results.

Reads two model result jsonls and reports the headline F1 differences
together with a 95% BCa bootstrap CI on the difference and a sign-flip
permutation p-value (where applicable).

Per arXiv:2511.19794, claim a significant improvement only when the BCa
CI excludes 0 AND the permutation p-value is below alpha.

Usage:
    python evals/significance.py windy-qwen35-27b-fp8 claude-opus-4-7
    python evals/significance.py windy-qwen35-27b-fp8 openai-gpt-5-5

Optional flags: --n-boot 2000  --n-perm 10000  --alpha 0.05  --seed 42
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

SENTINEL = "__PARSE_FAIL__"


# ---------------------------------------------------------------------------
# Per-row metric helpers
# ---------------------------------------------------------------------------

def per_row_f1(t_set: set, p_set: set) -> float:
    """Empty/empty → 1.0 (matches wind's `_samples_f1`)."""
    if not t_set and not p_set:
        return 1.0
    inter = len(t_set & p_set)
    if inter == 0:
        return 0.0
    prec = inter / len(p_set)
    rec = inter / len(t_set)
    return 2 * prec * rec / (prec + rec)


def samples_f1(yt: list[list[str]], yp: list[list[str]]) -> float:
    if not yt:
        return 0.0
    return float(np.mean([per_row_f1(set(t), set(p)) for t, p in zip(yt, yp)]))


def micro_f1(yt: list[list[str]], yp: list[list[str]], vocab: set[str]) -> float:
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


def macro_f1(yt: list[list[str]], yp: list[list[str]], vocab: list[str]) -> float:
    f1s = []
    for L in vocab:
        tp = sum(1 for t, p in zip(yt, yp) if L in t and L in p)
        fp = sum(1 for t, p in zip(yt, yp) if L not in t and L in p)
        fn = sum(1 for t, p in zip(yt, yp) if L in t and L not in p)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * P * R / (P + R) if P + R else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def detection_f1(yt: list[bool], yp: list[bool]) -> float:
    tp = sum(1 for t, p in zip(yt, yp) if t and p)
    fp = sum(1 for t, p in zip(yt, yp) if not t and p)
    fn = sum(1 for t, p in zip(yt, yp) if t and not p)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0


# ---------------------------------------------------------------------------
# Loader — applies the same parse-failure penalty the report does.
# ---------------------------------------------------------------------------

def load(slug: str, split: str = "test") -> dict:
    path = ROOT / "data" / "results" / split / f"{slug}.jsonl"
    if not path.exists():
        sys.exit(f"missing: {path}")
    rows = [json.loads(l) for l in open(path)]
    # Result files are written in completion order from the threadpool, so
    # rows aren't in a consistent order across models. Sort by (itemId,
    # content); itemId has 1 duplicate pair (both None) and content is
    # unique per row, so the tuple is unique.
    rows.sort(key=lambda r: ((r.get("itemId") or ""), r.get("content") or ""))
    yt_op, yp_op, yt_f, yp_f, yt_c, yp_c = [], [], [], [], [], []
    for r in rows:
        gold_op = bool(r.get("true_opposition_detected", False))
        gold_f = list(r.get("true_frames") or [])
        gold_c = list(r.get("true_claims") or [])
        resp = r.get("response", "")
        failed = False
        if isinstance(resp, str) and resp.startswith("ERROR:"):
            failed = True
        else:
            pred = parse_response(resp)
            if (pred["opposition_detected"] is None
                    or pred["frames"] is None
                    or pred["claims"] is None):
                failed = True
        yt_op.append(gold_op)
        yt_f.append(gold_f)
        yt_c.append(gold_c)
        if failed:
            yp_op.append(not gold_op)
            yp_f.append([SENTINEL])
            yp_c.append([SENTINEL])
        else:
            yp_op.append(bool(pred["opposition_detected"]))
            yp_f.append(list(pred["frames"]))
            yp_c.append(list(pred["claims"]))
    return {"op": (yt_op, yp_op), "frames": (yt_f, yp_f), "claims": (yt_c, yp_c)}


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------

def paired_bca_ci(yt, yp_a, yp_b, metric_fn, n_boot=2000, alpha=0.05, seed=42):
    """Paired BCa bootstrap CI on metric_fn(yt, yp_a) - metric_fn(yt, yp_b)."""
    n = len(yt)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)

    # scipy.stats.bootstrap will resample `indices` with replacement and pass
    # the resampled array to `stat`. Same indices are used for both A and B
    # within a single resample (paired).
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
    """Two-sided sign-flip permutation test on per-row deltas."""
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


def run(slug_a, slug_b, n_boot, n_perm, alpha):
    A = load(slug_a)
    B = load(slug_b)
    yt_op = A["op"][0]
    yp_a_op, yp_b_op = A["op"][1], B["op"][1]

    comps = []

    # Detection F1.
    pt, lo, hi = paired_bca_ci(yt_op, yp_a_op, yp_b_op, detection_f1, n_boot=n_boot)
    # Per-row binary correctness diff for the sign-flip.
    diffs_op = np.array(
        [(int(t == a) - int(t == b)) for t, a, b in zip(yt_op, yp_a_op, yp_b_op)],
        dtype=float,
    )
    p = sign_flip_pvalue(diffs_op, n_perm=n_perm)
    comps.append(("Detection F1",
                  detection_f1(yt_op, yp_a_op), detection_f1(yt_op, yp_b_op),
                  pt, lo, hi, p))

    # Frames + Claims, all-rows + opp-only.
    opp_idx = [i for i, t in enumerate(yt_op) if t]
    for key in ("frames", "claims"):
        yt_full, ya_full = A[key]
        _,        yb_full = B[key]
        vocab = sorted({l for row in yt_full for l in row})
        for view, idx in (("all", list(range(len(yt_full)))), ("opp", opp_idx)):
            yt_v = [yt_full[i] for i in idx]
            ya_v = [ya_full[i] for i in idx]
            yb_v = [yb_full[i] for i in idx]

            # Samples F1 — has per-row signal.
            pt, lo, hi = paired_bca_ci(yt_v, ya_v, yb_v, samples_f1, n_boot=n_boot)
            row_a = np.array([per_row_f1(set(t), set(p)) for t, p in zip(yt_v, ya_v)])
            row_b = np.array([per_row_f1(set(t), set(p)) for t, p in zip(yt_v, yb_v)])
            p = sign_flip_pvalue(row_a - row_b, n_perm=n_perm)
            comps.append((f"{key} samples F1 ({view})",
                          samples_f1(yt_v, ya_v), samples_f1(yt_v, yb_v),
                          pt, lo, hi, p))

            # Micro F1 — global, bootstrap only.
            mfn = lambda t, q, V=set(vocab): micro_f1(t, q, V)
            pt, lo, hi = paired_bca_ci(yt_v, ya_v, yb_v, mfn, n_boot=n_boot)
            comps.append((f"{key} micro F1 ({view})",
                          mfn(yt_v, ya_v), mfn(yt_v, yb_v),
                          pt, lo, hi, None))

            # Macro F1 — global, bootstrap only.
            mfn = lambda t, q, V=vocab: macro_f1(t, q, V)
            pt, lo, hi = paired_bca_ci(yt_v, ya_v, yb_v, mfn, n_boot=n_boot)
            comps.append((f"{key} macro F1 ({view})",
                          mfn(yt_v, ya_v), mfn(yt_v, yb_v),
                          pt, lo, hi, None))

    report(slug_a, slug_b, comps, n_boot, n_perm, alpha)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slug_a")
    ap.add_argument("slug_b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    run(args.slug_a, args.slug_b, args.n_boot, args.n_perm, args.alpha)


if __name__ == "__main__":
    main()
