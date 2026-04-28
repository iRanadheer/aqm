"""Parse YAML predictions and compute metrics.

Reads a results jsonl (produced by run_benchmark.py) that contains both the
ground truth (`true_opposition_detected`, `true_frames`, `true_claims`) and
the model's raw `response` string. Parses the response into structured
predictions, then computes detection, frames, and claims metrics.

Usage:
    python3 metrics.py
    python3 metrics.py --input data/results/predictions.jsonl --per-code
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = "data/results/predictions.jsonl"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(r"([NC]_\d+(?:_\d+)?)")
_LIST_RE_TMPL = r"{key}\s*:\s*(\[\s*\]|(?:\n\s*-\s*.+)+)"


def _extract_yaml_block(response: str) -> str:
    """Return the YAML block from a response string.

    Prefers content after </think>; then prefers fenced ```yaml ... ```; else
    returns the raw tail text.
    """
    if not isinstance(response, str):
        return ""
    tail = response.split("</think>", 1)[-1] if "</think>" in response else response
    m = re.search(r"```(?:yaml)?\s*(.*?)```", tail, re.DOTALL)
    return (m.group(1) if m else tail).strip()


def _extract_list(block: str, key: str) -> list[str]:
    m = re.search(_LIST_RE_TMPL.format(key=key), block)
    if not m:
        return []
    body = m.group(1)
    if body.strip() == "[]":
        return []
    codes = _CODE_RE.findall(body)
    # Dedupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def parse_response(response: str) -> dict:
    """Parse YAML output into {opposition_detected, frames, claims}."""
    block = _extract_yaml_block(response)
    parsed = {"opposition_detected": None, "frames": [], "claims": []}
    if not block:
        return parsed

    m = re.search(r"opposition_detected\s*:\s*(\w+)", block, re.IGNORECASE)
    if m:
        parsed["opposition_detected"] = m.group(1).lower() == "true"

    parsed["frames"] = _extract_list(block, "frames")
    parsed["claims"] = _extract_list(block, "claims")
    return parsed


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def detection_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    n = len(y_true)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
    }


def _samples_f1(y_true: list[list[str]], y_pred: list[list[str]]) -> float:
    """Per-sample F1, averaged. Empty-gold + empty-pred counts as 1.0 (perfect).

    sklearn's ``f1_score(average='samples', zero_division=0)`` scores empty/empty
    rows as 0, which is wrong here — an agreed-empty prediction is correct.
    """
    if not y_true:
        return 0.0
    total = 0.0
    for t, p in zip(y_true, y_pred):
        t_set, p_set = set(t), set(p)
        if not t_set and not p_set:
            total += 1.0
            continue
        tp = len(t_set & p_set)
        if tp == 0:
            continue  # F1 = 0
        prec = tp / len(p_set)
        rec = tp / len(t_set)
        total += 2 * prec * rec / (prec + rec)
    return total / len(y_true)


def multilabel_metrics(
    y_true: list[list[str]], y_pred: list[list[str]]
) -> dict:
    labels = set()
    for lab in y_true + y_pred:
        labels.update(lab)
    if not labels:
        return {"note": "no labels observed"}
    mlb = MultiLabelBinarizer()
    mlb.fit([sorted(labels)])
    yt = mlb.transform(y_true)
    yp = mlb.transform(y_pred)
    return {
        "samples_f1": round(_samples_f1(y_true, y_pred), 3),
        "macro_f1": round(f1_score(yt, yp, average="macro", zero_division=0), 3),
        "micro_f1": round(f1_score(yt, yp, average="micro", zero_division=0), 3),
        "micro_precision": round(precision_score(yt, yp, average="micro", zero_division=0), 3),
        "micro_recall": round(recall_score(yt, yp, average="micro", zero_division=0), 3),
        "exact_match": round(accuracy_score(yt, yp), 3),
        "hamming_loss": round(hamming_loss(yt, yp), 3),
    }


def per_code_breakdown(
    y_true: list[list[str]], y_pred: list[list[str]]
) -> list[tuple[str, int, int, int, float]]:
    """Return [(code, support, fn, fp, f1), ...] sorted by (fn+fp) descending."""
    codes = set()
    for lab in y_true + y_pred:
        codes.update(lab)
    out = []
    for code in sorted(codes):
        support = sum(1 for lab in y_true if code in lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if code in t and code not in p)
        fp = sum(1 for t, p in zip(y_true, y_pred) if code not in t and code in p)
        tp = support - fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append((code, support, fn, fp, f1))
    out.sort(key=lambda x: (-(x[2] + x[3]), x[0]))
    return out


def _print_per_code(
    title: str,
    y_true: list[list[str]],
    y_pred: list[list[str]],
    min_support: int,
) -> None:
    rows = per_code_breakdown(y_true, y_pred)
    shown = [r for r in rows if r[1] >= min_support and (r[2] or r[3])]
    print(f"=== {title} per-code (support >= {min_support}, ordered by FN+FP desc) ===")
    print(f"  {'code':<10} {'support':>7} {'FN':>5} {'FP':>5} {'F1':>7}")
    for code, support, fn, fp, f1 in shown:
        print(f"  {code:<10} {support:>7} {fn:>5} {fp:>5} {f1:>7.3f}")
    kept_for_macro = [r for r in rows if r[1] >= min_support]
    if kept_for_macro:
        macro = sum(r[4] for r in kept_for_macro) / len(kept_for_macro)
        print(f"  → macro_F1 over {len(kept_for_macro)} codes (support >= {min_support}): {macro:.3f}")
    print()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT, help="Results jsonl (repo-relative if not absolute).")
    p.add_argument("--per-code", action="store_true", help="Print per-code FN/FP/F1 breakdown for frames and claims.")
    p.add_argument("--min-support", type=int, default=0,
                   help="Minimum gold support (val-set occurrences) to include a code in per-code output and filtered macro_F1.")
    p.add_argument("--errors", action="store_true", help="Print rows that failed to parse.")
    args = p.parse_args()

    path = Path(args.input)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        sys.exit(f"Input not found: {path}")

    rows = [json.loads(l) for l in open(path)]

    parse_fail = 0
    api_error = 0
    y_true_opp: list[bool] = []
    y_pred_opp: list[bool] = []
    y_true_frames: list[list[str]] = []
    y_pred_frames: list[list[str]] = []
    y_true_claims: list[list[str]] = []
    y_pred_claims: list[list[str]] = []
    failed_rows: list[dict] = []

    for r in rows:
        resp = r.get("response", "")
        if isinstance(resp, str) and resp.startswith("ERROR:"):
            api_error += 1
            failed_rows.append(r)
            continue
        pred = parse_response(resp)
        if pred["opposition_detected"] is None:
            parse_fail += 1
            failed_rows.append(r)
            continue
        y_true_opp.append(bool(r.get("true_opposition_detected", False)))
        y_pred_opp.append(bool(pred["opposition_detected"]))
        y_true_frames.append(list(r.get("true_frames") or []))
        y_pred_frames.append(list(pred["frames"]))
        y_true_claims.append(list(r.get("true_claims") or []))
        y_pred_claims.append(list(pred["claims"]))

    print(f"Rows: {len(rows)}  |  API errors: {api_error}  |  parse failures: {parse_fail}")
    print(f"Evaluated: {len(y_true_opp)}\n")

    print("=== DETECTION (binary) ===")
    for k, v in detection_metrics(y_true_opp, y_pred_opp).items():
        print(f"  {k}: {v}")
    print()

    print("=== FRAMES — all rows ===")
    for k, v in multilabel_metrics(y_true_frames, y_pred_frames).items():
        print(f"  {k}: {v}")
    print()

    print("=== CLAIMS — all rows ===")
    for k, v in multilabel_metrics(y_true_claims, y_pred_claims).items():
        print(f"  {k}: {v}")
    print()

    # Opposition-only view: restrict to rows where gold says opposition_detected=true.
    # For comparison to evaluations where non-opposition rows were labeled
    # [C_0_0] and contributed trivial wins to every metric.
    opp_idx = [i for i, t in enumerate(y_true_opp) if t]
    yt_f_opp = [y_true_frames[i] for i in opp_idx]
    yp_f_opp = [y_pred_frames[i] for i in opp_idx]
    yt_c_opp = [y_true_claims[i] for i in opp_idx]
    yp_c_opp = [y_pred_claims[i] for i in opp_idx]

    print(f"=== FRAMES — opposition-only (n={len(opp_idx)}) ===")
    for k, v in multilabel_metrics(yt_f_opp, yp_f_opp).items():
        print(f"  {k}: {v}")
    print()

    print(f"=== CLAIMS — opposition-only (n={len(opp_idx)}) ===")
    for k, v in multilabel_metrics(yt_c_opp, yp_c_opp).items():
        print(f"  {k}: {v}")
    print()

    if args.per_code or args.min_support > 0:
        _print_per_code("FRAMES", y_true_frames, y_pred_frames, args.min_support)
        _print_per_code("CLAIMS", y_true_claims, y_pred_claims, args.min_support)

    if args.errors and failed_rows:
        print("=== FAILED ROWS ===")
        for r in failed_rows[:10]:
            print(f"  itemId={r.get('itemId')}  content={str(r.get('content',''))[:80]!r}")
            resp = r.get("response", "")
            tail = resp[-200:] if isinstance(resp, str) else repr(resp)
            print(f"    response tail: {tail!r}")
        if len(failed_rows) > 10:
            print(f"  ... {len(failed_rows) - 10} more")


if __name__ == "__main__":
    main()
