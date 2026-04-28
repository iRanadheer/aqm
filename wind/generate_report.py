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

import json
import re
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


def _extract_list(block: str, key: str) -> list[str] | None:
    """Return the parsed list, [] if the key is explicitly empty
    (`key: []`), or None if the key is absent (parse failure).

    The None vs [] distinction matters: with the lenient `[]`-on-absent
    behavior, a malformed response that drops the `frames`/`claims` key
    silently gets credit on rows where gold also has [] (typically
    non-opposition rows). Returning None lets the driver treat absent
    keys as parse failures.
    """
    m = re.search(_LIST_RE_TMPL.format(key=key), block)
    if not m:
        return None
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
    """Parse YAML output into {opposition_detected, frames, claims}.

    Strict: any of the three keys missing → that field is None. The
    driver treats any None as a parse failure (no silent crediting).
    """
    block = _extract_yaml_block(response)
    parsed: dict = {"opposition_detected": None, "frames": None, "claims": None}
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


# ---------------------------------------------------------------------------
# Lineup + driver — auto-discover, write metrics_summary.{json,md} per split
# ---------------------------------------------------------------------------

# Headline lineup. Each tuple is (display_label, slug). Files at
# data/results/<split>/<slug>.jsonl.
MODELS = [
    ("Windy-Qwen3.5-4B",         "windy-qwen35-4b"),
    ("Windy-Qwen3.5-9B",         "windy-qwen35-9b"),
    ("Windy-Qwen3.5-27B",        "windy-qwen35-27b"),
    ("Windy-Qwen3.5-27B FP8",    "windy-qwen35-27b-fp8"),
    ("CARDS-Wind-Qwen3.6-27B",   "cards-wind-qwen36-27b"),
    ("CARDS-Wind-Qwen3.6-27B FP8","cards-wind-qwen36-27b-fp8"),
    ("Claude Opus 4.7",          "claude-opus-4-7"),
]

SPLITS = ["test"]
F1_KEYS = [("samples_f1", "Samples F1"),
           ("macro_f1",   "Macro F1"),
           ("micro_f1",   "Micro F1")]

RESULTS_DIR = REPO_ROOT / "data" / "results"


def score_one(path: Path) -> dict | None:
    """Score one result jsonl. Returns dict with parse-fail count + all
    metrics, or None if file missing."""
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)

    api_error = parse_fail = 0
    yt_op, yp_op = [], []
    yt_f, yp_f = [], []
    yt_c, yp_c = [], []
    for r in rows:
        resp = r.get("response", "")
        if isinstance(resp, str) and resp.startswith("ERROR:"):
            api_error += 1
            continue
        pred = parse_response(resp)
        if (pred["opposition_detected"] is None
                or pred["frames"] is None
                or pred["claims"] is None):
            parse_fail += 1
            continue
        yt_op.append(bool(r.get("true_opposition_detected", False)))
        yp_op.append(bool(pred["opposition_detected"]))
        yt_f.append(list(r.get("true_frames") or []))
        yp_f.append(list(pred["frames"]))
        yt_c.append(list(r.get("true_claims") or []))
        yp_c.append(list(pred["claims"]))

    # Opposition-only subset: rows where gold says opposition was present.
    opp_idx = [i for i, t in enumerate(yt_op) if t]
    yt_f_opp = [yt_f[i] for i in opp_idx]
    yp_f_opp = [yp_f[i] for i in opp_idx]
    yt_c_opp = [yt_c[i] for i in opp_idx]
    yp_c_opp = [yp_c[i] for i in opp_idx]

    return {
        "n_rows": n,
        "evaluated": len(yt_op),
        "api_errors": api_error,
        "parse_failures": parse_fail,
        "n_opposition_only": len(opp_idx),
        "detection": detection_metrics(yt_op, yp_op),
        "frames_all":   multilabel_metrics(yt_f, yp_f),
        "frames_opp":   multilabel_metrics(yt_f_opp, yp_f_opp),
        "claims_all":   multilabel_metrics(yt_c, yp_c),
        "claims_opp":   multilabel_metrics(yt_c_opp, yp_c_opp),
    }


def _row(label, cells):
    return "| " + " | ".join([label] + [str(c) for c in cells]) + " |"


def write_summary(split: str, summary: dict[str, dict]) -> None:
    out_dir = RESULTS_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    labels = list(summary.keys())
    n_models = len(labels)
    header = "| " + " | ".join(["Metric"] + labels) + " |"
    sep    = "|" + "|".join(["---"] * (n_models + 1)) + "|"

    lines = [f"# Wind — {split} set\n"]
    # Coverage table
    lines.append("| Model | Rows | API errors | Parse failures | Opposition-only n |")
    lines.append("|-------|---|---|---|---|")
    for lbl in labels:
        e = summary[lbl]
        lines.append(f"| {lbl} | {e['n_rows']} | {e['api_errors']} | {e['parse_failures']} | {e['n_opposition_only']} |")

    # Detection (single F1 row).
    lines.append("\n## Detection (binary)\n")
    lines.append(header)
    lines.append(sep)
    lines.append(_row("F1", [summary[lbl]["detection"]["f1"] for lbl in labels]))

    # F1 family (samples / macro / micro), each with 4 rows: frames-all,
    # frames-opp, claims-all, claims-opp.
    rows = [
        ("Frames — all rows",       "frames_all"),
        ("Frames — opposition only","frames_opp"),
        ("Claims — all rows",       "claims_all"),
        ("Claims — opposition only","claims_opp"),
    ]
    for key, name in F1_KEYS:
        lines.append(f"\n## {name}\n")
        lines.append(header.replace("Metric", "View"))
        lines.append(sep)
        for view_label, view_key in rows:
            cells = [summary[lbl][view_key].get(key, "-") for lbl in labels]
            lines.append(_row(view_label, cells))

    with open(out_dir / "metrics_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  -> {out_dir}/metrics_summary.json")
    print(f"  -> {out_dir}/metrics_summary.md")


def main() -> None:
    for split in SPLITS:
        print(f"\n=== {split} ===")
        summary = {}
        for label, slug in MODELS:
            path = RESULTS_DIR / split / f"{slug}.jsonl"
            scored = score_one(path)
            if scored is None:
                print(f"  [{split}] missing: {label}")
                continue
            print(f"  [{split}] {label}: {scored['evaluated']}/{scored['n_rows']} rows  "
                  f"(api_err={scored['api_errors']}, parse_fail={scored['parse_failures']})")
            summary[label] = scored
        if summary:
            write_summary(split, summary)


if __name__ == "__main__":
    main()
