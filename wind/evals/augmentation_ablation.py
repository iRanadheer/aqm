"""Before/after synthetic-augmentation ablation on the val set.

Reuses generate_report.parse_response and replicates score_one's gold/pred
construction (including the parse-fail penalty), then scores the
pre-augmentation (`*-preaug.jsonl`) vs post-augmentation (`*.jsonl`) val runs
for each student size that has both, and emits a before/after delta table.

Macro F1 is reported support-restricted (codes with gold support >= MIN_SUPPORT
in val), matching the project's per-code convention. Samples F1 and detection
F1 are support-independent. Opus 4.7 and the 27B student are reference points
(no preaug counterpart).

Run:
    .venv/bin/python evals/augmentation_ablation.py
Writes data/results/val/augmentation_ablation.{json,md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from generate_report import parse_response, detection_metrics, _samples_f1  # noqa: E402

VAL_DIR = REPO_ROOT / "data" / "results" / "val"
MIN_SUPPORT = 3
SENTINEL = "__PARSE_FAIL__"

# (display, before_slug, after_slug).
PAIRS = [
    ("Windy-Qwen3.5-4B", "windy-qwen35-4b-preaug", "windy-qwen35-4b"),
    ("Windy-Qwen3.5-9B", "windy-qwen35-9b-preaug", "windy-qwen35-9b"),
]
REFERENCES = [
    ("Windy-Qwen3.5-27B", "windy-qwen35-27b"),
    ("Claude Opus 4.7 (teacher)", "claude-opus-4-7"),
]

# (view key, metric, display label). metric in {detection, samples, macro}.
METRICS = [
    ("detection", "detection", "Detection F1"),
    ("frames_opp", "samples", "Frames (opp) Samples F1"),
    ("frames_opp", "macro", f"Frames (opp) Macro F1 (sup>={MIN_SUPPORT})"),
    ("claims_opp", "samples", "Claims (opp) Samples F1"),
    ("claims_opp", "macro", f"Claims (opp) Macro F1 (sup>={MIN_SUPPORT})"),
    ("claims_all", "samples", "Claims (all) Samples F1"),
    ("claims_all", "macro", f"Claims (all) Macro F1 (sup>={MIN_SUPPORT})"),
]


def _macro_min_support(y_true, y_pred, min_support: int) -> float | None:
    """Macro F1 over codes whose gold support >= min_support."""
    y_pred = [[x for x in lab if x != SENTINEL] for lab in y_pred]
    support: dict[str, int] = {}
    for lab in y_true:
        for c in set(lab):
            support[c] = support.get(c, 0) + 1
    kept = sorted(c for c, s in support.items() if s >= min_support)
    if not kept:
        return None
    mlb = MultiLabelBinarizer(classes=kept)
    yt = mlb.fit_transform([[c for c in lab if c in kept] for lab in y_true])
    yp = mlb.transform([[c for c in lab if c in kept] for lab in y_pred])
    return round(float(f1_score(yt, yp, average="macro", zero_division=0)), 3)


def score(path: Path) -> dict | None:
    """Replicate score_one's gold/pred construction, return arrays + metrics."""
    if not path.exists():
        print(f"  missing: {path.name}")
        return None
    rows = [json.loads(l) for l in open(path)]
    yt_op, yp_op = [], []
    yt_f, yp_f, yt_c, yp_c = [], [], [], []
    for r in rows:
        gold_op = bool(r.get("true_opposition_detected", False))
        gold_f = list(r.get("true_frames") or [])
        gold_c = list(r.get("true_claims") or [])
        resp = r.get("response", "")
        failed = False
        pred = {"opposition_detected": None, "frames": None, "claims": None}
        if isinstance(resp, str) and resp.startswith("ERROR:"):
            failed = True
        else:
            pred = parse_response(resp)
            if pred["opposition_detected"] is None or pred["frames"] is None or pred["claims"] is None:
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

    opp = [i for i, t in enumerate(yt_op) if t]
    sub = lambda arr: [arr[i] for i in opp]  # noqa: E731

    def view(view_key):
        if view_key == "frames_opp":
            return sub(yt_f), sub(yp_f)
        if view_key == "claims_opp":
            return sub(yt_c), sub(yp_c)
        if view_key == "claims_all":
            return yt_c, yp_c
        raise KeyError(view_key)

    out: dict[str, float | None] = {}
    det = detection_metrics(yt_op, yp_op)["f1"]
    for view_key, metric, _ in METRICS:
        if metric == "detection":
            out[f"{view_key}.{metric}"] = det
            continue
        t, p = view(view_key)
        if metric == "samples":
            out[f"{view_key}.{metric}"] = round(_samples_f1(t, p), 3)
        else:  # macro, support-restricted
            out[f"{view_key}.{metric}"] = _macro_min_support(t, p, MIN_SUPPORT)
    return out


def main() -> None:
    out: dict = {"min_support": MIN_SUPPORT, "pairs": [], "references": {}}

    for display, before_slug, after_slug in PAIRS:
        b = score(VAL_DIR / f"{before_slug}.jsonl")
        a = score(VAL_DIR / f"{after_slug}.jsonl")
        if b is None or a is None:
            continue
        rec = {"model": display, "before": b, "after": a, "delta": {}}
        for view_key, metric, _ in METRICS:
            mk = f"{view_key}.{metric}"
            bv, av = b[mk], a[mk]
            rec["delta"][mk] = round(av - bv, 3) if bv is not None and av is not None else None
        out["pairs"].append(rec)

    for display, slug in REFERENCES:
        s = score(VAL_DIR / f"{slug}.jsonl")
        if s is not None:
            out["references"][display] = s

    (VAL_DIR / "augmentation_ablation.json").write_text(json.dumps(out, indent=2))

    L: list[str] = ["# Synthetic augmentation ablation — val set\n"]
    L.append(
        f"Before = pre-augmentation student (`*-preaug`); After = retrained on "
        f"`train_augmented`. Delta = After - Before. Macro F1 restricted to codes "
        f"with gold support >= {MIN_SUPPORT} in val (same kept-code set before and "
        f"after); Samples F1 and Detection F1 are support-independent.\n"
    )
    for rec in out["pairs"]:
        L.append(f"\n## {rec['model']}\n")
        L.append("| Metric | Before | After | Delta |")
        L.append("|---|---:|---:|---:|")
        for view_key, metric, label in METRICS:
            mk = f"{view_key}.{metric}"
            bv, av, dv = rec["before"][mk], rec["after"][mk], rec["delta"][mk]
            if dv is None:
                dcell = "n/a"
            elif dv > 0:
                dcell = f"**+{dv:.3f}**"
            elif dv < 0:
                dcell = f"{dv:.3f}"
            else:
                dcell = "0.000"
            L.append(f"| {label} | {bv} | {av} | {dcell} |")

    if out["references"]:
        L.append("\n## Reference points (no pre-aug counterpart)\n")
        ref_labels = list(out["references"].keys())
        L.append("| Metric | " + " | ".join(ref_labels) + " |")
        L.append("|---" * (len(ref_labels) + 1) + "|")
        for view_key, metric, label in METRICS:
            mk = f"{view_key}.{metric}"
            cells = [str(out["references"][r][mk]) for r in ref_labels]
            L.append(f"| {label} | " + " | ".join(cells) + " |")

    (VAL_DIR / "augmentation_ablation.md").write_text("\n".join(L) + "\n")
    print(f"  -> {VAL_DIR}/augmentation_ablation.json")
    print(f"  -> {VAL_DIR}/augmentation_ablation.md")


if __name__ == "__main__":
    main()
