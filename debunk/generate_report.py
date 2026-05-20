"""Score debunk inference outputs against Leippold2024 gold labels.

Two independent benchmarks — no cross-mapping between taxonomies:

  veracityV1  → scored against `true_veracity` (4-class)
  climinator  → scored against `true_cfb_label` (12-class Climate Feedback)

Reads every result file under `data/results/<split>/`, infers the prompt
variant from the filename suffix (`*-veracityV1.jsonl` / `*-climinator.jsonl`),
parses each model response with `extract_raw_label`, and writes ONE summary
per prompt variant: `metrics_summary_<variant>.{json,md}`.

Usage:
  python generate_report.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef,
)
from collections import Counter as _Counter

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from prompts import (  # noqa: E402
    LABEL_SETS, GOLD_FIELDS, CLIMINATOR_LEVEL_SETS,
    climinator_rollup, extract_raw_label, normalise_gold,
)

RESULTS_DIR = REPO_ROOT / "data" / "results"
SPLITS = ["test"]
PARSE_FAIL = "__PARSE_FAIL__"

# Headline lineup. Each tuple is (display_label, slug). Files at
# data/results/<split>/<slug>-<prompt>.jsonl — one row in each per-prompt
# summary table.
MODELS: list[tuple[str, str]] = [
    ("Sonar",                        "perplexity-sonar"),
    ("Sonar Pro",                    "perplexity-sonar-pro"),
    ("Claude Haiku 4.5 online",      "anthropic-claude-haiku-4-5-online"),
    ("Claude Opus 4.7 online",       "anthropic-claude-opus-4-7-online"),
    ("Claude Opus 4.7 offline",      "anthropic-claude-opus-4-7"),
    ("GPT-4o-mini offline",          "openai-gpt-4o-mini"),
    ("GPT-5.4-mini online",          "openai-gpt-5-4-mini-online"),
    ("GPT-5.5 online",               "openai-gpt-5-5-online"),
    ("GPT-5.5 offline",              "openai-gpt-5-5"),
    ("Gemini 3.1 Flash-Lite online", "google-gemini-3-1-flash-lite-online"),
    ("Gemini 3.1 Pro online",        "google-gemini-3-1-pro-preview-online"),
    ("Exa Answer",                   "exa"),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_one(
    path: Path,
    variant: str,
    *,
    rollup=None,
    label_set_override: tuple[str, ...] | None = None,
) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        return None

    label_set = label_set_override or LABEL_SETS[variant]
    base_label_set = LABEL_SETS[variant]
    gold_field = GOLD_FIELDS[variant]
    _rollup = rollup or (lambda x: x)

    api_error = parse_fail = oov = 0
    y_true: list[str] = []
    y_pred: list[str] = []
    confusion: Counter = Counter()
    raw_pred_counts: Counter = Counter()

    for r in rows:
        gold_raw = normalise_gold(r.get(gold_field, ""))
        if gold_raw not in base_label_set:
            continue  # row has no usable gold label for this variant
        gold = _rollup(gold_raw)
        resp = r.get("response", "")
        if isinstance(resp, str) and resp.startswith("ERROR:"):
            api_error += 1
            pred = PARSE_FAIL
        else:
            raw = extract_raw_label(resp)
            if raw is None:
                parse_fail += 1
                pred = PARSE_FAIL
            else:
                raw_pred_counts[raw] += 1
                if raw not in base_label_set:
                    oov += 1
                    pred = PARSE_FAIL
                else:
                    pred = _rollup(raw)
        y_true.append(gold)
        y_pred.append(pred)
        confusion[(gold, pred)] += 1

    labels = list(label_set)
    classes_present = sorted({l for l in y_true})  # for macro F1 excl. zero-support classes

    # Majority-class baseline: accuracy if we predicted the most-common gold label every time.
    if y_true:
        majority_label, majority_count = _Counter(y_true).most_common(1)[0]
        baseline_acc = majority_count / len(y_true)
    else:
        majority_label, baseline_acc = None, 0.0

    return {
        "n_rows":         len(rows),
        "evaluated":      len(y_true),
        "api_errors":     api_error,
        "parse_failures": parse_fail,
        "oov":            oov,
        "mcc":              round(matthews_corrcoef(y_true, y_pred), 3),
        "accuracy":         round(accuracy_score(y_true, y_pred), 3),
        "baseline_acc":     round(baseline_acc, 3),
        "baseline_label":   majority_label,
        "macro_f1_present": round(f1_score(y_true, y_pred, labels=classes_present,
                                           average="macro", zero_division=0), 3),
        "per_class": {
            lbl: {
                "precision": round(precision_score(y_true, y_pred, labels=[lbl],
                                                   average="micro", zero_division=0), 3),
                "recall":    round(recall_score(y_true, y_pred, labels=[lbl],
                                                average="micro", zero_division=0), 3),
                "f1":        round(f1_score(y_true, y_pred, labels=[lbl],
                                            average="micro", zero_division=0), 3),
                "support":   sum(1 for t in y_true if t == lbl),
            }
            for lbl in labels
        },
        "raw_pred_counts": dict(raw_pred_counts),
        "confusion":       {f"{g}->{p}": c for (g, p), c in sorted(confusion.items())},
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _row(label: str, cells: list) -> str:
    return "| " + " | ".join([label] + [str(c) for c in cells]) + " |"


def write_combined_summary(split: str, all_summaries: dict[str, dict]) -> None:
    """Single combined `metrics_summary.md` (+ `.json`) — one table, models as
    rows, metrics grouped by prompt variant. Per-class breakdowns live only in
    the JSON for downstream inspection."""
    out_dir = RESULTS_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    variants = list(all_summaries.keys())
    # Union of all models scored (so a model missing on one variant still appears)
    all_models: list[str] = []
    for summary in all_summaries.values():
        for m in summary:
            if m not in all_models:
                all_models.append(m)

    # Header row: Model | <var1 MCC> | <var1 Acc> | <var1 F1> | <var2 MCC> | ...
    metric_cols = [("MCC", "mcc"), ("Acc", "accuracy"), ("F1", "macro_f1_present")]
    header_cells = ["Model"]
    for v in variants:
        for short, _ in metric_cols:
            header_cells.append(f"{v} {short}")
    header = "| " + " | ".join(header_cells) + " |"
    sep = "|" + "|".join(["---"] * len(header_cells)) + "|"

    lines = [f"# Debunk — {split} set\n", header, sep]

    # Majority baseline row (MCC=0 by construction; baseline acc = majority count / N;
    # F1 left as `—` since "predict majority" yields trivial-and-uninformative F1).
    baseline_cells = ["*Majority baseline*"]
    for v in variants:
        any_model = next(iter(all_summaries[v].values()))
        baseline_cells += ["0.000", f"{any_model['baseline_acc']:.3f}", "—"]
    lines.append("| " + " | ".join(baseline_cells) + " |")

    # One row per model
    for m in all_models:
        cells = [m]
        for v in variants:
            entry = all_summaries[v].get(m)
            if entry is None:
                cells += ["—", "—", "—"]
            else:
                for _, key in metric_cols:
                    cells.append(f"{entry[key]:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    # Footer note pinning the baseline labels
    notes = []
    for v in variants:
        any_model = next(iter(all_summaries[v].values()))
        notes.append(f"`{v}` majority class: `{any_model['baseline_label']}`")
    lines += ["", "*" + "; ".join(notes) + "*"]

    # Climinator-by-level breakdown (L1=12, L2=5, L3=3, L4=2)
    if "climinator" in all_summaries:
        clim = all_summaries["climinator"]
        any_entry = next(iter(clim.values()))
        if "levels" in any_entry:
            lines += ["", "## Climinator hierarchy (Leippold 2024 Fig. 3)",
                      "Same predictions rolled up to the 5/3/2-class credibility taxonomies.", ""]
            level_labels = [("L1", "1", "12"), ("L2", "2", "5"), ("L3", "3", "3"), ("L4", "4", "2")]
            header_cells = ["Model"]
            for short, _, k in level_labels:
                header_cells += [f"{short} ({k}c) MCC", f"{short} Acc", f"{short} F1"]
            lines.append("| " + " | ".join(header_cells) + " |")
            lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")

            baseline_cells = ["*Majority baseline*"]
            for _, key, _ in level_labels:
                base = any_entry["levels"][key]["baseline_acc"]
                baseline_cells += ["0.000", f"{base:.3f}", "—"]
            lines.append("| " + " | ".join(baseline_cells) + " |")

            for m, entry in clim.items():
                cells = [m]
                for _, key, _ in level_labels:
                    lvl = entry.get("levels", {}).get(key)
                    if lvl is None:
                        cells += ["—", "—", "—"]
                    else:
                        cells += [f"{lvl['mcc']:.3f}", f"{lvl['accuracy']:.3f}",
                                  f"{lvl['macro_f1_present']:.3f}"]
                lines.append("| " + " | ".join(cells) + " |")

            base_notes = []
            for short, key, _ in level_labels:
                bl = any_entry["levels"][key]["baseline_label"]
                base_notes.append(f"{short}=`{bl}`")
            lines += ["", "*Majority class per level: " + "; ".join(base_notes) + "*"]

    with open(out_dir / "metrics_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  -> {out_dir}/metrics_summary.json")
    print(f"  -> {out_dir}/metrics_summary.md")


def _discover_extras(split: str) -> None:
    split_dir = RESULTS_DIR / split
    if not split_dir.exists():
        return
    expected = {f"{slug}-{v}.jsonl" for _, slug in MODELS for v in LABEL_SETS}
    extras = sorted(p.name for p in split_dir.glob("*.jsonl")
                    if p.name not in expected and not p.name.startswith("metrics_summary"))
    if extras:
        print(f"  [{split}] extra result files (not in MODELS): {extras}")


def main() -> None:
    for split in SPLITS:
        print(f"\n=== {split} ===")
        _discover_extras(split)
        all_summaries: dict[str, dict] = {}
        for variant in LABEL_SETS:
            summary: dict = {}
            for label, slug in MODELS:
                path = RESULTS_DIR / split / f"{slug}-{variant}.jsonl"
                scored = score_one(path, variant)
                if scored is None:
                    print(f"  [{variant}] missing: {label} ({path.name})")
                    continue
                print(f"  [{variant}] {label}: {scored['evaluated']}/{scored['n_rows']} rows  "
                      f"MCC={scored['mcc']:.3f}  acc={scored['accuracy']:.3f} "
                      f"(baseline={scored['baseline_acc']:.3f})  "
                      f"macroF1={scored['macro_f1_present']:.3f}  "
                      f"[api_err={scored['api_errors']}, parse_fail={scored['parse_failures']}, "
                      f"oov={scored['oov']}]")
                if variant == "climinator":
                    # Climinator has a 4-level credibility hierarchy (Leippold 2024
                    # Fig. 3). Score each rollup level so we can see how the
                    # metrics improve as the taxonomy is coarsened. L1 already
                    # scored above; reuse those numbers for level "1".
                    levels: dict[str, dict] = {"1": {k: scored[k] for k in scored
                                                     if k not in ("per_class", "raw_pred_counts", "confusion")}}
                    levels["1"]["per_class"] = scored["per_class"]
                    levels["1"]["confusion"] = scored["confusion"]
                    for lvl in (2, 3, 4):
                        lvl_set = CLIMINATOR_LEVEL_SETS[lvl]
                        lvl_scored = score_one(
                            path, variant,
                            rollup=lambda x, _l=lvl: climinator_rollup(x, _l),
                            label_set_override=lvl_set,
                        )
                        if lvl_scored is None:
                            continue
                        print(f"    L{lvl}: MCC={lvl_scored['mcc']:.3f}  acc={lvl_scored['accuracy']:.3f} "
                              f"(baseline={lvl_scored['baseline_acc']:.3f})  "
                              f"macroF1={lvl_scored['macro_f1_present']:.3f}")
                        levels[str(lvl)] = {k: lvl_scored[k] for k in lvl_scored
                                            if k != "raw_pred_counts"}
                    scored["levels"] = levels
                summary[label] = scored
            if summary:
                all_summaries[variant] = summary
        if all_summaries:
            write_combined_summary(split, all_summaries)


if __name__ == "__main__":
    main()
