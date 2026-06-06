"""Compute metrics for CARDS result jsonls.

Writes three reports under data/results/<split>/ (test + twitter):
  metrics_summary.{json,md}              — headline FT + API across all models
  test/recot_ablation.{json,md}          — 4B + 9B: Base vs No-RECoT vs RECoT
  test/scaling_ablation.{json,md}        — Base vs RECoT-FT across sizes
plus LaTeX row fragments for the chapter tables (test only):
  test/{recot_ablation,scaling_frontier,quantization}.tex

Re-running overwrites. Per-variant classification reports are NOT written.

Usage:
    python generate_report.py
"""

import json
import os
import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
SPLITS = ["test", "twitter"]
LEVELS = [1, 2, 3]
MIN_SUPPORTS = [3]  # 0 = all labels (dropped — too noisy on long-tail rare classes)

# Headline lineup: FT'd CARDS variants vs API models. Base entries live in
# the ablation, not here.
MODELS = [
    ("CARDS-Qwen3.5-2B",      "cards-qwen35-2b"),
    ("CARDS-Qwen3.5-4B",      "cards-qwen35-4b"),
    ("CARDS-Qwen3.5-9B",      "cards-qwen35-9b"),
    ("CARDS-Qwen3.5-27B",     "cards-qwen35-27b"),
    ("CARDS-Qwen3.5-27B FP8", "cards-qwen35-27b-fp8"),
    ("CARDS-Qwen3.6-27B",     "cards-qwen36-27b"),
    ("Claude Opus 4.7",       "claude-opus-4-7"),
    ("GPT-5.5",               "gpt-5-5"),
    ("GPT-4o-mini",           "gpt-4o-mini"),
    ("CARDS-mini-opus",       "cards-mini-opus"),
]


_CODE_RE = re.compile(r"<?(\d[\d_]+\d)>?")


def load_canonical_gold():
    """text -> true_claims from data/cards_test.jsonl (final_claims gold).

    Several result jsonls embed a stale pre-final_claims gold for 67 test
    rows, so test-split scoring re-pairs gold from the canonical split file
    instead of trusting the copy stored alongside each response. Keyed by
    text — unique across the split, unlike `id` (one duplicate).
    """
    gold = {}
    with open(os.path.join(BASE_DIR, "data", "cards_test.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            gold[r["text"]] = r["true_claims"]
    return gold


def parse_response(response, require_think=True):
    """Parse a model response into a list of category codes.

    require_think=True (default, for RECoT-trained models): require </think>
    followed by a categories: YAML block. Anything else returns [].

    require_think=False (for no-RECoT-trained models): accept the categories:
    block anywhere in the response. The no-RECoT training data has no
    <think>...</think>, so models trained on it shouldn't be required to
    emit one at inference.

    Pre-parsed list-shape responses (legacy base-model outputs) are accepted
    as-is in either mode.
    """
    if isinstance(response, list):
        if response and isinstance(response[0], str) and not _CODE_RE.fullmatch(response[0]):
            return parse_response(response[0], require_think=require_think)
        return [_CODE_RE.match(str(c).strip()).group(1)
                for c in response if _CODE_RE.match(str(c).strip())]
    if not isinstance(response, str):
        return []
    if require_think:
        if "</think>" not in response:
            return []
        body = response.split("</think>")[-1].strip()
    else:
        body = response.split("</think>")[-1].strip() if "</think>" in response else response.strip()
    m = re.search(r"categories:\s*\n((?:\s*-\s*.+\n?)+)", body)
    return re.findall(r"-\s*<?(\d[\d_]+\d)>?", m.group(1)) if m else []


def truncate_to_level(codes, level):
    return list({"_".join(c.split("_")[:level]) for c in codes})


def binarize(y_true_lists, y_pred_lists, min_support=0):
    # Fix the label vocab to gold codes only. Hallucinated labels in y_pred
    # are dropped silently — keeping them inflates each model's macro
    # denominator differently and breaks cross-model comparison. The model
    # is still penalized for hallucinating: the correct gold label gets an
    # FN since the model didn't predict it.
    gold_labels = sorted({l for row in y_true_lists for l in row})
    mlb = MultiLabelBinarizer()
    mlb.fit([gold_labels])
    gold_set = set(gold_labels)
    y_pred_clean = [[l for l in row if l in gold_set] for row in y_pred_lists]
    y_t = mlb.transform(y_true_lists)
    y_p = mlb.transform(y_pred_clean)
    if min_support > 0:
        mask = y_t.sum(axis=0) >= min_support
        y_t, y_p = y_t[:, mask], y_p[:, mask]
    return y_t, y_p


def compute_metrics(df, level, min_support):
    y_true = df["true_claims"].apply(lambda x: truncate_to_level(x, level)).tolist()
    y_pred = df["pred_claims"].apply(lambda x: truncate_to_level(x, level)).tolist()
    y_t, y_p = binarize(y_true, y_pred, min_support)
    out = {}
    for avg in ("samples", "macro", "micro"):
        out[f"{avg}_f1"]        = round(f1_score(y_t, y_p, average=avg, zero_division=0), 3)
        out[f"{avg}_precision"] = round(precision_score(y_t, y_p, average=avg, zero_division=0), 3)
        out[f"{avg}_recall"]    = round(recall_score(y_t, y_p, average=avg, zero_division=0), 3)
    out["accuracy"] = round(accuracy_score(y_t, y_p), 3)
    return out


def score_one(path, label_field, label, prefix="", gold=None):
    df = pd.read_json(path, lines=True)
    if gold is not None:
        df = df[df["text"].isin(gold)].copy()
        df["true_claims"] = df["text"].map(gold)
    elif label_field != "true_claims":
        df["true_claims"] = df[label_field]
    # Slugs that opt into the relaxed parser (no </think> required):
    #   *norecot* — FT models trained without RECoT
    #   *nothink* — base models run with --no-recot (thinking-off inference)
    name = os.path.basename(path).lower()
    require_think = "norecot" not in name and "nothink" not in name
    df["pred_claims"] = df["response"].apply(lambda r: parse_response(r, require_think=require_think))
    parse_failures = int((df["pred_claims"].map(len) == 0).sum())
    print(f"  {prefix}{label}: {len(df)} rows, {parse_failures} parse failures")

    entry = {"label": label, "n_samples": len(df), "parse_failures": parse_failures}
    for ms in MIN_SUPPORTS:
        for lvl in LEVELS:
            suffix = "all" if ms == 0 else f"minsup_{ms}"
            entry[f"level_{lvl}_{suffix}"] = compute_metrics(df, lvl, ms)
    return entry


# Metric families surfaced in the markdown, one sub-table each. Precision and
# recall accompany F1 at the samples and macro averages; micro p/r round it out.
DISPLAY_METRICS = [
    ("samples_f1",        "Samples F1"),
    ("samples_precision", "Samples Precision"),
    ("samples_recall",    "Samples Recall"),
    ("macro_f1",          "Macro F1"),
    ("macro_precision",   "Macro Precision"),
    ("macro_recall",      "Macro Recall"),
    ("micro_f1",          "Micro F1"),
    ("micro_precision",   "Micro Precision"),
    ("micro_recall",      "Micro Recall"),
]


def write_summary(title, summary, out_dir, basename):
    with open(os.path.join(out_dir, f"{basename}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    labels = [e["label"] for e in summary.values()]
    lines = [f"# {title}\n",
             "| Model | N | Parse fails |",
             "|-------|---|-------------|"]
    for e in summary.values():
        lines.append(f"| {e['label']} | {e['n_samples']} | {e['parse_failures']} |")

    # One section per support threshold; one sub-table per F1 metric.
    # Models are columns; rows are levels.
    header_row = "| Level | " + " | ".join(labels) + " |"
    sep_row    = "|-------|" + "|".join(["---"] * len(labels)) + "|"
    for ms in MIN_SUPPORTS:
        suffix = "all" if ms == 0 else f"minsup_{ms}"
        section = "All labels" if ms == 0 else f"Support ≥ {ms}"
        lines.append(f"\n## {section}\n")
        for key, name in DISPLAY_METRICS:
            lines.append(f"### {name}\n")
            lines.append(header_row)
            lines.append(sep_row)
            for lvl in LEVELS:
                cells = [str(e[f"level_{lvl}_{suffix}"][key]) for e in summary.values()]
                lines.append(f"| {lvl} | " + " | ".join(cells) + " |")
            lines.append("")

    with open(os.path.join(out_dir, f"{basename}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  -> {out_dir}/{basename}.json")
    print(f"  -> {out_dir}/{basename}.md")


def report_for_split(split):
    split_dir = os.path.join(RESULTS_DIR, split)
    label_field = "labels" if split == "twitter" else "true_claims"
    gold = load_canonical_gold() if split == "test" else None
    summary = {}
    for label, stem in MODELS:
        path = os.path.join(split_dir, f"{stem}.jsonl")
        if not os.path.exists(path):
            print(f"  [{split}] missing: {label}")
            continue
        summary[label] = score_one(path, label_field, label, prefix=f"[{split}] ", gold=gold)
    write_summary(f"CARDS — {split} set", summary, split_dir, "metrics_summary")


# Inner ablation (4B + 9B): four columns per size — does base + simpler
# prompt close the gap? does FT add value? does RECoT-FT help further?
RECOT_ABLATION = [
    ("Qwen3.5-4B Base (think)",                  "qwen35-4b-base"),
    ("Qwen3.5-4B Base (no-think)",               "qwen35-4b-base-nothink"),
    ("CARDS-Qwen3.5-4B No RECoT (no-think)",     "cards-qwen35-4b-norecot-nothink"),
    ("CARDS-Qwen3.5-4B No RECoT (think)",        "cards-qwen35-4b-norecot"),
    ("CARDS-Qwen3.5-4B",                         "cards-qwen35-4b"),
    ("Qwen3.5-9B Base (think)",                  "qwen35-9b-base"),
    ("Qwen3.5-9B Base (no-think)",               "qwen35-9b-base-nothink"),
    ("CARDS-Qwen3.5-9B No RECoT (no-think)",     "cards-qwen35-9b-norecot-nothink"),
    ("CARDS-Qwen3.5-9B No RECoT (think)",        "cards-qwen35-9b-norecot"),
    ("CARDS-Qwen3.5-9B",                         "cards-qwen35-9b"),
    ("Qwen3.5-27B Base (think)",                 "qwen35-27b-base"),
    ("Qwen3.5-27B Base (no-think)",              "qwen35-27b-base-nothink"),
    ("CARDS-Qwen3.5-27B No RECoT (no-think)",    "cards-qwen35-27b-norecot-nothink"),
    ("CARDS-Qwen3.5-27B No RECoT (think)",       "cards-qwen35-27b-norecot"),
    ("CARDS-Qwen3.5-27B",                        "cards-qwen35-27b"),
]

# ---------------------------------------------------------------------------
# LaTeX row fragments for the chapter tables (data/results/test/*.tex).
# Rows only — main.tex owns the table scaffolding and \inputs these.
# Columns: model, Samples F1 L1-3, Macro F1 L1-3 (minsup 3, %.3f keeps
# trailing zeros that the md tables drop).
# ---------------------------------------------------------------------------

# recot_ablation.tex — chapter Table 1 (\midrule between size groups)
ABLATION_TEX = [
    [("CARDS-Qwen3.5-4B (no RECoT)",  "cards-qwen35-4b-norecot-nothink"),
     ("CARDS-Qwen3.5-4B",             "cards-qwen35-4b")],
    [("CARDS-Qwen3.5-9B (no RECoT)",  "cards-qwen35-9b-norecot-nothink"),
     ("CARDS-Qwen3.5-9B",             "cards-qwen35-9b")],
    [("CARDS-Qwen3.5-27B (no RECoT)", "cards-qwen35-27b-norecot-nothink"),
     ("CARDS-Qwen3.5-27B",            "cards-qwen35-27b")],
]

# scaling_frontier.tex — chapter Table 2 (groups -> \multicolumn{7}{l}{\textit{...}})
SCALING_TEX = [
    ("Open-source base (zero-shot)",
     [("Qwen3.5-4B Base", "qwen35-4b-base"), ("Qwen3.5-9B Base", "qwen35-9b-base"),
      ("Qwen3.5-27B Base", "qwen35-27b-base")]),
    ("Open-source RECoT fine-tuned",
     [("CARDS-Qwen3.5-4B", "cards-qwen35-4b"), ("CARDS-Qwen3.5-9B", "cards-qwen35-9b"),
      ("CARDS-Qwen3.5-27B", "cards-qwen35-27b")]),
    ("Closed-source (zero-shot)",
     [("GPT-4o-mini", "gpt-4o-mini"), ("Claude Opus 4.7", "claude-opus-4-7"),
      ("GPT-5.5", "gpt-5-5")]),
    ("Closed-source RECoT fine-tuned",
     [("CARDS-mini-opus", "cards-mini-opus")]),
]

# quantization.tex — chapter Table 3
QUANT_TEX = [("CARDS-Qwen3.5-27B (BF16)", "cards-qwen35-27b"),
             ("CARDS-Qwen3.5-27B FP8",    "cards-qwen35-27b-fp8")]


def fmt_row(name, m, pad=33):
    cells = [m[f"level_{l}_minsup_3"]["samples_f1"] for l in (1, 2, 3)] + \
            [m[f"level_{l}_minsup_3"]["macro_f1"]   for l in (1, 2, 3)]
    return f"{name:<{pad}s} & " + " & ".join(f"{c:.3f}" for c in cells) + r" \\"


def write_tex_fragments():
    split_dir = os.path.join(RESULTS_DIR, "test")
    gold = load_canonical_gold()
    memo = {}

    def entry(stem):
        if stem not in memo:
            path = os.path.join(split_dir, f"{stem}.jsonl")
            memo[stem] = (score_one(path, "true_claims", stem, prefix="[tex] ", gold=gold)
                          if os.path.exists(path) else None)
            if memo[stem] is None:
                print(f"  [tex] missing: {stem}")
        return memo[stem]

    def emit(basename, lines):
        out_path = os.path.join(split_dir, f"{basename}.tex")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  -> {out_path}")

    lines = []
    for i, group in enumerate(ABLATION_TEX):
        if i:
            lines.append(r"\midrule")
        lines += [fmt_row(name, e) for name, stem in group if (e := entry(stem))]
    emit("recot_ablation", lines)

    lines = []
    for group, models in SCALING_TEX:
        rows = [fmt_row(name, e) for name, stem in models if (e := entry(stem))]
        if rows:
            if lines:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{group}}}}} \\")
            lines += rows
    emit("scaling_frontier", lines)

    emit("quantization",
         [fmt_row(name, e) for name, stem in QUANT_TEX if (e := entry(stem))])


# Scaling ablation: base vs RECoT-FT across model sizes.
SCALING_ABLATION = [
    ("Qwen3.5-2B Base",   "qwen35-2b-base"),
    ("CARDS-Qwen3.5-2B",  "cards-qwen35-2b"),
    ("Qwen3.5-4B Base",   "qwen35-4b-base"),
    ("CARDS-Qwen3.5-4B",  "cards-qwen35-4b"),
    ("Qwen3.5-9B Base",   "qwen35-9b-base"),
    ("CARDS-Qwen3.5-9B",  "cards-qwen35-9b"),
    ("Qwen3.5-27B Base",  "qwen35-27b-base"),
    ("CARDS-Qwen3.5-27B", "cards-qwen35-27b"),
]


def report_ablation(entries, title, basename, tag):
    split_dir = os.path.join(RESULTS_DIR, "test")
    gold = load_canonical_gold()
    summary = {}
    for label, stem in entries:
        path = os.path.join(split_dir, f"{stem}.jsonl")
        if not os.path.exists(path):
            print(f"  [{tag}] missing: {label}")
            continue
        summary[label] = score_one(path, "true_claims", label, prefix=f"[{tag}] ", gold=gold)
    write_summary(title, summary, split_dir, basename)


if __name__ == "__main__":
    for split in SPLITS:
        print(f"\n=== {split} ===")
        report_for_split(split)
    print("\n=== recot ablation (4B + 9B, test) ===")
    report_ablation(RECOT_ABLATION,
                    "CARDS — RECoT-FT ablation (Qwen3.5 4B + 9B, test set)",
                    "recot_ablation", "recot")
    print("\n=== scaling ablation (base vs RECoT-FT, test) ===")
    report_ablation(SCALING_ABLATION,
                    "CARDS — Scaling ablation (Base vs RECoT-FT, test set)",
                    "scaling_ablation", "scaling")
    print("\n=== LaTeX fragments (chapter tables, test) ===")
    write_tex_fragments()
