# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Emit LaTeX table bodies for the dissertation's fallacy-classification study.

Reads the FLICC pipeline outputs and writes row files to
dissertation/chapters/debunk/tables/, consumed via \\CatchFileDef in
preamble/chapter-macros.tex (the convention used by the cards and wind
chapters). Apart from the published point estimates from Zanartu et al. 2024
(Table 4), which no pipeline of ours produces, every number is read from a
pipeline artifact.

    uv run make_tables.py

Writes  flicc_results.tex       test split, macro F1 + accuracy, grouped
        flicc_ablation.tex      prompt condition x frontier model (macro F1)
        flicc_adjudication.tex  original vs review-rule accuracy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "evals"))
from score_adjudicated import DISPLAY, assess  # noqa: E402
from build_multilabel_gold import read  # noqa: E402

OUT = ROOT.parents[2] / "dissertation" / "chapters" / "debunk" / "tables"
TEST = ROOT / "data" / "results" / "test"

# Models the chapter's Results prose argues about; the rest go to the appendix.
MAIN_MODELS = {
    "fzanartu-flicc-deberta-v2-xlarge",
    "openai-gpt-5-5-deconstruct-revised",
    "anthropic-claude-opus-4-7-deconstruct-revised",
    "c3ds-flicc-qwen3-5-9b-deconstruct-revised",
    "c3ds-flicc-qwen3-5-27b-deconstruct-revised",
}

# Zanartu et al. 2024, Table 4 — published point estimates.
PUBLISHED = [
    ("DeBERTa-v2-xlarge (published)", 0.73, 0.74),
    ("GPT-4-0125 (published)", 0.30, 0.32),
    ("Gemini-pro (published)", 0.11, 0.21),
]

# slug -> (display name, group)
RUNS = {
    "openai-gpt-5-5-deconstruct-revised": ("GPT-5.5", "frontier"),
    "anthropic-claude-opus-4-7-deconstruct-revised": ("Claude Opus 4.7", "frontier"),
    "c3ds-flicc-qwen3-5-27b-deconstruct-revised": ("FLICC-Qwen3.5-27B", "ft"),
    "c3ds-flicc-qwen3-5-9b-deconstruct-revised": ("FLICC-Qwen3.5-9B", "ft"),
    "c3ds-flicc-qwen3-5-4b-deconstruct-revised": ("FLICC-Qwen3.5-4B", "ft"),
    "c3ds-flicc-qwen3-5-2b-deconstruct-revised": ("FLICC-Qwen3.5-2B", "ft"),
    "c3ds-flicc-qwen3-5-0-8b-deconstruct-revised": ("FLICC-Qwen3.5-0.8B", "ft"),
    "qwen-qwen3-5-27b-deconstruct-revised-structured": ("Qwen3.5-27B (base)", "base"),
    "qwen-qwen3-5-9b-deconstruct-revised-structured": ("Qwen3.5-9B (base)", "base"),
    "qwen-qwen3-5-4b-deconstruct-revised-structured": ("Qwen3.5-4B (base)", "base"),
    "qwen-qwen3-5-2b-deconstruct-revised-structured": ("Qwen3.5-2B (base)", "base"),
    "qwen-qwen3-5-0-8b-deconstruct-revised-structured": ("Qwen3.5-0.8B (base)", "base"),
}
GROUP_LABEL = {
    "frontier": "Frontier, zero-shot (revised codebook)",
    "ft": "RECoT fine-tuned",
    "base": "Base Qwen3.5 (schema-constrained)",
}


def results_table(only_main: bool) -> str:
    metrics = json.loads((TEST / "metrics_summary.json").read_text())
    deberta = json.loads((TEST / "fzanartu-flicc-deberta-v2-xlarge.summary.json").read_text())
    sig = {}
    for line in (ROOT / "data" / "significance" / "summary_test.md").read_text().splitlines():
        if line.startswith("|") and "fzanartu-flicc-deberta-v2-xlarge" in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 5 and cells[0] != "A":  # paired rows: A | B | Delta | CI | verdict
                sig[cells[0]] = (cells[2], cells[3], cells[4])

    def ci(slug, name):
        if slug not in sig:
            return "--"
        delta, ci_text, _ = sig[slug]
        return f"${delta}$ ${ci_text}$"

    lines: list[str] = []
    if not only_main:
        lines.append("\\multicolumn{4}{l}{\\textit{Published (Zanartu et al.)}} \\\\")
        for name, f1, acc in PUBLISHED:
            lines.append(f"{name} & {f1:.3f} & {acc:.3f} & -- \\\\")
        lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\textit{Reproduced baseline}} \\\\")
    lines.append(f"DeBERTa-v2-xlarge & {deberta['original_macro_f1']:.3f} & "
                 f"{deberta['original_accuracy']:.3f} & -- \\\\")
    groups = ("frontier", "ft") if only_main else ("frontier", "ft", "base")
    for group in groups:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{GROUP_LABEL[group]}}}}} \\\\")
        rows = [(name, metrics[slug]["macro_f1"], metrics[slug]["accuracy"], ci(slug, name))
                for slug, (name, g) in RUNS.items()
                if g == group and (not only_main or slug in MAIN_MODELS)]
        for name, f1, acc, ci_text in sorted(rows, key=lambda r: -r[1]):
            lines.append(f"{name} & {f1:.3f} & {acc:.3f} & {ci_text} \\\\")
    return "\n".join(lines) + "\n"


def ablation_table() -> str:
    metrics = json.loads((TEST / "metrics_summary.json").read_text())
    conds = [("Bare prompt", "bare"),
             ("Published codebook", "deconstruct-published"),
             ("Revised codebook", "deconstruct-revised")]
    lines = []
    for label, suffix in conds:
        opus = metrics[f"anthropic-claude-opus-4-7-{suffix}"]["macro_f1"]
        gpt = metrics[f"openai-gpt-5-5-{suffix}"]["macro_f1"]
        lines.append(f"{label} & {opus:.3f} & {gpt:.3f} \\\\")
    return "\n".join(lines) + "\n"


def adjudication_table() -> str:
    expert = {r["item_id"]: r for r in read(ROOT / "data/adjudication/multilabel_gold.jsonl")}
    kinds = {"fzanartu-flicc-deberta-v2-xlarge": "Reproduced baseline",
             "frontier": "Frontier, zero-shot (revised codebook)",
             "ft": "RECoT fine-tuned",
             "base": "Base Qwen3.5 (schema-constrained)"}
    groups: dict[str, list] = {}
    for stem, name in DISPLAY.items():
        if stem not in RUNS and stem != "fzanartu-flicc-deberta-v2-xlarge":
            continue  # bare/published arms: not part of the adjudication story
        name = name.replace(" (reproduced)", "").replace(" (published)", "")
        r = assess(read(TEST / f"{stem}.jsonl"), expert)
        if stem == "fzanartu-flicc-deberta-v2-xlarge":
            kind = kinds[stem]
        else:
            kind = kinds[RUNS[stem][1]]
        groups.setdefault(kind, []).append((name, r))
    lines: list[str] = []
    for kind in kinds.values():
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{kind}}}}} \\\\")
        for name, r in sorted(groups[kind], key=lambda nr: -nr[1]["original"]):
            delta = (r["member"] - r["original"]) * 100
            lines.append(f"{name} & {r['original']:.3f} & {r['member']:.3f} & "
                         f"{delta:+.1f} & {r['reviewed']}/{r['errors']} & "
                         f"{r['rescued']}/{r['lost']} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{6}{l}{\\textit{Published (Zanartu et al.)}} \\\\")
    for name, _, acc in PUBLISHED[1:]:
        name = name.replace(" (published)", "")
        lines.append(f"{name} & {acc:.3f} & -- & -- & -- & -- \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, body in (("flicc_results.tex", results_table(only_main=True)),
                        ("flicc_results_full.tex", results_table(only_main=False)),
                        ("flicc_ablation.tex", ablation_table()),
                        ("flicc_adjudication.tex", adjudication_table())):
        (OUT / fname).write_text(body)
        print(f"wrote {OUT / fname}  ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
