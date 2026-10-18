# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Emit LaTeX table bodies for the debunk chapter's verification study.

Reads the verification pipeline artifacts and writes row files to
dissertation/chapters/debunk/tables/, consumed via \\CatchFileDef in
preamble/chapter-macros.tex (same convention as flicc/make_tables.py).

Every number is read from a pipeline artifact; nothing is hard-coded except
the model display names and their order.

    uv run make_verification_tables.py

Writes  verification_veracity.tex  acc + MCC + macro-F1 per model, 4-class task
        verification_cred.tex      acc + MCC + macro-F1 per model, 12-class task
        verification_perclass_v.tex per-class P/R/F1/support, 4-class veracity
        verification_perclass_c.tex per-class P/R/F1/support, 12-class credibility
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # aqm/debunk
RESULTS = ROOT / "data" / "results" / "test"
OUT = ROOT.parents[1] / "dissertation" / "chapters" / "debunk" / "tables"

LINEUP = [
    ("Claude Opus 4.7 offline",            "Claude Opus 4.7"),
    ("Claude Opus 4.7 + RAG (pplx-ctx)",   "Claude Opus 4.7 + RAG"),
    ("GPT-5.5 offline",                    "GPT-5.5"),
    ("GPT-5.5 + RAG (pplx-ctx)",           "GPT-5.5 + RAG"),
    ("Qwen3.5-27B offline",                "Qwen3.5-27B"),
    ("Qwen3.5-27B + RAG (pplx-ctx)",       "Qwen3.5-27B + RAG"),
    ("GPT-4o-mini offline",                "GPT-4o-mini"),
    ("GPT-4o-mini + RAG (pplx-ctx)",       "GPT-4o-mini + RAG"),
    ("DeepSeek V4 Flash offline",          "DeepSeek V4 Flash"),
    ("DeepSeek V4 Flash + RAG (pplx-ctx)", "DeepSeek V4 Flash + RAG"),
]

VERACITY_LABELS = ["TRUE", "MISLEADING", "FALSE"]  # UNVERIFIABLE dropped: zero gold support


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def write_summary(d, var, fname):
    lines = []
    for key, disp in LINEUP:
        m = d[var].get(key, {})
        lines.append(
            f"{disp} & {fmt(m.get('accuracy'))} & {fmt(m.get('mcc'))} & "
            f"{fmt(m.get('macro_f1_present'))} \\\\"
        )
    (OUT / fname).write_text("\n".join(lines) + "\n")
    print(f"wrote {fname} ({len(lines)} rows)")


def write_perclass(d, var, labels, fname):
    lines = []
    for key, disp in LINEUP:
        m = d[var].get(key)
        if not m or "per_class" not in m:
            continue
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{disp}}}}} \\\\")
        for lbl in labels:
            c = m["per_class"].get(lbl, {})
            lines.append(
                f"\\quad {lbl} & {fmt(c.get('precision'),2)} & {fmt(c.get('recall'),2)} & "
                f"{fmt(c.get('f1'),2)} & {c.get('support','--')} \\\\"
            )
        lines.append("\\addlinespace")
    if lines and lines[-1] == "\\addlinespace":
        lines.pop()
    (OUT / fname).write_text("\n".join(lines) + "\n")
    print(f"wrote {fname} ({len(lines)} rows)")


def main():
    d = json.loads((RESULTS / "metrics_summary.json").read_text())
    assert d, "empty metrics summary"
    write_summary(d, "veracityV2", "verification_veracity.tex")
    write_summary(d, "climinator_v4", "verification_cred.tex")
    write_perclass(d, "veracityV2", VERACITY_LABELS, "verification_perclass_v.tex")
    cred_labels = sorted(
        d["climinator_v4"][next(iter(d["climinator_v4"]))]["per_class"].keys())
    write_perclass(d, "climinator_v4", cred_labels, "verification_perclass_c.tex")


if __name__ == "__main__":
    main()
