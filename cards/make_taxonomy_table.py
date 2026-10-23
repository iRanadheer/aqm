# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Emit the revised CARDS taxonomy as LaTeX longtable bodies for the dissertation.

Reads aqm/cards/data/taxonomy.csv (codes + labels) and
aqm/cards/data/taxonomy_mapping.json (old/new status per code), and writes two
files into dissertation/chapters/cards/, consumed via \\input in
chapters/cards/appendix.tex:

  taxonomy_table.tex    flat Code | Claim list, colour-coded by status
  taxonomy_nested.tex   nested Super-claim | Sub-claim | Sub-sub-claim | Claim,
                        colour-coded by status, parents written once

The super-claims at codes 4_0_0 and 6_0_0 have blank short_labels in the CSV;
the condensed names below are supplied to match the other super-claim labels.

    python make_taxonomy_table.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data"   # aqm/cards/data
OUT = ROOT.parents[2] / "dissertation" / "chapters" / "cards" / "taxonomy_table.tex"

# Colour macro applied to both the code and its label, by relationship to the
# original Coan et al. (2021) taxonomy.
STYLE = {
    "unchanged":  ("", ""),
    "relabelled": ("\\textcolor{relabelled}", "\\textcolor{relabelled}"),
    "new":        ("\\textcolor{newcode}", "\\textcolor{newcode}"),
    "catch-all":  ("", ""),
}

# Condensed short labels for super-claims whose short_label is blank in the CSV.
OVERRIDE = {
    "4_0_0": "Climate solutions are harmful or unnecessary",
    "6_0_0": "Climate scientists and proponents are alarmist or corrupt",
}


def tex_escape(s):
    """Escape characters that break LaTeX table cells."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def key(code):
    return tuple(int(x) for x in code.split("_"))


def indent(code):
    """Indentation by taxonomy depth: super-claim flush, sub-claim +1, deeper +2."""
    parts = code.split("_")
    if len(parts) == 3 and parts[1] == "0" and parts[2] == "0":
        return 0
    if len(parts) == 3 and parts[2] == "0":
        return 1
    return 2


PAD = {0: "", 1: "\\hspace{0.6em}", 2: "\\hspace{1.2em}"}


def colorize(code, label, status):
    ccode, clabel = STYLE[status.get(code, "unchanged")]
    ccell = (ccode + "{\\texttt{" + code.replace("_", chr(92) + "_") + "}}") if ccode \
        else ("\\texttt{" + code.replace("_", chr(92) + "_") + "}")
    lcell = (clabel + "{" + label + "}") if clabel else label
    return ccell, lcell


def parent(code):
    p = code.split("_")
    if p[1] == "0" and p[2] == "0":
        return None
    if p[2] == "0":
        return p[0] + "_0_0"
    return p[0] + "_" + p[1] + "_0"


def is_super(code):
    p = code.split("_")
    return p[1] == "0" and p[2] == "0"


def label_of(r):
    return tex_escape(OVERRIDE.get(r["category_number"])
                      or r["short_label"].strip() or r["prompt_label"].strip())


def write_nested(seen, status):
    """Three columns: Super-claim | Sub-claim | Sub-sub-claim. Parent labels are
    written once, and a heavier rule marks each super-claim boundary; sub-claim
    groups are separated by the sub-claim label itself rather than by extra rules,
    which keeps row heights even."""
    out = []

    def row(cells):
        cells = (cells + [""] * 3)[:3]
        out.append(" & ".join(cells) + r" \\")

    supers = sorted([c for c in seen if is_super(c) and c != "0_0_0"], key=key)
    for si, s in enumerate(supers):
        if si > 0:
            out.append(r"\hline")
        sc, sl = colorize(s, label_of(seen[s]), status)
        s_first = True
        for sub in sorted([c for c in seen if parent(c) == s], key=key):
            bc, bl = colorize(sub, label_of(seen[sub]), status)
            leaves = sorted([c for c in seen if parent(c) == sub], key=key)
            if not leaves:
                row(([f"{sc} {sl}"] if s_first else [""]) + [f"{bc} {bl}", ""])
            else:
                for i, leaf in enumerate(leaves):
                    lc, ll = colorize(leaf, label_of(seen[leaf]), status)
                    if leaf.count("_") - 2 > 0:
                        ll = "\\hspace{1.1em}" + ll
                    s_cell = [f"{sc} {sl}"] if (s_first and i == 0) else [""]
                    b_cell = f"{bc} {bl}" if i == 0 else ""
                    row(s_cell + [b_cell, f"{lc} {ll}"])
            s_first = False
    return out


def main():
    rows = list(csv.DictReader((ROOT / "taxonomy.csv").open()))
    seen = {}
    for r in rows:
        seen.setdefault(r["category_number"], r)

    status = {e["new_code"]: e["status"] for e in
              json.loads((ROOT / "taxonomy_mapping.json").read_text())["mapping"]}

    lines = []
    n_super = n_sub = n_deep = 0
    for code in sorted(seen, key=key):
        r = seen[code]
        label = (OVERRIDE.get(code) or r["short_label"].strip()
                 or r["prompt_label"].strip())
        d = indent(code)
        if d == 0 and code != "0_0_0":
            n_super += 1
        elif d == 1:
            n_sub += 1
        elif d == 2:
            n_deep += 1
        stars = {0: "", 1: "\\hspace{0.6em}", 2: "\\hspace{1.2em}"}[d]
        label = tex_escape(label)
        ccode, clabel = STYLE[status.get(code, "unchanged")]
        ccell = (ccode + "{\\texttt{" + code.replace("_", chr(92) + "_") + "}}") if ccode \
            else ("\\texttt{" + code.replace("_", chr(92) + "_") + "}")
        lcell = (clabel + "{" + stars + label + "}") if clabel else (stars + label)
        lines.append(f"{ccell} & {lcell} \\\\")

    (OUT.parent / "taxonomy_table.tex").write_text(
        "\\begin{longtable}{@{}p{2.0cm} p{12.0cm}@{}}\n"
        "\\caption{The revised CARDS taxonomy: %d super-claims and %d claims, plus a "
        "catch-all no-claim code. Blue entries are unchanged from \\citet{coan2021computer}; "
        "orange entries were relabelled or re-parented; magenta entries are new.}"
        "\\label{tab:cards-taxonomy}\\\\\n"
        "\\toprule\n"
        "\\textbf{Code} & \\textbf{Claim} \\\\ \\midrule\n"
        "\\endfirsthead\n"
        "\\toprule \\textbf{Code} & \\textbf{Claim} \\\\ \\midrule \\endhead\n"
        "\\bottomrule \\endfoot\n"
        % (n_super, len(seen) - n_super - 1)
        + "\n".join(lines) + "\n\\end{longtable}\n"
    )
    print(f"wrote taxonomy_table.tex: {n_super} super, {n_sub} sub, {n_deep} sub-sub "
          f"({len(seen)} unique codes)")

    nested = write_nested(seen, status)
    (OUT.parent / "taxonomy_nested.tex").write_text(
        "\\begingroup\\scriptsize\\setlength{\\tabcolsep}{4pt}"
        "\\renewcommand{\\arraystretch}{1.0}\n"
        "\\begin{longtable}{|>{\\raggedright\\arraybackslash}p{3.0cm}|"
        ">{\\raggedright\\arraybackslash}p{3.9cm}|"
        ">{\\raggedright\\arraybackslash}p{7.9cm}|@{}}\n"
        "\\caption{The revised CARDS taxonomy as a hierarchy. Claims unchanged from "
        "\\citet{coan2021computer} appear in black; orange entries were relabelled or "
        "re-parented; magenta entries are new. Each super-claim and sub-claim is written "
        "once, on the first row to which it applies.}\label{tab:cards-taxonomy-nested}\\\\\n"
        "\\hline\n"
        "\\textbf{Super-claim} & \\textbf{Sub-claim} & \\textbf{Sub-sub-claim} \\\\ \\hline\n"
        "\\endfirsthead\n"
        "\\hline \\textbf{Super-claim} & \\textbf{Sub-claim} & \\textbf{Sub-sub-claim} \\\\ \\hline \\endhead\n"
        + "\n".join(nested) + "\n\\hline\n\\end{longtable}\n\\endgroup\n"
    )
    print(f"wrote taxonomy_nested.tex ({len(nested)} rows)")


if __name__ == "__main__":
    main()
