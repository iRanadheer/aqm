"""Per-class precision/recall/F1 appendix tables for the cards chapter.

Table A (perclass_l1.tex): per super-claim, model coverage (CARDS-27B, Opus 4.6,
GPT-5.5), P/R/F1 each + shared support.
Table B (perclass_l2.tex): per sub-claim for CARDS-27B, grouped by super-claim,
P/R/F1 + support, restricted to support >= 3.

Per-class rows are the components of Macro F1 (their mean = Macro), NOT Samples.
"""
import os
from collections import Counter

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from generate_report import load_canonical_gold, parse_response, RESULTS_DIR

SUPER = {"0": "No claim", "1": "Not happening", "2": "It's not us",
         "3": "Impacts not bad", "4": "Solutions won't work",
         "5": "Science unreliable", "6": "Proponents biased", "7": "Fossil fuels needed"}
L1_ORDER = ["1", "2", "3", "4", "5", "6", "7", "0"]
L1_MODELS = [("CARDS-Qwen3.5-27B", "cards-qwen35-27b"),
             ("Claude Opus 4.6", "claude-opus-4-6"), ("GPT-5.5", "gpt-5-5")]

tax = pd.read_csv("data/taxonomy.csv").dropna(subset=["short_label"])
LAB = dict(zip(tax["category_number"], tax["short_label"]))
gold = load_canonical_gold()


def trunc(codes, level):
    return sorted({"_".join(c.split("_")[:level]) for c in codes})


def yt_yp(slug, level):
    df = pd.read_json(f"{RESULTS_DIR}/test/{slug}.jsonl", lines=True)
    df = df[df.text.isin(gold)]
    yt = [trunc(gold[t], level) for t in df.text]
    yp = [trunc(parse_response(r), level) for r in df.response]
    return yt, yp


def per_class(yt, yp, labels):
    """P/R/F1/support arrays aligned to `labels` (gold-only vocab)."""
    s = set(labels)
    mlb = MultiLabelBinarizer(classes=labels)
    YT = mlb.fit_transform([[c for c in r if c in s] for r in yt])
    YP = mlb.transform([[c for c in r if c in s] for r in yp])
    P, R, F, S = precision_recall_fscore_support(YT, YP, average=None, zero_division=0)
    return {l: (P[i], R[i], F[i], int(S[i])) for i, l in enumerate(labels)}


def emit(basename, lines):
    p = f"{RESULTS_DIR}/test/{basename}.tex"
    open(p, "w").write("\n".join(lines) + "\n")
    print("  ->", p)


# --- Table A: Level 1, model coverage --------------------------------------
percls = {name: per_class(*yt_yp(slug, 1), L1_ORDER) for name, slug in L1_MODELS}
rows = []
for s in L1_ORDER:
    sup = percls[L1_MODELS[0][0]][s][3]
    cells = []
    for name, _ in L1_MODELS:
        p, r, f, _ = percls[name][s]
        cells += [f"{p:.2f}", f"{r:.2f}", f"{f:.2f}"]
    rows.append(f"{SUPER[s]:<22s} & {sup:>4d} & " + " & ".join(cells) + r" \\")
emit("perclass_l1", rows)

# --- Table B: Level 2, CARDS-27B granularity (support >= 3, grouped) --------
yt2, yp2 = yt_yp("cards-qwen35-27b", 2)
sup2 = Counter(c for r in yt2 for c in r)
keep = sorted([l for l, n in sup2.items() if n >= 3])
pc2 = per_class(yt2, yp2, keep)
lines = []
for sc in L1_ORDER:
    subs = sorted([l for l in keep if l.split("_")[0] == sc], key=lambda x: int(x.split("_")[1]))
    if not subs:
        continue
    if lines:
        lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{SUPER[sc]}}}}} \\")
    for l in subs:
        lab = (LAB.get(f"{l}_0", l).replace("&", r"\&")
               .replace("/", r"/\allowbreak "))   # let long slash-runs wrap
        p, r, f, n = pc2[l]
        lines.append(f"{lab} & {n} & {p:.2f} & {r:.2f} & {f:.2f} " + r"\\")
emit("perclass_l2", lines)

# copy to chapter
dst = "/home/projects/overbeef/cards-chapter/tables"
for b in ("perclass_l1", "perclass_l2"):
    os.system(f"cp {RESULTS_DIR}/test/{b}.tex {dst}/{b}.tex")
print("copied to", dst)
