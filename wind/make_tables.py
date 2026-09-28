"""Generate LaTeX table-row files for the wind chapter from the results data.

Mirrors the cards-chapter approach: emit body rows into
``wind-chapter/tables/*.tex`` that the chapter pulls in via ``\\CatchFileDef``.
Nothing is hand-typed; rerun this whenever results change.

    python make_tables.py

Outputs (under wind-chapter/tables/):
  wind_results.tex          lineup test-set table (detection + frames/claims, all rows)
  wind_augmentation.tex     val before/after ablation (4B, 9B)
  wind_perclass_frames.tex  per-frame P/R/F1 + support, 27B, support >= 3
  wind_perclass_claims.tex  per-claim P/R/F1 + support, 27B, support >= 3
"""
import json
import re
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

import generate_report as gr

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "wind-chapter" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TEST = gr.RESULTS_DIR / "test"
MIN_SUP = 3

# --- short labels from the codebook --------------------------------------
def _labels() -> dict:
    txt = (HERE / "prompts.py").read_text()
    out = {}
    for tag in ("N", "C"):
        block = re.search(rf"slim_{'frames' if tag=='N' else 'claims'}_codebook = \"\"\"(.*?)\"\"\"", txt, re.S).group(1)
        for code, lbl in re.findall(rf"<({tag}_\d+(?:_\d+)?)>\s*(.*)", block):
            for sep in (" — ", " – ", " Applies", " Triggers", " NOT ", ". ", " (e.g"):
                i = lbl.find(sep)
                if i > 0:
                    lbl = lbl[:i]
            out[code] = lbl.strip().rstrip(".")
    out["N_0"] = "No specific frame (catch-all)"
    out["C_0_1"] = "No specific claim (catch-all)"
    return out

LAB = _labels()

def esc(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

def code_tt(c: str) -> str:
    return r"\texttt{" + c.replace("_", r"\_") + "}"

# --- gold/pred lists for one level ---------------------------------------
SENTINEL = ["__PARSE_FAIL__"]
def level_lists(slug: str, level: str):
    """Return (y_true, y_pred) lists of code-lists for 'frames' or 'claims',
    over all rows, applying the same parse-fail sentinel as generate_report."""
    key = "true_frames" if level == "frames" else "true_claims"
    yt, yp = [], []
    for line in open(TEST / f"{slug}.jsonl"):
        r = json.loads(line)
        yt.append(list(r.get(key) or []))
        resp = r.get("response", "")
        if isinstance(resp, str) and resp.startswith("ERROR:"):
            yp.append(list(SENTINEL)); continue
        p = gr.parse_response(resp)
        codes = p["frames"] if level == "frames" else p["claims"]
        yp.append(list(codes) if codes is not None else list(SENTINEL))
    return yt, yp

# --- 1. lineup results table ---------------------------------------------
def results_table():
    base = [("Qwen3.5-4B", "qwen35-4b-base"),
            ("Qwen3.5-9B", "qwen35-9b-base"),
            ("Qwen3.5-27B", "qwen35-27b-base")]
    students = [("Windy-Qwen3.5-4B", "windy-qwen35-4b"),
                ("Windy-Qwen3.5-9B", "windy-qwen35-9b"),
                ("Windy-Qwen3.5-27B", "windy-qwen35-27b"),
                ("Windy-Qwen3.5-27B (FP8)", "windy-qwen35-27b-fp8")]
    reference = [("Claude Opus 4.7 (teacher)", "claude-opus-4-7"),
                 ("GPT-5.5", "openai-gpt-5-5")]
    def row(label, slug):
        m = gr.score_one(TEST / f"{slug}.jsonl")
        c = [m["detection"]["f1"], m["frames_all"]["samples_f1"], m["frames_all"]["macro_f1"],
             m["claims_all"]["samples_f1"], m["claims_all"]["macro_f1"]]
        return f"{label} & " + " & ".join(f"{x:.3f}" for x in c) + r" \\"
    def grp(title):
        return r"\multicolumn{6}{@{}l}{\textit{" + title + r"}} \\"
    lines = [grp("Open-source base (zero-shot)")] + [row(l, s) for l, s in base]
    lines += [r"\midrule", grp("Open-source RECoT fine-tuned")] + [row(l, s) for l, s in students]
    lines += [r"\midrule", grp("Frontier reference (zero-shot)")] + [row(l, s) for l, s in reference]
    (OUT / "wind_results.tex").write_text("\n".join(lines) + "\n")

# --- 2. augmentation ablation table --------------------------------------
def augmentation_table():
    d = json.load(open(gr.RESULTS_DIR / "val" / "augmentation_ablation.json"))
    pairs = d["pairs"]  # [4B, 9B]
    def cells(key):
        out = []
        for p in pairs:
            out += [f"{p['before'][key]:.3f}", f"{p['after'][key]:.3f}", f"{p['delta'][key]:+.3f}"]
        return " & ".join(out)
    L = [(r"\multirow{2}{*}{Frames}", "Samp.", "frames_opp.samples"),
         ("", "Macro", "frames_opp.macro"),
         (r"\multirow{2}{*}{Claims}", "Samp.", "claims_opp.samples"),
         ("", "Macro", "claims_opp.macro"),
         ("Detection", "F1", "detection.detection")]
    lines = []
    for lvl, met, key in L:
        lines.append(f"{lvl} & {met} & {cells(key)} " + r"\\")
        if met == "Macro":
            lines.append(r"\addlinespace")
    (OUT / "wind_augmentation.tex").write_text("\n".join(lines) + "\n")

# --- 3. per-class P/R/F1 (27B, support >= 3) -----------------------------
def perclass_table(level: str, fname: str):
    yt, yp = level_lists("windy-qwen35-27b", level)
    gold_labels = sorted({l for row in yt for l in row})
    gset = set(gold_labels)
    yp_clean = [[l for l in row if l in gset] for row in yp]
    mlb = MultiLabelBinarizer(classes=gold_labels)
    YT, YP = mlb.fit_transform(yt), mlb.transform(yp_clean)
    P, R, F, S = precision_recall_fscore_support(YT, YP, average=None, zero_division=0)
    rows = []
    for code, p, r, f, s in sorted(zip(gold_labels, P, R, F, S),
                                   key=lambda t: (int(t[0].split("_")[1]), int((t[0].split("_")+["0"])[2]))):
        if s < MIN_SUP:
            continue
        rows.append(f"{code_tt(code)} & {esc(LAB.get(code, ''))} & {p:.2f} & {r:.2f} & {f:.2f} & {int(s)} " + r"\\")
    (OUT / fname).write_text("\n".join(rows) + "\n")

# --- 4. validation error table (4B pre-aug, over-/under-predicted) -------
def val_errors_table():
    """Per-class support/predicted/FP/FN on val for the 4B pre-aug student.
    Over-applied: FP >= 17 and FP > FN. Under-recalled: FN >= 16 and FN > FP."""
    from collections import defaultdict
    sup = defaultdict(int); pred = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for line in open(gr.RESULTS_DIR / "val" / "windy-qwen35-4b-preaug.jsonl"):
        r = json.loads(line)
        gold = set(r.get("true_frames") or []) | set(r.get("true_claims") or [])
        p = gr.parse_response(r.get("response", ""))
        pf = set() if (p["frames"] is None and p["claims"] is None) else (set(p["frames"] or []) | set(p["claims"] or []))
        for c in gold: sup[c] += 1
        for c in pf: pred[c] += 1
        for c in pf - gold: fp[c] += 1
        for c in gold - pf: fn[c] += 1
    codes = set(sup) | set(pred) | set(fp) | set(fn)
    over = sorted([c for c in codes if fp[c] >= 17 and fp[c] > fn[c]], key=lambda c: -fp[c])
    under = sorted([c for c in codes if fn[c] >= 16 and fn[c] > fp[c]], key=lambda c: -fn[c])
    def row(c):
        return f"{code_tt(c)} & {esc(LAB.get(c, ''))} & {sup[c]} & {pred[c]} & {fp[c]} & {fn[c]} " + r"\\"
    lines = [r"\multicolumn{6}{@{}l}{\emph{Over-applied}}\\"] + [row(c) for c in over]
    lines += [r"\addlinespace", r"\multicolumn{6}{@{}l}{\emph{Under-recalled}}\\"] + [row(c) for c in under]
    (OUT / "wind_val_errors.tex").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    results_table()
    augmentation_table()
    val_errors_table()
    perclass_table("frames", "wind_perclass_frames.tex")
    perclass_table("claims", "wind_perclass_claims.tex")
    print("wrote:", *(p.name for p in sorted(OUT.glob("wind_*.tex"))))
