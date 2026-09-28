"""Generate data for the application-section figures from the deployment output.

Reads aqm/wind/data/application/thinktanks_wind_results.jsonl and emits pgfplots
data into wind-chapter/generated/. No hardcoded numbers in the chapter.

    python make_app_figs.py
"""
import json, re, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / "data" / "application" / "thinktanks_wind_results.jsonl"
OUT = HERE.parent.parent / "wind-chapter" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

# concise axis labels (full frame wording is in the appendix taxonomy table)
FRAME_LABEL = {
    "N_1": "Environmental harm",
    "N_2": "Harms communities",
    "N_3": "Too expensive",
    "N_4": "Inefficient, unreliable",
    "N_5": "Poorly regulated",
    "N_6": "Unethical, misleading",
    "N_7": "Energy insecurity",
    "N_8": "Health harms",
}

def listvals(body, key):
    m = re.search(rf"{key}:\s*\[(.*?)\]", body, re.S)
    if m:
        return [x.strip().strip('"\'') for x in m.group(1).split(",") if x.strip()]
    m = re.search(rf"{key}:\s*\n((?:\s*-\s*.+\n?)+)", body)
    if m:
        return [re.sub(r'^\s*-\s*', '', l).strip().strip('"\'') for l in m.group(1).splitlines() if l.strip()]
    return []

ORG_NAME = {
    "heartland_org": "Heartland",
    "cfact_org": "CFACT",
    "instituteforenergyresearch_org": "Inst. for Energy Research",
    "cei_org": "Competitive Enterprise Inst.",
    "americanexperiment_org": "American Experiment",
    "americanenergyalliance_org": "American Energy Alliance",
    "heritage_org": "Heritage Foundation",
    "countoncoal_org": "Count on Coal",
}
SYS = ["N_3", "N_4", "N_5", "N_6", "N_7"]   # systemic frames
GR  = ["N_1", "N_2", "N_8"]                 # local/grassroots frames
STACK = SYS + GR
ORDER = ["N_3", "N_4", "N_5", "N_6", "N_1", "N_2", "N_7", "N_8"]  # Fig 4 stack: global prevalence
SEG_COLOR = {"N_3": "fEcon", "N_4": "fIneff", "N_5": "fReg", "N_6": "fUneth",
             "N_7": "fSec", "N_1": "fEnv", "N_2": "fComm", "N_8": "fHealth"}

TABLES = HERE.parent.parent / "wind-chapter" / "tables"

def claim_tables():
    # clean claim text from the codebook (not hardcoded), counts + % over opposition docs
    block = re.search(r'full_claims_codebook\s*=\s*"""(.*?)"""',
                      (HERE / "prompts.py").read_text(), re.S).group(1)
    def clean(t):
        t = re.split(r"\s+—\s+| -- ", t.strip())[0]
        t = re.split(r"\.\s", t)[0]
        return re.sub(r"\s*\([^)]*\)\s*$", "", t).rstrip(". ").strip()
    LAB = {c: clean(b) for c, b in re.findall(r"<(C_\d+(?:_\d+)?)>(.*?)</C", block, re.S)}
    opp = 0; cc = collections.Counter()
    for line in open(APP):
        r = json.loads(line); wc = str(r.get("wind_content") or "")
        if not re.search(r"opposition_detected:\s*true", wc):
            continue
        opp += 1
        for c in set(listvals(wc, "claims")):
            cc[c] += 1
    def esc(s): return s.replace("&", r"\&").replace("%", r"\%")
    items = sorted([(c, n) for c, n in cc.items()
                    if c in LAB and not c.startswith("C_0_")], key=lambda x: -x[1])
    def row(c, n): return f"{esc(LAB[c])} & {n:,} & {100*n/opp:.1f} " + r"\\"
    (TABLES / "wind_top_claims.tex").write_text("\n".join(row(c, n) for c, n in items[:10]) + "\n")
    (TABLES / "wind_all_claims.tex").write_text("\n".join(row(c, n) for c, n in items) + "\n")
    print(f"claim tables: {len(items)} substantive claims, top = {items[0][0]} ({items[0][1]:,})")

def claims_econ_env_time(lo=2009, hi=2025):
    # yearly prevalence (% of that year's opposition docs) for the five leading
    # economic claims and the five leading environmental claims; feeds the
    # economic/environmental small-multiples grid (Fig. 5)
    cols = [("costs","C_20_0"),("subs","C_24_0"),("build","C_27_0"),
            ("tax","C_21_0"),("jobs","C_23_0"),
            ("unsust","C_3_0"),("land","C_4_0"),("birds","C_1_1"),
            ("wild","C_1_0"),("marine","C_1_2")]
    tot = collections.Counter(); yr = collections.defaultdict(collections.Counter)
    for line in open(APP):
        r = json.loads(line); wc = str(r.get("wind_content") or "")
        if not re.search(r"opposition_detected:\s*true", wc):
            continue
        y = str(r.get("published_date") or "")[:4]
        if not y.isdigit() or not (lo <= int(y) <= hi):
            continue
        tot[int(y)] += 1
        cs = set(listvals(wc, "claims"))
        for _, c in cols:
            if c in cs:
                yr[int(y)][c] += 1
    rows = ["year " + " ".join(k for k, _ in cols)]
    for y in sorted(yr):
        n = tot[y] or 1
        rows.append(f"{y} " + " ".join(f"{100*yr[y][c]/n:.1f}" for _, c in cols))
    (OUT / "claims_econ_env_time.dat").write_text("\n".join(rows) + "\n")

def frame_over_time(lo=2007, hi=2026):
    # yearly share of frame mentions per frame; columns ordered econ..health
    cols = [("econ","N_3"),("ineff","N_4"),("reg","N_5"),("uneth","N_6"),
            ("sec","N_7"),("env","N_1"),("comm","N_2"),("health","N_8")]
    yr = collections.defaultdict(collections.Counter)
    for line in open(APP):
        r = json.loads(line); wc = str(r.get("wind_content") or "")
        if not re.search(r"opposition_detected:\s*true", wc):
            continue
        y = str(r.get("published_date") or "")[:4]
        if y.isdigit() and lo <= int(y) <= hi:
            for f in set(listvals(wc, "frames")) & set(STACK):
                yr[int(y)][f] += 1
    rows = ["year " + " ".join(c for c, _ in cols)]
    for y in sorted(yr):
        tot = sum(yr[y][f] for _, f in cols) or 1
        rows.append(f"{y} " + " ".join(f"{100*yr[y][f]/tot:.1f}" for _, f in cols))
    (OUT / "frame_over_time.dat").write_text("\n".join(rows) + "\n")

def org_stacks(n=6):
    opp = collections.Counter()
    fbo = collections.defaultdict(collections.Counter)
    for line in open(APP):
        r = json.loads(line)
        wc = str(r.get("wind_content") or "")
        if not re.search(r"opposition_detected:\s*true", wc):
            continue
        s = r.get("spider"); opp[s] += 1
        for f in set(listvals(wc, "frames")):
            if f in STACK:
                fbo[s][f] += 1
    top = [s for s, _ in opp.most_common(n)]
    lines = []
    for rank, s in enumerate(top, start=1):
        y = n - rank                      # rank 1 at the top
        comp = fbo[s]; tot = sum(comp.values()) or 1
        lines.append(f"\\windlabel{{{y}}}{{{ORG_NAME.get(s, s)}}}{{{opp[s]:,}}}")
        x = 0.0
        for f in ORDER:
            w = 100 * comp[f] / tot
            lines.append(f"\\windseg{{{y}}}{{{x:.2f}}}{{{x+w:.2f}}}{{{SEG_COLOR[f]}}}")
            x += w
    (OUT / "frame_by_org_8.tex").write_text("\n".join(lines) + "\n")
    print("org stacks (top", n, "):", top)

def main():
    opp_docs = 0
    frame_docs = collections.Counter()
    for line in open(APP):
        r = json.loads(line)
        wc = str(r.get("wind_content") or "")
        if not re.search(r"opposition_detected:\s*true", wc):
            continue
        opp_docs += 1
        frames = {f for f in listvals(wc, "frames") if f in FRAME_LABEL}
        for f in frames:
            frame_docs[f] += 1
    # ascending by prevalence so the largest sits at the top of an xbar chart
    order = sorted(FRAME_LABEL, key=lambda f: frame_docs[f])
    coords, labels = [], []
    for i, f in enumerate(order, start=1):
        pct = 100 * frame_docs[f] / opp_docs
        coords.append(f"({pct:.1f},{i})")
        labels.append(FRAME_LABEL[f])
    # emit TikZ \windbar{row}{pct}{color}{label} calls; single colour, no
    # systemic/local grouping (that contrast is made in the prose)
    rows = []
    for i, f in enumerate(order, start=1):
        pct = 100 * frame_docs[f] / opp_docs
        rows.append(f"\\windbar{{{i}}}{{{pct:.1f}}}{{slatebar}}{{{FRAME_LABEL[f]}}}")
    (OUT / "frame_prevalence_bars.tex").write_text("\n".join(rows) + "\n")
    org_stacks()
    claims_econ_env_time()
    claim_tables()
    print("opposition docs:", opp_docs)
    for f in reversed(order):
        print(f"  {f} {FRAME_LABEL[f]:35s} {frame_docs[f]:6d}  {100*frame_docs[f]/opp_docs:5.1f}%")
    print("wrote", OUT / "frame_prevalence.tex")

if __name__ == "__main__":
    main()
