"""Single descriptive-stats + validity report for the new (cards-latest)
classifier over the congressional speech corpus. Parses `content`, joins
taxonomy short labels, reads the SC-7 audit, and writes ONE self-contained
interactive HTML (Plotly inline).

    python descriptive_stats.py   ->  descriptive_stats.html
"""
import json
import os
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.offline import get_plotlyjs

from claims import parse_claims

PRED = "speech_cards_predictions.jsonl"
TAX = "../data/taxonomy.csv"
AUDIT = "sc7_audit.jsonl"
OUT = "descriptive_stats.html"

SUPER = {"1": "Not happening", "2": "It's not us", "3": "Impacts not bad",
         "4": "Solutions won't work", "5": "Science unreliable",
         "6": "Proponents biased", "7": "Fossil fuels needed"}
DEM, REP = "#2166ac", "#b2182b"

# ---------------------------------------------------------------------------
# Load + parse
# ---------------------------------------------------------------------------
df = pd.DataFrame(json.loads(l) for l in open(PRED))
df["year"] = pd.to_numeric(df["date"].astype(str).str[:4], errors="coerce")
df = df[df.year.notna()].copy(); df.year = df.year.astype(int)
df["codes"] = df["content"].map(parse_claims)
df["contr"] = df["codes"].map(lambda cs: [c for c in cs if not c.startswith("0")])
df["scs"] = df["contr"].map(lambda cs: {c.split("_")[0] for c in cs})
df["is_contr"] = df["scs"].map(len) > 0
df["hard"] = df["scs"].map(lambda s: bool(s - {"7"}))   # contrarian excl. low-precision SC-7
N = len(df)
contr = df[df.is_contr]
pd2 = df[df.party.isin(["Democrat", "Republican"])]

tax = pd.read_csv(TAX).dropna(subset=["short_label"])
LAB = dict(zip(tax["category_number"], tax["short_label"]))
lab_l2 = lambda c: LAB.get(f"{c.split('_')[0]}_{c.split('_')[1]}_0", "_".join(c.split("_")[:2]))
lab_l3 = lambda c: LAB.get("_".join(c.split("_")[:3]), "_".join(c.split("_")[:3]))

figs = []   # (title, note_html, obj)  — obj: Figure | html-str | None(=part divider)

# ===========================================================================
figs.append(("PART I — Corpus & claim description", "", None))

# 1. Overview table
ov = [
    ("Speeches (climate-related)", f"{N:,}"),
    ("Unique speakers", f"{df['bioguide_id'].nunique():,}"),
    ("Date range", f"{df.year.min()}–{df.year.max()}"),
    ("Congresses", f"{df['congress'].nunique()}"),
    ("Party split", f"Dem {(df.party=='Democrat').sum():,} / Rep {(df.party=='Republican').sum():,}"),
    ("Chamber", f"House {(df.role=='Representative').sum():,} / Senate {(df.role=='Senator').sum():,}"),
    ("Speeches w/ ≥1 contrarian claim", f"{df.is_contr.sum():,} ({df.is_contr.mean()*100:.1f}%)"),
    ("Contrarian claims per contrarian speech (mean)", f"{contr['contr'].map(len).mean():.2f}"),
]
ov_html = "<table class='t'><tr><th>Metric</th><th>Value</th></tr>" + \
    "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in ov) + "</table>"
figs.append(("1. Corpus overview", "All speeches pre-filtered to climate-related.", ov_html))

# 2. Super-claim prevalence
sc_count = Counter(s for scs in contr["scs"] for s in scs)
order = sorted(SUPER, key=lambda s: sc_count.get(s, 0))
f = go.Figure(go.Bar(x=[sc_count[s] / len(contr) * 100 for s in order], y=[SUPER[s] for s in order],
                     orientation="h", marker_color="#4575b4",
                     text=[f"{sc_count[s]/len(contr)*100:.1f}%" for s in order], textposition="outside"))
f.update_layout(xaxis_title="% of contrarian speeches")
figs.append(("2. Super-claim prevalence", f"Among {len(contr):,} contrarian speeches (multi-label).", f))

# 3-4. Top-15 L2 / L3
for lvl, fn in [("Level 2", lab_l2), ("Level 3", lab_l3)]:
    inst = [fn(c) for cs in contr["contr"] for c in cs]
    tot = len(inst); top = Counter(inst).most_common(15)
    f = go.Figure(go.Bar(x=[c / tot for _, c in reversed(top)], y=[l for l, _ in reversed(top)],
                         orientation="h", marker_color="#762a83"))
    f.update_layout(xaxis_title="Proportion of all contrarian claims", height=520, margin=dict(l=10))
    figs.append((f"3. Top-15 {lvl} claims", f"Share of {tot:,} contrarian claim mentions.", f))

# 5. Contrarian over time
by_year = df.groupby("year")["is_contr"].mean() * 100
f = go.Figure(go.Scatter(x=by_year.index, y=by_year.values, mode="lines+markers", line_color="#d73027"))
f.update_layout(xaxis_title="Year", yaxis_title="% of speeches contrarian")
figs.append(("4. Contrarian discourse over time", "Speeches making any contrarian claim, by year.", f))

# 6. Super-claim over time
yt = df.explode("scs")
pivot = (yt[yt.scs.notna()].assign(one=1).pivot_table(index="year", columns="scs", values="one", aggfunc="count", fill_value=0))
share = pivot.div(df.groupby("year").size(), axis=0) * 100
f = go.Figure()
for s in sorted(SUPER):
    if s in share:
        f.add_trace(go.Scatter(x=share.index, y=share[s], mode="lines", name=SUPER[s]))
f.update_layout(xaxis_title="Year", yaxis_title="% of speeches")
figs.append(("5. Each super-claim over time", "Proportion of speeches per year, by super-claim.", f))

# 7. By party
rate = pd2.groupby("party")["is_contr"].mean() * 100
f = go.Figure(go.Bar(x=["Democrat", "Republican"], y=[rate["Democrat"], rate["Republican"]],
                     marker_color=[DEM, REP], text=[f"{rate['Democrat']:.1f}%", f"{rate['Republican']:.1f}%"], textposition="outside"))
f.update_layout(yaxis_title="% of speeches contrarian")
figs.append(("6. Contrarian rate by party", "Share of each party's speeches that are contrarian.", f))

# 8. Super-claim mix by party
f = go.Figure()
for party, col in [("Democrat", DEM), ("Republican", REP)]:
    sub = pd2[pd2.party == party]; nc = sub.is_contr.sum()
    cc = Counter(s for scs in sub[sub.is_contr]["scs"] for s in scs)
    f.add_trace(go.Bar(name=party, x=[SUPER[s] for s in sorted(SUPER)],
                       y=[cc.get(s, 0) / nc * 100 for s in sorted(SUPER)], marker_color=col))
f.update_layout(barmode="group", yaxis_title="% of party's contrarian speeches", xaxis_tickangle=-30)
figs.append(("7. Super-claim mix by party", "Within each party's contrarian speeches.", f))

# 9. Party x time
f = go.Figure()
for party, col in [("Democrat", DEM), ("Republican", REP)]:
    s = pd2[pd2.party == party].groupby("year")["is_contr"].mean() * 100
    f.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=party, line_color=col))
f.update_layout(xaxis_title="Year", yaxis_title="% of speeches contrarian")
figs.append(("8. Contrarian rate by party over time", "", f))

# 10. Chamber
ch = df[df.role.isin(["Representative", "Senator"])].groupby("role")["is_contr"].mean() * 100
f = go.Figure(go.Bar(x=ch.index, y=ch.values, marker_color="#1b7837",
                     text=[f"{v:.1f}%" for v in ch.values], textposition="outside"))
f.update_layout(yaxis_title="% of speeches contrarian")
figs.append(("9. Contrarian rate by chamber", "", f))

# 11. Geography choropleth
ST = df[df.state.str.fullmatch(r"[A-Z]{2}", na=False)]
gr = ST.groupby("state")["is_contr"].mean() * 100
f = px.choropleth(locations=gr.index, locationmode="USA-states", color=gr.values, scope="usa",
                  color_continuous_scale="Reds", labels={"color": "% contrarian"})
figs.append(("10. Contrarian rate by state", "Share of each state's climate speeches that are contrarian.", f))

# ===========================================================================
figs.append(("PART II — Validity & robustness", "", None))

# 12. SC-7 audit
if os.path.exists(AUDIT):
    aud = [json.loads(l) for l in open(AUDIT)]
    rows = []
    for p in ["Democrat", "Republican"]:
        sub = [r for r in aud if r["party"] == p and r["genuine"] is not None]
        tp = sum(1 for r in sub if r["genuine"]); rows.append((p, len(sub), tp, tp / len(sub) * 100))
    tbl = "<table class='t'><tr><th>Party</th><th>n</th><th>True pos.</th><th>Precision</th></tr>" + \
        "".join(f"<tr><td>{p}</td><td>{n}</td><td>{tp}</td><td><b>{pr:.0f}%</b></td></tr>" for p, n, tp, pr in rows) + "</table>"
    modes = {p: Counter(r["mode"] for r in aud if r["party"] == p) for p in ["Democrat", "Republican"]}
    allm = ["endorses", "neutral_mention", "rebuts", "off_target"]
    f = go.Figure()
    for p, col in [("Democrat", DEM), ("Republican", REP)]:
        t = sum(modes[p].values())
        f.add_trace(go.Bar(name=p, x=allm, y=[modes[p].get(m, 0) / t * 100 for m in allm], marker_color=col))
    f.update_layout(barmode="group", yaxis_title="% of sampled SC-7 tags")
    figs.append(("11. SC-7 precision audit (independent Claude judge)",
                 "Republican SC-7 is reliable (85%); Democratic SC-7 is only ~52% genuine — mostly "
                 "neutral mentions and rebuttals. Treat Democratic 'Fossil fuels needed' with caution." + tbl, f))

# 13. Partisan gap robust to SC-7
rr = pd2.groupby("party")[["is_contr", "hard"]].mean() * 100
f = go.Figure()
f.add_trace(go.Bar(name="All claims", x=["Democrat", "Republican"], marker_color="#bdbdbd",
                   y=[rr.loc["Democrat", "is_contr"], rr.loc["Republican", "is_contr"]],
                   text=[f"{rr.loc['Democrat','is_contr']:.1f}%", f"{rr.loc['Republican','is_contr']:.1f}%"], textposition="outside"))
f.add_trace(go.Bar(name="Excl. SC-7 (robust)", x=["Democrat", "Republican"], marker_color=[DEM, REP],
                   y=[rr.loc["Democrat", "hard"], rr.loc["Republican", "hard"]],
                   text=[f"{rr.loc['Democrat','hard']:.1f}%", f"{rr.loc['Republican','hard']:.1f}%"], textposition="outside"))
f.update_layout(barmode="group", yaxis_title="% of speeches contrarian")
figs.append(("12. Partisan gap is robust to the SC-7 problem",
             f"Dropping low-precision SC-7 <i>widens</i> the gap to "
             f"{rr.loc['Republican','hard']/rr.loc['Democrat','hard']:.0f}× "
             f"(from {rr.loc['Republican','is_contr']/rr.loc['Democrat','is_contr']:.0f}×).", f))

# 14. Denial vs delay over time
GROUPS = [("Denial (not happening/us/bad)", {"1", "2", "3"}, "#2166ac", "solid"),
          ("Science/process attacks", {"5", "6"}, "#f1a340", "solid"),
          ("Solutions won't work (SC-4)", {"4"}, "#b2182b", "solid"),
          ("Fossil fuels needed (SC-7, low prec.)", {"7"}, "#fdae61", "dot")]
tot_y = df.groupby("year").size()
f = go.Figure()
for name, grp, col, dash in GROUPS:
    s = df[df.scs.map(lambda x: bool(x & grp))].groupby("year").size().reindex(tot_y.index, fill_value=0) / tot_y * 100
    f.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name, line=dict(color=col, width=2.5, dash=dash)))
f.update_layout(xaxis_title="Year", yaxis_title="% of all speeches")
figs.append(("13. Denial vs. delay over time",
             "Lamb et al. (2020) framing. Hard denial stays marginal; 'delay' dominates — and the rise "
             "holds on the reliable SC-4, independent of the noisier SC-7 (dotted).", f))

# 15. Within-party money gradient (robust)
cc = pd2[pd.to_numeric(pd2.cc_fossilfuel, errors="coerce").notna()].copy()
cc["cc"] = pd.to_numeric(cc.cc_fossilfuel)
f = go.Figure()
for party, col in [("Democrat", DEM), ("Republican", REP)]:
    g = cc[cc.party == party].copy()
    g["q"] = pd.qcut(g.cc.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    r = g.groupby("q", observed=True)["hard"].mean() * 100
    f.add_trace(go.Scatter(x=list(r.index), y=r.values, mode="lines+markers", name=party, line_color=col, line_width=2.5))
f.update_layout(xaxis_title="Fossil-fuel campaign $ (within-party quintile)", yaxis_title="% contrarian (excl. SC-7)")
figs.append(("14. Fossil-money gradient, within party (robust)",
             "Contrarian rate by within-party fossil-money quintile, excluding low-precision SC-7. "
             "Holds for Republicans (9.5%→18.6%); weak but positive for Democrats. Descriptive, not causal.", f))

# ---------------------------------------------------------------------------
# Assemble single HTML
# ---------------------------------------------------------------------------
import re
blocks, n_sec = [], 0
for title, note, obj in figs:
    if obj is None:
        blocks.append(f"<h2 class='part'>{title}</h2>")
        continue
    n_sec += 1
    title = f"{n_sec}. " + re.sub(r"^\d+\.\s*", "", title)   # sequential numbering
    body = obj if isinstance(obj, str) else pio.to_html(obj, include_plotlyjs=False, full_html=False,
                                                        default_height=440, config={"displayModeBar": False})
    note_html = f"<div class='note'>{note}</div>" if note else ""
    blocks.append(f"<section><h3>{title}</h3>{note_html}{body}</section>")

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Contrarian climate claims in Congress — descriptive stats</title>
<script>{get_plotlyjs()}</script>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1f2937;background:#f8fafc;margin:0}}
 .wrap{{max-width:980px;margin:0 auto;padding:30px 22px 80px}}
 h1{{font-size:25px;margin:0 0 4px}} .lead{{color:#6b7280;margin:0 0 10px}}
 h2.part{{font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:#6b7280;
   margin:34px 0 12px;padding-bottom:6px;border-bottom:2px solid #e5e7eb;
   break-before:page;page-break-before:always}}
 h2.part:first-of-type{{break-before:auto;page-break-before:auto}}
 section{{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:18px 20px;
   margin:0 0 22px;box-shadow:0 1px 3px rgba(0,0,0,.05);
   break-inside:avoid;page-break-inside:avoid}}
 h3{{font-size:17px;margin:0 0 4px}} .note{{color:#6b7280;font-size:13px;margin:0 0 10px}}
 table.t{{border-collapse:collapse;margin:8px 0 2px}}
 table.t th,table.t td{{border-bottom:1px solid #eee;padding:6px 16px 6px 0;text-align:left}}
 table.t th{{color:#6b7280;font-size:12px;text-transform:uppercase}}
</style></head><body><div class="wrap">
<h1>Contrarian climate claims in U.S. congressional speech</h1>
<p class="lead">{N:,} climate-related speeches, {df.year.min()}–{df.year.max()}, classified by the
fine-tuned cards-latest model. Super-claim 0 (“no claim”) excluded from contrarian cuts.</p>
{''.join(blocks)}
</div></body></html>"""
open(OUT, "w").write(html)
print(f"wrote {OUT}  ({len(html)//1024} KB, {sum(o is not None for _,_,o in figs)} figures)")
