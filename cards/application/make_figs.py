"""Render the application figures paper-ready into cards-chapter/figures/.

Plain bars/lines, party splits; no in-chart titles (the LaTeX caption carries
the message). Vector PDF for the charts; high-DPI PNG for the choropleth maps
(rendered via plotly + Playwright, the chapter's existing path).

    python make_figs.py
"""
import json, pathlib
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent.parent.parent / "cards-chapter" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PRED = HERE / "speech_cards_predictions.jsonl"
TAX = HERE.parent / "data" / "taxonomy.csv"
import sys; sys.path.insert(0, str(HERE))
from claims import parse_claims

DEM, REP = "#2166ac", "#b2182b"
SUPER = {"1": "Not Happening", "2": "It's Not Us", "3": "Impacts Not Bad",
         "4": "Solutions Won't Work", "5": "Science Unreliable",
         "6": "Proponents Biased", "7": "Need Fossil Fuels"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})

def despine(ax):
    for sp in ax.spines.values():
        sp.set_color("#888")
    ax.tick_params(colors="#444")

df = pd.DataFrame(json.loads(l) for l in open(PRED))
df["year"] = pd.to_numeric(df["date"].astype(str).str[:4], errors="coerce")
df = df[df.year.notna()].copy(); df.year = df.year.astype(int)
df["codes"] = df["content"].map(parse_claims)
df["contr"] = df["codes"].map(lambda cs: [c for c in cs if not c.startswith("0")])
df["scs"] = df["contr"].map(lambda cs: {c.split("_")[0] for c in cs})
df["is_contr"] = df["scs"].map(len) > 0
pd2 = df[df.party.isin(["Democrat", "Republican"])].copy()

# --- Fig 1: contrarian claims by party (stacked, Dem first) -----------------
demn = {s: int(pd2[(pd2.party == "Democrat") & pd2.scs.map(lambda x: s in x)].shape[0]) for s in SUPER}
repn = {s: int(pd2[(pd2.party == "Republican") & pd2.scs.map(lambda x: s in x)].shape[0]) for s in SUPER}
order = sorted(SUPER, key=lambda s: -(demn[s] + repn[s]))
fig, ax = plt.subplots(figsize=(6.6, 3.4))
y = np.arange(len(order))
ax.barh(y, [demn[s] for s in order], color=DEM, label="Democrat")
ax.barh(y, [repn[s] for s in order], left=[demn[s] for s in order], color=REP, label="Republican")
xmax = max(demn[s] + repn[s] for s in order)
for i, s in enumerate(order):
    ax.text(demn[s] + repn[s] + xmax * 0.012, i, f"D: {demn[s]:,} / R: {repn[s]:,}",
            va="center", fontsize=7.5, color="#444")
ax.set_yticks(y); ax.set_yticklabels([SUPER[s] for s in order])
ax.set_xlabel("Number of paragraphs"); ax.set_xlim(0, xmax * 1.22)
despine(ax); ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.savefig(FIG / "cards_claims_by_party.pdf"); plt.close(fig)

# --- Fig 2: partisan gap over time ------------------------------------------
yr = pd2.groupby(["year", "party"]).is_contr.mean().unstack() * 100
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.plot(yr.index, yr["Republican"], color=REP, lw=2)
ax.plot(yr.index, yr["Democrat"], color=DEM, lw=2)
ax.text(yr.index[-1] + 0.4, yr["Republican"].iloc[-1], "Republicans", color=REP, va="center", fontsize=10)
ax.text(yr.index[-1] + 0.4, yr["Democrat"].iloc[-1], "Democrats", color=DEM, va="center", fontsize=10)
ax.set_ylabel("% of paragraphs contrarian"); ax.set_xlim(1994, 2029); despine(ax)
fig.savefig(FIG / "cards_gap_over_time.pdf"); plt.close(fig)

# --- Fig 4: fossil-money gradient -------------------------------------------
cc = pd2[pd.to_numeric(pd2.cc_fossilfuel, errors="coerce").notna()].copy()
cc["cc"] = pd.to_numeric(cc.cc_fossilfuel)
fig, ax = plt.subplots(figsize=(6.0, 3.4))
for party, col in [("Republican", REP), ("Democrat", DEM)]:
    g = cc[cc.party == party].copy()
    g["q"] = pd.qcut(g.cc.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    r = g.groupby("q", observed=True)["is_contr"].mean() * 100
    ax.plot(range(5), r.values, marker="o", color=col, lw=2)
    ax.text(4.12, r.values[-1], party + "s", color=col, va="center", fontsize=10)
ax.set_xticks(range(5)); ax.set_xticklabels(["Q1\n(least)", "Q2", "Q3", "Q4", "Q5\n(most)"])
ax.set_xlabel("Fossil-fuel campaign money (within-party quintile)")
ax.set_ylabel("% of paragraphs contrarian"); ax.set_xlim(-0.2, 5.2); ax.set_ylim(0, 35); despine(ax)
fig.savefig(FIG / "cards_money_gradient.pdf"); plt.close(fig)

print("wrote 3 vector figures to", FIG)

# --- Fig 3: choropleth maps (plotly + Playwright) ---------------------------
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    members = pd2.groupby(["state", "party"]).bioguide_id.nunique()
    CLAIMS = {"4": "Solutions Won't Work", "7": "Need Fossil Fuels"}
    PARTY = [("Democrat", "Blues"), ("Republican", "Reds")]
    titles = [f"{n} — {p}s" for n in CLAIMS.values() for p, _ in PARTY]
    figm = make_subplots(rows=2, cols=2, specs=[[{"type": "choropleth"}] * 2] * 2,
                         subplot_titles=titles, horizontal_spacing=0.03, vertical_spacing=0.07)
    for ri, (c, cn) in enumerate(CLAIMS.items()):
        for ci, (party, scale) in enumerate(PARTY):
            sub = pd2[pd2.party == party]
            cnt = sub[sub.scs.map(lambda x: c in x)].groupby("state").size()
            mem = members.xs(party, level="party")
            metric = (cnt / mem).replace([np.inf], np.nan).dropna()
            figm.add_trace(go.Choropleth(locations=metric.index, z=metric.values,
                locationmode="USA-states", colorscale=scale, marker_line_color="white",
                marker_line_width=0.4, colorbar=dict(title="claims<br>per member", len=0.42,
                thickness=9, x=(0.46 if ci == 0 else 1.0), y=(0.79 if ri == 0 else 0.22))),
                row=ri + 1, col=ci + 1)
    figm.update_geos(scope="usa", bgcolor="white", showlakes=False)
    figm.update_layout(margin=dict(t=28, l=6, r=6, b=6), height=620, width=1000,
                       font=dict(size=12), paper_bgcolor="white")
    for a in figm.layout.annotations:
        a.font.size = 13
    figm.write_html(HERE / "maps.html", include_plotlyjs="cdn", full_html=True)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1000, "height": 620}, device_scale_factor=2)
        pg.goto((HERE / "maps.html").resolve().as_uri(), wait_until="networkidle")
        pg.wait_for_timeout(2500)
        pg.locator(".main-svg").first.screenshot(path=str(FIG / "cards_maps.png"))
        b.close()
    print("wrote cards_maps.png to", FIG)
except Exception as e:
    print("maps skipped:", e)
