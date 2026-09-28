"""Exploration deck for the CARDS Congress application.

Builds ONE multi-page explore_cards.pdf in a plain, intuitive style (bars/lines,
party splits, explanatory titles). Re-running rebuilds every page, so the deck
only grows as pages are added here; it never loses earlier work.

    python explore.py
"""
import json, pathlib
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from claims import parse_claims

PRED = "speech_cards_predictions.jsonl"
TAX = "../data/taxonomy.csv"
DEM, REP = "#2166ac", "#b2182b"
SUPER = {"1": "Not Happening", "2": "It's Not Us", "3": "Impacts Not Bad",
         "4": "Solutions Won't Work", "5": "Science Unreliable",
         "6": "Proponents Biased", "7": "Need Fossil Fuels"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 120})

def despine(ax):
    for sp in ax.spines.values():
        sp.set_color("#888")
    ax.tick_params(colors="#555")

# --- load -------------------------------------------------------------------
df = pd.DataFrame(json.loads(l) for l in open(PRED))
df["year"] = pd.to_numeric(df["date"].astype(str).str[:4], errors="coerce")
df = df[df.year.notna()].copy(); df.year = df.year.astype(int)
df["codes"] = df["content"].map(parse_claims)
df["contr"] = df["codes"].map(lambda cs: [c for c in cs if not c.startswith("0")])
df["scs"] = df["contr"].map(lambda cs: {c.split("_")[0] for c in cs})
df["is_contr"] = df["scs"].map(len) > 0
df["hard"] = df["scs"].map(lambda s: bool(s - {"7"}))
pd2 = df[df.party.isin(["Democrat", "Republican"])].copy()
LAB = dict(zip(pd.read_csv(TAX).category_number, pd.read_csv(TAX).short_label))

def stacked_party_bar(ax, dem, rep, labels, order, xlabel, title):
    y = np.arange(len(order))
    ax.barh(y, [dem[s] for s in order], color=DEM, label="Democrat")
    ax.barh(y, [rep[s] for s in order], left=[dem[s] for s in order], color=REP, label="Republican")
    xmax = max(dem[s] + rep[s] for s in order)
    for i, s in enumerate(order):
        ax.text(dem[s] + rep[s] + xmax * 0.012, i, f"D: {dem[s]:,} / R: {rep[s]:,}",
                va="center", fontsize=8, color="#444")
    ax.set_yticks(y); ax.set_yticklabels([labels[s] for s in order])
    ax.set_xlabel(xlabel); ax.set_xlim(0, xmax * 1.20); despine(ax)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(title, fontsize=12, loc="left", color="#222")

pp = PdfPages("explore_cards.pdf")

# ===== MAIN TEXT =====

# Fig 1. super-claim by party (published Fig 3a style) -----------------------
demn = {s: int(pd2[(pd2.party == "Democrat") & pd2.scs.map(lambda x: s in x)].shape[0]) for s in SUPER}
repn = {s: int(pd2[(pd2.party == "Republican") & pd2.scs.map(lambda x: s in x)].shape[0]) for s in SUPER}
order = sorted(SUPER, key=lambda s: -(demn[s] + repn[s]))
fig, ax = plt.subplots(figsize=(8.5, 4.4))
stacked_party_bar(ax, demn, repn, SUPER, order, "Number of speeches making claim",
                  "Fig 1.  Climate contrarian claims by party")
fig.tight_layout(); pp.savefig(fig); plt.close(fig)

# Fig 2. partisan gap over time ----------------------------------------------
yr = pd2.groupby(["year", "party"]).is_contr.mean().unstack() * 100
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(yr.index, yr["Republican"], color=REP, lw=2)
ax.plot(yr.index, yr["Democrat"], color=DEM, lw=2)
ax.text(yr.index[-1] + 0.3, yr["Republican"].iloc[-1], "Republicans", color=REP, va="center")
ax.text(yr.index[-1] + 0.3, yr["Democrat"].iloc[-1], "Democrats", color=DEM, va="center")
ax.set_ylabel("% of party's speeches contrarian"); ax.set_xlim(1994, 2028); despine(ax)
ax.set_title("Fig 2.  The partisan gap widens over time", fontsize=12, loc="left", color="#222")
fig.tight_layout(); pp.savefig(fig); plt.close(fig)

# Fig 3. geography: key claims x party (published Fig 3b) ---------------------
try:
    img = mpimg.imread("/tmp/cards_maps.png")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(img); ax.axis("off")
    ax.set_title("Fig 3.  Geographic distribution of key claims, by party",
                 fontsize=12, loc="left", color="#222")
    pp.savefig(fig); plt.close(fig)
except Exception as e:
    print("maps page skipped:", e)

# Fig 4. fossil-money gradient -----------------------------------------------
cc = pd2[pd.to_numeric(pd2.cc_fossilfuel, errors="coerce").notna()].copy()
cc["cc"] = pd.to_numeric(cc.cc_fossilfuel)
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for party, col in [("Republican", REP), ("Democrat", DEM)]:
    g = cc[cc.party == party].copy()
    g["q"] = pd.qcut(g.cc.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    r = g.groupby("q", observed=True)["is_contr"].mean() * 100
    ax.plot(range(5), r.values, marker="o", color=col, lw=2)
    ax.text(4.1, r.values[-1], party + "s", color=col, va="center")
ax.set_xticks(range(5)); ax.set_xticklabels(["Q1\n(least)", "Q2", "Q3", "Q4", "Q5\n(most)"])
ax.set_xlabel("Fossil-fuel campaign money (within-party quintile)")
ax.set_ylabel("% of speeches contrarian")
ax.set_xlim(-0.2, 5); ax.set_ylim(0, 35); despine(ax)
ax.set_title("Fig 4.  More fossil-fuel money, more contrarian claims (Republicans)",
             fontsize=12, loc="left", color="#222")
fig.tight_layout(); pp.savefig(fig); plt.close(fig)

# ===== APPENDIX =====

# A1. top specific (level-3) claims by party ---------------------------------
l3 = pd2.explode("contr")
l3 = l3[l3.contr.map(lambda c: isinstance(c, str) and c.count("_") == 2)]
ct = l3.groupby(["contr", "party"]).size().unstack(fill_value=0)
ct["tot"] = ct.get("Democrat", 0) + ct.get("Republican", 0)
ct = ct.sort_values("tot", ascending=False).head(12)
codes = list(ct.index)
demc = {c: int(ct.loc[c].get("Democrat", 0)) for c in codes}
repc = {c: int(ct.loc[c].get("Republican", 0)) for c in codes}
labc = {c: (LAB.get(c, c)[:42]) for c in codes}
fig, ax = plt.subplots(figsize=(9.5, 5.2))
stacked_party_bar(ax, demc, repc, labc, codes, "Number of claim mentions",
                  "Appendix.  The specific claims Congress makes, by party")
fig.tight_layout(); pp.savefig(fig); plt.close(fig)

# A2. contrarian rate by chamber x party -------------------------------------
role = pd2[pd2.role.isin(["Representative", "Senator"])]
rate = role.groupby(["role", "party"]).is_contr.mean().unstack() * 100
fig, ax = plt.subplots(figsize=(6.5, 4.4))
x = np.arange(2); w = 0.38
for off, p, col in [(-w/2, "Democrat", DEM), (w/2, "Republican", REP)]:
    vals = [rate.loc[r, p] for r in ["Representative", "Senator"]]
    ax.bar(x + off, vals, width=w, color=col, label=p)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color="#444")
ax.set_xticks(x); ax.set_xticklabels(["House", "Senate"])
ax.set_ylabel("% of speeches contrarian"); despine(ax); ax.legend(frameon=False)
ax.set_title("Appendix.  Contrarian rate by chamber and party", fontsize=12, loc="left", color="#222")
fig.tight_layout(); pp.savefig(fig); plt.close(fig)

pp.close()
print("wrote explore_cards.pdf (final: 4 figs + 2 appendix)")
