# CARDS chapter — application section (FINALIZED)

**Section title:** `Contrarian climate claims in the U.S. Congress`

**Application:** "Contrarian climate claims in the U.S. Congress."
Applies the fine-tuned CARDS classifier to **110,238 climate-related congressional
speeches (1994–2024)**; 13,661 (12.4%) carry ≥1 contrarian claim.

Finalized figures are in **`explore_cards.pdf`** (rebuilt by `explore.py`).
Reference: published Nature figure `44458_2025_29_Fig3_HTML.webp`.
Design rule: plain bars/lines, one comparison per chart, takeaway in the title;
party is the recurring lens because the partisan asymmetry is the headline.

---

## Locked: 4 figures (main text)

| # | Question | Finding | Figure | Plot type |
|---|----------|---------|--------|-----------|
| Data | corpus | 110,238 climate speeches; 12.4% contrarian | table / prose | — |
| Q1 | **What** claims, and **who** makes them? | Delay / pro-fossil dominates ("solutions won't work," "need fossil fuels"); science denial marginal; almost entirely Republican | **Fig 1** | horizontal **stacked bar**, super-claims, Democrat→Republican, counts *(= published Fig 3a)* |
| Q2 | How has it **changed over time**? | Republican rate climbs to ~40–50%; Democrats flat; gap widens | **Fig 2** | **line**, two parties |
| Q3 | **Where** is it concentrated? | Fossil/coal states (WY, KY, WV, ND, TX, AK) | **Fig 3** | **4-panel choropleth**, 2 key claims × party *(= published Fig 3b)* |
| Q4 | **What predicts** it — does fossil money track it? | Within Republicans, contrarian rate rises with fossil-fuel campaign money (≈15% → 34% across quintiles); Democrats flat | **Fig 4** | **line**, within-party money quintiles |

Plot kinds: 1 stacked bar (Fig 1), 2 lines (Figs 2, 4), 1 map grid (Fig 3).

---

## Appendix

- Top specific (Level-3) claims by party — granular drill-down (built, in `explore_cards.pdf`).
- Contrarian rate by chamber × party (House Republicans most contrarian, ~29%).

## Dropped / set aside

- **SC-7 precision audit / validity table** — set aside for now (not part of the
  finalized analysis). The money gradient (Fig 4) uses plain "% contrarian" on all
  claims, so nothing depends on the SC-7 exclusion.
- Not-intuitive gallery (already deleted): streamgraph, ridgeline, sunburst,
  treemap, diverging stacked area, overall state map, top-15 Level-2 bar.

---

## Why this set

- Reproduces the **published Figure 3** (Fig 1 + Fig 3), keeping the chapter
  consistent with the paper it is based on.
- **Adds** the two analyses the better classifier + richer metadata make possible:
  the over-time gap (Fig 2) and the **fossil-money gradient (Fig 4)** — the new,
  most distinctive result.

---

## Next step

Build the four figures paper-ready in TikZ/pgfplots from generated data files
(the wind-chapter workflow: a script emits `.dat`/`.tex` that the chapter
`\input`s — no hardcoded numbers), then draft the section prose for approval.
