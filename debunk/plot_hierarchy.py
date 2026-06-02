"""Charts for the debunk benchmark.

Produces (up to) two PNGs under `data/results/test/`:
  - `debunk_benchmark.png` — one figure, two rows:
      Row 1: climinator across the 4-level Leippold 2025 credibility
             hierarchy (L1=12 → L2=5 → L3=3 → L4=2), Acc + Macro-F1,
             with Paper CLIM as a reference tick per level.
      Row 2: veracityV2 (flat 4-class), Acc + Macro-F1, no reference
             tick (Leippold 2025 does not publish a veracity benchmark).
    Paired offline/RAG bars per model family, shared legend.
  - `climinator_online_vs_offline.png` — stacked-bar comparison of
    online (web-search) vs offline runs, only emitted when both halves
    of a model pair are available.

Usage:
  python3 plot_hierarchy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "data" / "results" / "test"

# Paper baseline is no longer hardcoded — it's recomputed dynamically by
# generate_report.py from the `true_climinator` column of test.jsonl on the
# same 160-row subset we score our models on, and arrives in the summary as
# the regular model entry "Paper CLIM (recomputed)". This keeps the
# comparison strictly fair (same rows, same scoring code, same denominator).
PAPER_RESULTS: dict[str, dict[str, dict[int, float]]] = {}

# Shared model family ordering — left→right, light=offline, dark=RAG.
# Used by both the climinator-hierarchy and veracity bar charts so the
# two figures read as a matched pair.
FAMILIES = [
    ("GPT-4o-mini",         "GPT-4o-mini offline",          "GPT-4o-mini + RAG (pplx-ctx)"),
    ("Qwen3.5-9B",          "Qwen3.5-9B offline",           "Qwen3.5-9B + RAG (pplx-ctx)"),
    ("Qwen3.5-27B",         "Qwen3.5-27B offline",          "Qwen3.5-27B + RAG (pplx-ctx)"),
    ("DeepSeek V4 Flash",   "DeepSeek V4 Flash offline",    "DeepSeek V4 Flash + RAG (pplx-ctx)"),
    ("Claude Opus 4.7",     "Claude Opus 4.7 offline",      "Claude Opus 4.7 + RAG (pplx-ctx)"),
    ("GPT-5.5",             "GPT-5.5 offline",              "GPT-5.5 + RAG (pplx-ctx)"),
]
FAMILY_COLOURS = ["#888888", "#2ca02c", "#17becf", "#9467bd", "#d62728", "#1f77b4"]
# grey, green, cyan, purple, red, blue — Qwens grouped by adjacent cool hues


def plot_bars() -> Path:
    summary = json.loads((RESULTS_DIR / "metrics_summary.json").read_text())
    # Merge climinator (original) + climinator_v4 (current frozen variant).
    # v4 takes precedence on name collision. Earlier prompt variants
    # (v2, v3, v5) are intentionally excluded from headline plots — see
    # methods discussion of why v4 was chosen.
    clim = {**summary.get("climinator", {}), **summary.get("climinator_v4", {})}
    if not clim:
        raise SystemExit("no climinator results in metrics_summary.json")

    levels = ["1", "2", "3", "4"]
    level_labels = ["L1", "L2", "L3", "L4"]
    metrics = [("accuracy",         "Accuracy"),
               ("macro_f1_present", "Macro-F1 (classes present)")]
    PAPER_KEY = "Paper CLIM (recomputed)"
    paper_entry = clim.get(PAPER_KEY)

    # Family ordering and colours are module constants (FAMILIES,
    # FAMILY_COLOURS) so the climinator chart and the veracity chart read
    # as a matched pair. Within each family: offline = lighter shade, RAG
    # = darker shade. A small inter-family gap makes the pairings obvious.

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    x = np.arange(len(levels))
    # 2 bars per family, hairline gap between families
    bar_w = 0.09
    pair_gap = 0.005
    family_gap = 0.04
    pair_span = 2 * bar_w + pair_gap  # one offline + one RAG
    total_span = len(FAMILIES) * pair_span + (len(FAMILIES) - 1) * family_gap

    legend_seen: set[str] = set()

    for ax, (key, title) in zip(axes, metrics):
        for fi, (fam_name, off_key, rag_key) in enumerate(FAMILIES):
            base_x = -total_span / 2 + fi * (pair_span + family_gap)
            for vi, (kkey, suffix, dark) in enumerate([
                (off_key, "offline", False),
                (rag_key, "+ RAG",   True),
            ]):
                entry = clim.get(kkey)
                if entry is None or "levels" not in entry:
                    continue
                ys = [entry["levels"][l].get(key, np.nan) for l in levels]
                offset = base_x + vi * (bar_w + pair_gap) + bar_w / 2
                colour = FAMILY_COLOURS[fi]
                # Offline = lighter (alpha 0.55), RAG = darker (alpha 1.0)
                alpha = 1.0 if dark else 0.55
                edge = "white"
                label = f"{fam_name} {suffix}"
                show_label = label not in legend_seen
                legend_seen.add(label)
                ax.bar(x + offset, ys, width=bar_w, color=colour, alpha=alpha,
                       edgecolor=edge, linewidth=0.6,
                       label=label if show_label else None)
        # Paper baseline: thin black tick across each bar-group at its level
        # value. Reads as a target line over the bars without taking a slot
        # in the bar lineup.
        if paper_entry is not None and "levels" in paper_entry:
            half = total_span / 2 + 0.02
            tick_label = "Paper CLIM (recomputed)"
            show_paper = tick_label not in legend_seen
            for xi, lvl in zip(x, levels):
                y = paper_entry["levels"][lvl].get(key, np.nan)
                ax.hlines(y, xi - half, xi + half,
                          colors="black", linewidth=1.4, linestyles="-",
                          label=tick_label if (xi == 0 and show_paper) else None,
                          zorder=5)
            legend_seen.add(tick_label)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(level_labels)
        ax.set_xlabel("Climinator hierarchy level")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    fig.suptitle("Climinator metrics across the Leippold 2025 credibility hierarchy",
                 y=1.00)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    handles, labels_ = axes[0].get_legend_handles_labels()
    # Two-row legend: offline variants top row, RAG variants + Paper CLIM bottom.
    fig.legend(handles, labels_, loc="lower center",
               ncol=(len(handles) + 1) // 2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    out = RESULTS_DIR / "climinator_levels_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_online_vs_offline() -> Path:
    """Per (level, model) one stacked bar: solid lower segment = the smaller of
    online/offline (always-achievable accuracy); lighter top segment = the
    delta between online and offline. Crosshatch on the top segment marks the
    direction (// when web search adds; \\\\ when web search hurts)."""
    summary = json.loads((RESULTS_DIR / "metrics_summary.json").read_text())
    # Merge climinator (original) + climinator_v4 (current frozen variant).
    # v4 takes precedence on name collision. Earlier prompt variants
    # (v2, v3, v5) are intentionally excluded from headline plots — see
    # methods discussion of why v4 was chosen.
    clim = {**summary.get("climinator", {}), **summary.get("climinator_v4", {})}
    if not clim:
        raise SystemExit("no climinator results in metrics_summary.json")

    pairs = [
        ("Claude Opus 4.7", "Claude Opus 4.7 online", "Claude Opus 4.7 offline"),
        ("GPT-5.5",         "GPT-5.5 online",         "GPT-5.5 offline"),
    ]
    # Only keep pairs where both halves are present.
    pairs = [(short, on, off) for short, on, off in pairs
             if on in clim and off in clim and "levels" in clim[on] and "levels" in clim[off]]
    if not pairs:
        raise SystemExit("no online/offline pairs available — run the offline inferences first")

    levels = ["1", "2", "3", "4"]
    level_labels = ["L1", "L2", "L3", "L4"]
    metrics = [("accuracy",         "Accuracy"),
               ("macro_f1_present", "Macro-F1 (classes present)")]
    colours = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    n = len(pairs)
    x = np.arange(len(levels))
    width = 0.8 / max(n, 1)

    for ax, (key, title) in zip(axes, metrics):
        for i, (short, on_name, off_name) in enumerate(pairs):
            on_vals  = [clim[on_name]["levels"][l].get(key, np.nan)  for l in levels]
            off_vals = [clim[off_name]["levels"][l].get(key, np.nan) for l in levels]
            base = np.minimum(on_vals, off_vals)
            delta = np.abs(np.subtract(on_vals, off_vals))
            # Per-bar hatch: '//' when online > offline (search helps),
            # '\\\\' when offline > online (search hurts). matplotlib needs a
            # per-bar hatch via a loop, not a vector arg.
            offset = (i - (n - 1) / 2) * width
            colour = colours[i % len(colours)]
            ax.bar(x + offset, base, width=width, color=colour,
                   edgecolor="white", linewidth=0.6, label=short)
            for j, (b, d) in enumerate(zip(base, delta)):
                if d <= 1e-6:
                    continue
                hatch = "//" if on_vals[j] > off_vals[j] else "\\\\"
                ax.bar(x[j] + offset, d, bottom=b, width=width,
                       color=colour, alpha=0.35,
                       edgecolor="black", linewidth=0.4, hatch=hatch)
        # Majority baseline only meaningful for accuracy.
        if key == "accuracy":
            any_entry = next(iter(clim.values()))
            baseline = [any_entry["levels"][l]["baseline_acc"] for l in levels]
            ax.plot(x, baseline, "--", color="grey", linewidth=1.2,
                    marker="o", markersize=4, label="Majority baseline")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(level_labels)
        ax.set_xlabel("Climinator hierarchy level")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")

    # Extra legend entries for the hatch convention.
    from matplotlib.patches import Patch
    extra = [
        Patch(facecolor="lightgrey", edgecolor="black", hatch="//",
              label="online > offline (web search helps)"),
        Patch(facecolor="lightgrey", edgecolor="black", hatch="\\\\",
              label="offline > online (web search hurts)"),
    ]
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.suptitle("Online (web search) vs offline (built-in knowledge)", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.legend(handles + extra, labels_ + [p.get_label() for p in extra],
               loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.02))
    out = RESULTS_DIR / "climinator_online_vs_offline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_veracity() -> Path:
    """veracityV2 (4-class) bar chart. Two panels (Accuracy, Macro-F1).
    Paired offline/RAG bars per family in the shared family order, delta
    annotations under each pair. No baseline / no derived reference —
    Leippold 2025 does not publish a veracity benchmark, so the chart
    shows only our model outputs."""
    summary = json.loads((RESULTS_DIR / "metrics_summary.json").read_text())
    ver = summary.get("veracityV2", {})
    if not ver:
        raise SystemExit("no veracityV2 results in metrics_summary.json")

    metrics = [("accuracy",         "Accuracy"),
               ("macro_f1_present", "Macro-F1 (classes present)")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    n = len(FAMILIES)
    x = np.arange(n)
    bar_w = 0.32
    legend_seen: set[str] = set()

    for ax, (key, title) in zip(axes, metrics):
        for fi, (fam, off_key, rag_key) in enumerate(FAMILIES):
            off = ver.get(off_key); rag = ver.get(rag_key)
            off_v = off[key] if off else np.nan
            rag_v = rag[key] if rag else np.nan
            colour = FAMILY_COLOURS[fi]
            off_lbl = f"{fam} offline"
            rag_lbl = f"{fam} + RAG"
            ax.bar(x[fi] - bar_w/2, off_v, width=bar_w, color=colour, alpha=0.55,
                   edgecolor="white", linewidth=0.6,
                   label=off_lbl if off_lbl not in legend_seen else None)
            legend_seen.add(off_lbl)
            ax.bar(x[fi] + bar_w/2, rag_v, width=bar_w, color=colour, alpha=1.0,
                   edgecolor="white", linewidth=0.6,
                   label=rag_lbl if rag_lbl not in legend_seen else None)
            legend_seen.add(rag_lbl)
            ax.text(x[fi] - bar_w/2, off_v + 0.012, f"{off_v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=colour)
            ax.text(x[fi] + bar_w/2, rag_v + 0.012, f"{rag_v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=colour)
            delta = rag_v - off_v
            if abs(delta) >= 0.005:
                sign = "+" if delta > 0 else "−"
                dcol = "darkgreen" if delta > 0 else "darkred"
                ax.text(x[fi], -0.07, f"{sign}{abs(delta):.2f}",
                        ha="center", fontsize=8, color=dcol, fontweight="bold")

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f[0] for f in FAMILIES], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.10, 1.0)

    axes[0].set_ylabel("Score")
    fig.suptitle("veracity (4-class) — accuracy + macro-F1 by model and RAG",
                 y=1.00, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center",
               ncol=(len(handles) + 1) // 2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    out = RESULTS_DIR / "veracity_v2_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_combined() -> Path:
    """Single figure with two rows:
      Row 1 — climinator (4-level hierarchy) Acc + Macro-F1, Paper CLIM tick.
      Row 2 — veracityV2 (flat 4-class) Acc + Macro-F1, no reference.
    Paired offline/RAG bars per family in the shared family order. One
    shared legend at the bottom."""
    summary = json.loads((RESULTS_DIR / "metrics_summary.json").read_text())
    clim = {**summary.get("climinator", {}), **summary.get("climinator_v4", {})}
    ver  = summary.get("veracityV2", {})
    if not clim:
        raise SystemExit("no climinator results in metrics_summary.json")
    if not ver:
        raise SystemExit("no veracityV2 results in metrics_summary.json")

    PAPER_KEY = "Paper CLIM (recomputed)"
    paper_entry = clim.get(PAPER_KEY)

    levels = ["1", "2", "3", "4"]
    level_labels = ["L1", "L2", "L3", "L4"]
    metrics = [("accuracy", "Accuracy"),
               ("macro_f1_present", "Macro-F1 (classes present)")]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    n_fam = len(FAMILIES)
    legend_seen: set[str] = set()

    # ---- Row 1: climinator (4 levels × N families × 2 bars per level) ------
    # Bar geometry sized so 12 bars per level (6 families × 2 variants) fit
    # within the unit x-step with margin — at n=6 the total_span comes out
    # to ~0.78 which leaves 0.11 padding either side of each level group.
    x_clim = np.arange(len(levels))
    bar_w = 0.055
    pair_gap = 0.003
    family_gap = 0.02
    pair_span = 2 * bar_w + pair_gap
    total_span = n_fam * pair_span + (n_fam - 1) * family_gap

    for ax, (key, title) in zip(axes[0], metrics):
        for fi, (fam, off_key, rag_key) in enumerate(FAMILIES):
            base_x = -total_span / 2 + fi * (pair_span + family_gap)
            for vi, (kkey, suffix, dark) in enumerate([(off_key, "offline", False),
                                                       (rag_key, "+ RAG",   True)]):
                entry = clim.get(kkey)
                if entry is None or "levels" not in entry:
                    continue
                ys = [entry["levels"][l].get(key, np.nan) for l in levels]
                offset = base_x + vi * (bar_w + pair_gap) + bar_w / 2
                colour = FAMILY_COLOURS[fi]
                alpha = 1.0 if dark else 0.55
                label = f"{fam} {suffix}"
                show_label = label not in legend_seen
                legend_seen.add(label)
                ax.bar(x_clim + offset, ys, width=bar_w, color=colour, alpha=alpha,
                       edgecolor="white", linewidth=0.6,
                       label=label if show_label else None)
        # Paper CLIM tick across each level. NOT added to the legend —
        # annotated separately in the chart caption / axis title since the
        # bar legend reads cleanest with two rows (offline / + RAG) and one
        # column per family.
        if paper_entry is not None and "levels" in paper_entry:
            half = total_span / 2 + 0.02
            for xi, lvl in zip(x_clim, levels):
                y = paper_entry["levels"][lvl].get(key, np.nan)
                ax.hlines(y, xi - half, xi + half,
                          colors="black", linewidth=1.4, linestyles="-",
                          zorder=5)
        ax.set_title(f"climinator · {title}", fontsize=11)
        ax.set_xticks(x_clim)
        ax.set_xticklabels(level_labels)
        ax.set_xlabel("Climinator hierarchy level")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Score")

    # ---- Row 2: veracityV2 (1 level × N families × 2 bars) -----------------
    x_ver = np.arange(n_fam)
    ver_bar_w = 0.34
    for ax, (key, title) in zip(axes[1], metrics):
        for fi, (fam, off_key, rag_key) in enumerate(FAMILIES):
            off = ver.get(off_key); rag = ver.get(rag_key)
            off_v = off[key] if off else np.nan
            rag_v = rag[key] if rag else np.nan
            colour = FAMILY_COLOURS[fi]
            ax.bar(x_ver[fi] - ver_bar_w/2 - 0.02, off_v, width=ver_bar_w,
                   color=colour, alpha=0.55, edgecolor="white", linewidth=0.6)
            ax.bar(x_ver[fi] + ver_bar_w/2 + 0.02, rag_v, width=ver_bar_w,
                   color=colour, alpha=1.0, edgecolor="white", linewidth=0.6)
            ax.text(x_ver[fi] - ver_bar_w/2 - 0.02, off_v + 0.015, f"{off_v:.2f}",
                    ha="center", va="bottom", fontsize=9, color=colour)
            ax.text(x_ver[fi] + ver_bar_w/2 + 0.02, rag_v + 0.015, f"{rag_v:.2f}",
                    ha="center", va="bottom", fontsize=9, color=colour)
            delta = rag_v - off_v
            if abs(delta) >= 0.005:
                sign = "+" if delta > 0 else "−"
                dcol = "darkgreen" if delta > 0 else "darkred"
                ax.text(x_ver[fi], -0.08, f"{sign}{abs(delta):.2f}",
                        ha="center", fontsize=9, color=dcol, fontweight="bold")
        ax.set_title(f"veracity · {title}", fontsize=12)
        ax.set_xticks(x_ver)
        ax.set_xticklabels([f[0] for f in FAMILIES], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.12, 1.05)
    axes[1, 0].set_ylabel("Score")

    fig.suptitle("Debunk benchmark — climinator (top) and veracity (bottom). "
                 "Black ticks on the climinator panels mark Paper CLIM (recomputed).",
                 y=0.995, fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, hspace=0.35)
    # Build the legend column-by-column: each column is one model family with
    # offline on top and + RAG below. matplotlib's legend fills column-major
    # when ncol < len(handles), so we INTERLEAVE the handles as
    # [off1, rag1, off2, rag2, …]. With ncol = n_families that gives each
    # column = one family, 2 rows = offline / + RAG.
    from matplotlib.patches import Patch
    handles: list[Patch] = []
    for (fam, _, _), c in zip(FAMILIES, FAMILY_COLOURS):
        handles.append(Patch(facecolor=c, alpha=0.55, edgecolor="white",
                              label=f"{fam} offline"))
        handles.append(Patch(facecolor=c, alpha=1.0, edgecolor="white",
                              label=f"{fam} + RAG"))
    fig.legend(handles=handles, loc="lower center",
               ncol=len(FAMILIES), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    out = RESULTS_DIR / "debunk_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = plot_combined()
    print(f"  -> {p}")
    try:
        p2 = plot_online_vs_offline()
        print(f"  -> {p2}")
    except SystemExit as e:
        print(f"  [skip] online-vs-offline plot: {e}")


if __name__ == "__main__":
    main()
