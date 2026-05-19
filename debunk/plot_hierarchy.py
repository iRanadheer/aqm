"""Grouped-bar chart of climinator metrics across the Leippold 2024 4-level
credibility hierarchy (L1=12 → L2=5 → L3=3 → L4=2).

Writes `climinator_levels_bars.png` under `data/results/test/`: two panels
(Accuracy, Macro-F1). At each level, one bar per model.

Usage:
  python3 plot_climinator_levels.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "data" / "results" / "test"

# Numbers reported by Leippold et al. 2025 (npj Climate Action 4:17), Table 1,
# computed on n=165 after excluding "not enough information" verdicts. MCC is
# not reported in the paper, so those bars are omitted on that panel.
PAPER_RESULTS: dict[str, dict[str, dict[int, float]]] = {
    "CLIM (paper, GPT-4o + RAG)": {
        "accuracy":         {1: 0.345, 2: 0.727, 3: 0.958, 4: 0.964},
        "macro_f1_present": {1: 0.176, 2: 0.430, 3: 0.612, 4: 0.896},
    },
}


def plot_bars() -> Path:
    summary = json.loads((RESULTS_DIR / "metrics_summary.json").read_text())
    clim = summary.get("climinator", {})
    if not clim:
        raise SystemExit("no climinator results in metrics_summary.json")

    levels = ["1", "2", "3", "4"]
    level_labels = ["L1", "L2", "L3", "L4"]
    metrics = [("accuracy",         "Accuracy"),
               ("macro_f1_present", "Macro-F1 (classes present)")]
    # Drop Sonar Pro from the lineup — keeps the chart legible.
    HIDDEN = {"Sonar Pro"}
    models = [(m, e) for m, e in clim.items() if "levels" in e and m not in HIDDEN]
    paper_labels = list(PAPER_RESULTS.keys())
    n_bars = len(models) + len(paper_labels)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(levels))
    width = 0.8 / max(n_bars, 1)
    colours = plt.get_cmap("tab10").colors

    for ax, (key, title) in zip(axes, metrics):
        for i, (label, entry) in enumerate(models):
            ys = [entry["levels"][l].get(key, np.nan) for l in levels]
            offset = (i - (n_bars - 1) / 2) * width
            ax.bar(x + offset, ys, width=width, label=label,
                   color=colours[i % len(colours)], edgecolor="white", linewidth=0.6)
        # Paper baselines — drawn last so they sit on the right of each group.
        # Hatched to mark them as externally reported numbers (not our runs).
        for j, plabel in enumerate(paper_labels):
            i = len(models) + j
            offset = (i - (n_bars - 1) / 2) * width
            ys = [PAPER_RESULTS[plabel].get(key, {}).get(int(l), np.nan) for l in levels]
            ax.bar(x + offset, ys, width=width, label=plabel,
                   color="lightgrey", edgecolor="black", linewidth=0.6,
                   hatch="//")
        # Majority baseline only meaningful for accuracy.
        if key == "accuracy":
            any_entry = models[0][1]
            base = [any_entry["levels"][l]["baseline_acc"] for l in levels]
            ax.plot(x, base, "--", color="grey", linewidth=1.2,
                    marker="o", markersize=4, label="Majority baseline")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(level_labels)
        ax.set_xlabel("Climinator hierarchy level")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    fig.suptitle("Climinator metrics across the Leippold 2024 credibility hierarchy",
                 y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.02))
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
    clim = summary.get("climinator", {})
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


def main() -> None:
    p = plot_bars()
    print(f"  -> {p}")
    try:
        p2 = plot_online_vs_offline()
        print(f"  -> {p2}")
    except SystemExit as e:
        print(f"  [skip] online-vs-offline plot: {e}")


if __name__ == "__main__":
    main()
