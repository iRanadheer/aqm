"""
Intercoder Reliability (ICR) Analysis for CARDS Taxonomy Coding.

Two coders independently applied taxonomy codes to a 50-item subset of the
chapter sample (`augCards_chapter_*` IDs from data/cards_icr.json).

Computes Krippendorff's Alpha, percent agreement, and Jaccard similarity at
each of the three hierarchy levels (top-level, sub-category, claim) and
writes:
    docs/icr_report.md           - human-readable summary
    data/icr_disagreements.csv   - per-item disagreements at level 3

Usage:
    python icr_analysis.py
"""

import csv
import json
import os
from collections import Counter

import krippendorff
import numpy as np


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_DIR, "data")
DOCS_DIR = os.path.join(REPO_DIR, "docs")
DATA_PATH = os.path.join(DATA_DIR, "cards_icr.json")
CODERS = ["travcoan", "mirjamnanko"]


def load_data(path):
    """Load JSON and extract code prefixes for each coder per item.

    Also returns a mapping of code -> human-readable label (first occurrence).
    """
    with open(path) as f:
        items = json.load(f)

    parsed = []
    code_labels = {}
    for item in items:
        entry = {"itemId": item["itemId"], "content": item["content"]}
        for coder in CODERS:
            codes = set()
            for label in item[coder]:
                code, _, text = label.partition(" : ")
                code = code.strip()
                codes.add(code)
                if code not in code_labels and text:
                    code_labels[code] = text.strip()
            entry[coder] = codes
        parsed.append(entry)
    return parsed, code_labels


def normalize_codes(codes):
    """Remove redundant parent codes from a set of codes.

    A code A is redundant if a more specific code B exists such that:
    - They share the same top-level category
    - B has non-zero values at positions where A has zero
    - A's non-zero values match B's at corresponding positions

    E.g., 2_1_0 is redundant when 2_1_4 exists; 2_0_0 is redundant when 2_1_0 exists.
    """
    codes = set(codes)
    to_remove = set()

    for a in codes:
        a_parts = a.split("_")
        for b in codes:
            if a == b:
                continue
            b_parts = b.split("_")

            # Must share same top-level category
            if a_parts[0] != b_parts[0]:
                continue

            # Check if A is a parent of B:
            # A's non-zero parts must match B's, and B must be strictly more specific
            max_len = max(len(a_parts), len(b_parts))
            a_ext = a_parts + ["0"] * (max_len - len(a_parts))
            b_ext = b_parts + ["0"] * (max_len - len(b_parts))

            is_parent = True
            strictly_more_specific = False
            for i in range(max_len):
                if a_ext[i] == "0" and b_ext[i] != "0":
                    strictly_more_specific = True
                elif a_ext[i] != "0" and a_ext[i] != b_ext[i]:
                    is_parent = False
                    break

            if is_parent and strictly_more_specific:
                to_remove.add(a)
                break

    return codes - to_remove


def truncate_codes(codes, level):
    """Truncate codes to the given hierarchy level (number of components)."""
    truncated = set()
    for code in codes:
        parts = code.split("_")
        truncated.add("_".join(parts[:level]))
    return truncated


def compute_krippendorff_alpha(data, level):
    """Compute Krippendorff's Alpha using binary coding at the given level.

    Each unique code at this level becomes a binary variable.
    The reliability matrix has shape (2 coders, n_items * n_codes).
    """
    # Get all code sets at this level
    all_codes_per_item = []
    for item in data:
        coder_codes = {}
        for coder in CODERS:
            coder_codes[coder] = truncate_codes(item[coder], level)
        all_codes_per_item.append(coder_codes)

    # Collect all unique codes at this level
    all_codes = set()
    for item_codes in all_codes_per_item:
        for coder in CODERS:
            all_codes.update(item_codes[coder])
    all_codes = sorted(all_codes)

    # Build reliability matrix: rows = coders, columns = item-code pairs
    # Each item x code combination is a unit; each coder rates it 0 or 1
    n_units = len(data) * len(all_codes)
    reliability_matrix = np.zeros((2, n_units), dtype=float)

    for i, item_codes in enumerate(all_codes_per_item):
        for j, code in enumerate(all_codes):
            col = i * len(all_codes) + j
            for c_idx, coder in enumerate(CODERS):
                reliability_matrix[c_idx, col] = 1 if code in item_codes[coder] else 0

    alpha = krippendorff.alpha(
        reliability_data=reliability_matrix, level_of_measurement="nominal"
    )
    return alpha, all_codes


def compute_agreement_stats(data, level):
    """Compute percent agreement (exact match) and mean Jaccard similarity."""
    exact_matches = 0
    jaccard_sum = 0.0

    for item in data:
        sets = {coder: truncate_codes(item[coder], level) for coder in CODERS}
        s1, s2 = sets[CODERS[0]], sets[CODERS[1]]

        if s1 == s2:
            exact_matches += 1

        union = s1 | s2
        intersection = s1 & s2
        jaccard = len(intersection) / len(union) if union else 1.0
        jaccard_sum += jaccard

    n = len(data)
    return exact_matches / n, jaccard_sum / n


def compute_descriptive_stats(data):
    """Compute mean codes per item per coder and total unique codes."""
    stats = {}
    for coder in CODERS:
        counts = [len(item[coder]) for item in data]
        all_codes = set()
        for item in data:
            all_codes.update(item[coder])
        stats[coder] = {
            "mean_codes": sum(counts) / len(counts),
            "total_unique": len(all_codes),
        }
    return stats


def compute_claim_frequencies(data):
    """Frequency of each Level-3 code across items.

    A code is counted once per item if either coder applied it (union),
    and we also report per-coder counts and the number of items where
    both coders agreed on that code (intersection).
    """
    union_counts = Counter()
    intersection_counts = Counter()
    per_coder = {coder: Counter() for coder in CODERS}

    for item in data:
        sets = {coder: truncate_codes(item[coder], 3) for coder in CODERS}
        for coder in CODERS:
            for code in sets[coder]:
                per_coder[coder][code] += 1
        for code in sets[CODERS[0]] | sets[CODERS[1]]:
            union_counts[code] += 1
        for code in sets[CODERS[0]] & sets[CODERS[1]]:
            intersection_counts[code] += 1

    return union_counts, intersection_counts, per_coder


METHOD_TEXT = """\
## Method

Two coders ({coder1}, {coder2}) independently applied codes from the CARDS
contrarian-claims taxonomy to {n_items} items. Codes follow a three-level
hierarchical scheme (e.g. `2_1_4`), where each item may receive multiple codes.

**Code normalization.** Before scoring, we removed redundant parent codes from
each coder's set on a per-item basis. A code `A` is redundant when a strictly
more specific code `B` exists in the same set such that `A`'s non-zero
components match `B`'s at the corresponding positions (e.g. `2_1_0` is dropped
when `2_1_4` is also present, and `2_0_0` is dropped when `2_1_0` is present).
This avoids double-counting the same claim at parent and child levels.

**Hierarchy levels.** We evaluate agreement at three levels of granularity by
truncating each code to its first 1, 2, or 3 components. Level 1 reflects the
top-level category, Level 2 the sub-category, and Level 3 the most specific
claim.

**Krippendorff's Alpha.** Because items are multi-label, we transform each
(item, code) pair at a given level into a binary unit (1 if the coder applied
the code, 0 otherwise). Alpha is then computed on the resulting `2 x (n_items
* n_codes)` reliability matrix using the `nominal` level of measurement, via
the `krippendorff` Python package.

**Supporting statistics.** For each level we also report (i) percent exact
agreement on the full code set per item, (ii) mean Jaccard similarity of the
two coders' code sets per item, and (iii) the number of unique codes that
appear at that level.
"""


def export_markdown(
    path, data, level_results, descriptive_before, descriptive_after,
    disagree_count, code_labels,
):
    union_counts, intersection_counts, per_coder = compute_claim_frequencies(data)

    lines = []
    lines.append("# Intercoder Reliability (ICR) Analysis")
    lines.append("")
    lines.append(f"- **Items coded:** {len(data)}")
    lines.append(f"- **Coders:** `{CODERS[0]}`, `{CODERS[1]}`")
    lines.append(f"- **Items with disagreement at Level 3:** "
                 f"{disagree_count}/{len(data)}")
    lines.append("")
    lines.append(METHOD_TEXT.format(
        coder1=CODERS[0], coder2=CODERS[1], n_items=len(data),
    ))
    lines.append("## Reliability by Hierarchy Level")
    lines.append("")
    lines.append("| Level | Krippendorff's Alpha | % Exact Agreement | "
                 "Mean Jaccard | # Unique Codes |")
    lines.append("|-------|----------------------|-------------------|"
                 "--------------|----------------|")
    for level, alpha, pct_agree, mean_jaccard, n_codes in level_results:
        lines.append(
            f"| {level} | {alpha:.4f} | {pct_agree:.1%} | "
            f"{mean_jaccard:.4f} | {n_codes} |"
        )
    lines.append("")

    lines.append("## Descriptive Statistics")
    lines.append("")
    lines.append("| Stage | Coder | Mean codes / item | Total unique codes |")
    lines.append("|-------|-------|-------------------|--------------------|")
    for stage_name, stats in [
        ("Before normalization", descriptive_before),
        ("After normalization", descriptive_after),
    ]:
        for coder in CODERS:
            s = stats[coder]
            lines.append(
                f"| {stage_name} | `{coder}` | {s['mean_codes']:.2f} | "
                f"{s['total_unique']} |"
            )
    lines.append("")

    lines.append("## Claim Frequency Distribution (Level 3)")
    lines.append("")
    lines.append(
        "Counts are the number of items (out of "
        f"{len(data)}) on which each Level-3 claim was applied. "
        "*Either* counts items where at least one coder used the claim; "
        "*Both* counts items where the coders agreed on it; per-coder "
        "columns count items where that coder applied it."
    )
    lines.append("")
    lines.append(f"| Code | Claim | {CODERS[0]} | {CODERS[1]} | "
                 "Either | Both |")
    lines.append("|------|-------|"
                 + "-" * (len(CODERS[0]) + 2) + "|"
                 + "-" * (len(CODERS[1]) + 2) + "|"
                 "--------|------|")
    for code, _ in sorted(union_counts.items(),
                           key=lambda kv: (-kv[1], kv[0])):
        label = code_labels.get(code, "")
        label_md = label.replace("|", "\\|")
        lines.append(
            f"| `{code}` | {label_md} | {per_coder[CODERS[0]][code]} | "
            f"{per_coder[CODERS[1]][code]} | {union_counts[code]} | "
            f"{intersection_counts[code]} |"
        )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    # Load and parse
    data, code_labels = load_data(DATA_PATH)
    print(f"Loaded {len(data)} items\n")

    # Descriptive stats before normalization
    print("=== Descriptive Stats (before normalization) ===")
    descriptive_before = compute_descriptive_stats(data)
    for coder in CODERS:
        s = descriptive_before[coder]
        print(
            f"  {coder}: mean codes/item = {s['mean_codes']:.2f}, "
            f"total unique codes = {s['total_unique']}"
        )

    # Normalize codes
    for item in data:
        for coder in CODERS:
            item[coder] = normalize_codes(item[coder])

    print("\n=== Descriptive Stats (after normalization) ===")
    descriptive_after = compute_descriptive_stats(data)
    for coder in CODERS:
        s = descriptive_after[coder]
        print(
            f"  {coder}: mean codes/item = {s['mean_codes']:.2f}, "
            f"total unique codes = {s['total_unique']}"
        )

    # Compute ICR at each hierarchy level
    print("\n" + "=" * 65)
    print(f"{'Level':<10} {'Alpha':>10} {'% Agree':>10} {'Jaccard':>10} {'# Codes':>10}")
    print("-" * 65)

    level_results = []
    for level in [1, 2, 3]:
        alpha, codes = compute_krippendorff_alpha(data, level)
        pct_agree, mean_jaccard = compute_agreement_stats(data, level)
        level_results.append((level, alpha, pct_agree, mean_jaccard, len(codes)))
        print(
            f"Level {level:<5} {alpha:>10.4f} {pct_agree:>9.1%} {mean_jaccard:>10.4f} {len(codes):>10}"
        )

    print("=" * 65)

    # Export disagreements to CSV
    csv_path = os.path.join(DATA_DIR, "icr_disagreements.csv")
    disagree_count = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "itemId", "content",
            "travcoan_codes", "mirjamnanko_codes",
            "shared_codes", "travcoan_only", "mirjamnanko_only",
            "jaccard",
        ])
        for item in data:
            s1 = truncate_codes(item[CODERS[0]], 3)
            s2 = truncate_codes(item[CODERS[1]], 3)
            if s1 == s2:
                continue
            disagree_count += 1
            shared = sorted(s1 & s2)
            only_1 = sorted(s1 - s2)
            only_2 = sorted(s2 - s1)
            union = s1 | s2
            jaccard = len(s1 & s2) / len(union) if union else 1.0
            writer.writerow([
                item["itemId"],
                item["content"],
                "; ".join(sorted(s1)),
                "; ".join(sorted(s2)),
                "; ".join(shared),
                "; ".join(only_1),
                "; ".join(only_2),
                f"{jaccard:.3f}",
            ])

    print(f"\nDisagreements: {disagree_count}/{len(data)} items")
    print(f"Disagreement details saved to {csv_path}")

    md_path = os.path.join(DOCS_DIR, "icr_report.md")
    export_markdown(
        md_path, data, level_results, descriptive_before, descriptive_after,
        disagree_count, code_labels,
    )
    print(f"ICR report saved to {md_path}")


if __name__ == "__main__":
    main()
