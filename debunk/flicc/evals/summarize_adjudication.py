"""Summarize review clicks separately from the expert's final label sets."""

import argparse
from collections import Counter

from build_multilabel_gold import DATA, build, demo, read


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        demo()
        raise SystemExit

    export = read(DATA / "flicc-disagreement-review-export.jsonl")
    key = read(DATA / "test_disagreements_key.jsonl")
    rows = build(export, key, read(DATA / "test_disagreements.jsonl"),
                 {r["label"] for r in read(DATA / "labels.jsonl")})
    clicks = Counter()
    by_choice = Counter()
    for r in rows:
        choice = r["expert_choice"]
        if choice in {"a", "b"}:
            clicks["gold" if r["gold_was_right"] else "model"] += 1
        else:
            clicks[choice] += 1
        by_choice[(choice, r["gold_was_right"], r["model_was_right"])] += 1
    print(f"{len(rows)} items reviewed")
    print("Clicks: " + ", ".join(f"{c} {clicks[c]}" for c in
                               ("gold", "model", "both", "neither")))
    print("Final label sets: " + ", ".join(
        f"{n} labels {count}" for n, count in
        sorted(Counter(len(r["gold_labels"]) for r in rows).items())))
    for side, field in (("Gold", "gold_was_right"), ("Model", "model_was_right")):
        print(f"{side} label appears in final set: "
              f"{sum(r[field] for r in rows)}/{len(rows)}")
    neither = [r for r in rows if r["expert_choice"] == "neither"]
    print(f"Neither clicks with gold and/or model in final set: "
          f"{sum(r['gold_was_right'] or r['model_was_right'] for r in neither)}/{len(neither)}")
    extras = sum(len(set(r["gold_labels"]) - {r["gold_label"], r["model_label"]})
                 for r in rows)
    print(f"Additional expert-label occurrences: {extras}")
    print(f"Items with notes (not automatically labels): "
          f"{sum(bool(r['expert_notes'].strip()) for r in export)}")
