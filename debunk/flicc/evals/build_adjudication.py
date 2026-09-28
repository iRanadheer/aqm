# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build the blind adjudication set: items where a model disagrees with gold.

Motivation. The FLICC gold standard assigns exactly one fallacy per text, but a
text can exhibit more than one — a passage may insult an opponent AND allege a
conspiracy, and the annotator had to pick. Where that happens, a model naming
the other present fallacy is scored wrong for a label that is arguably correct.
This script produces the material for a human expert to adjudicate that, so the
question is settled by judgement rather than assumed either way.

Design constraints, all of which matter for the result to mean anything:

  * LABEL ONLY on both sides. The model emits a full <think> deconstruction; the
    gold standard is a bare label. Showing the trace would tell the adjudicator
    which option is the model's and would likely make it the more persuasive
    one. Both sides therefore carry the label and nothing else.
  * SLOT RANDOMISED PER ROW. Gold is not fixed to `a`. Assignment is redrawn for
    every item, so there is no positional cue to learn part-way through.
  * ROWS SHUFFLED. Otherwise items arrive grouped by gold class, and a run of
    one class tells the adjudicator what to expect. Shuffling also means any
    prefix of the file is a usable sample if the expert stops early.
  * SEEDED. The same seed reproduces the same file byte for byte, so the set
    that was adjudicated can always be rebuilt.

The key file is written alongside and MUST NOT be shown to the adjudicator. It
records which slot held gold for each item.

A note on what the verdicts can support. These are the items one model got
wrong, not a random sample of the split, so they bound how much of THAT model's
error is annotation ambiguity — they cannot estimate the ambiguity rate of the
dataset as a whole. And a "both valid" verdict does not license an adjusted
score: the same ambiguity was present when the published baseline was scored
against the same gold, so repairing one side only would not be a fair
comparison. What it does support is a ceiling argument (single-label scoring
caps every model below 1.0) and an asymmetry argument (a classifier trained on
these annotators' tie-breaks inherits their conventions; a model reasoning from
a written codebook does not).

  uv run evals/build_adjudication.py
  uv run evals/build_adjudication.py --split val --model c3ds-flicc-qwen3-5-9b
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prompts import FALLACY_LABELS, extract_raw_label  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--split", default="test", choices=["val", "test"],
                help="Default test — the split the headline numbers and the "
                     "comparison to the published baseline are reported on.")
ap.add_argument("--model", default="c3ds-flicc-qwen3-5-27b",
                help="Result-file slug, without the prompt suffix.")
ap.add_argument("--prompt", default="deconstruct-revised")
ap.add_argument("--seed", type=int, default=20260812,
                help="Change this ONLY to build a genuinely new set. Reusing "
                     "the default reproduces the set that was adjudicated.")
ap.add_argument("--out-dir", default="data/adjudication")
args = ap.parse_args()

src = ROOT / "data" / "results" / args.split / f"{args.model}-{args.prompt}.jsonl"
if not src.exists():
    sys.exit(f"No such result file: {src}")

rows = [json.loads(line) for line in src.open()]

# Unparseable responses are excluded, not counted as disagreements. A parse
# failure is a formatting defect with no label to adjudicate — including it
# would ask the expert to rule on an empty option.
pairs = []
for r in rows:
    pred = extract_raw_label(str(r["response"]))
    if pred is not None and pred != r["true_fallacy"]:
        pairs.append((r, pred))

rng = random.Random(args.seed)
rng.shuffle(pairs)

items, key = [], []
for r, pred in pairs:
    gold = r["true_fallacy"]
    gold_is_a = rng.random() < 0.5
    a_label, b_label = (gold, pred) if gold_is_a else (pred, gold)
    items.append({
        "itemId": r["itemId"],
        "content": r["text"],
        "a": {"label": a_label},
        "b": {"label": b_label},
    })
    key.append({
        "itemId": r["itemId"],
        "a_source": "gold" if gold_is_a else args.model,
        "b_source": args.model if gold_is_a else "gold",
        "gold_label": gold,
        "model_label": pred,
        "gold_slot": "a" if gold_is_a else "b",
    })

out_dir = ROOT / args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)
stem = f"{args.split}_disagreements"
for path, data in [(out_dir / f"{stem}.jsonl", items),
                   (out_dir / f"{stem}_key.jsonl", key)]:
    with path.open("w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

# The twelve labels, so the adjudicator is not confined to the two options —
# "neither, it is actually X" is a different finding from "both are valid", and
# only the first implicates the model rather than the annotation scheme.
with (out_dir / "labels.jsonl").open("w") as f:
    for i, label in enumerate(FALLACY_LABELS, 1):
        f.write(json.dumps({"index": i, "label": label}, ensure_ascii=False) + "\n")

n_a = sum(1 for k in key if k["gold_slot"] == "a")
by_class = collections.Counter(k["gold_label"] for k in key)
missing = [l for l in FALLACY_LABELS if l not in by_class]

print(f"{len(items)} disagreements  ({args.model}, {args.split}, seed {args.seed})")
print(f"  {len(rows)} items, {len(rows) - len(pairs)} agreed or unparseable")
print(f"  gold in slot a: {n_a} ({100 * n_a / len(items):.0f}%), slot b: {len(items) - n_a}")
print("  per gold class: " + ", ".join(f"{k} {v}" for k, v in sorted(by_class.items())))
if missing:
    print(f"  absent (model got every item right): {', '.join(missing)}")
print()
print(f"  wrote {out_dir / f'{stem}.jsonl'}")
print(f"  wrote {out_dir / f'{stem}_key.jsonl'}   <- withhold from the adjudicator")
print(f"  wrote {out_dir / 'labels.jsonl'}")
