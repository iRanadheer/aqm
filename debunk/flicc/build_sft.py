# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas"]
# ///
"""Turn validated teacher traces into SFT chat data.

    uv run build_sft.py --input data/train/recot-anthropic-claude-opus-4-7.jsonl
    uv run build_sft.py --input ... --labels-only --out-dir data/sft-labels

Writes <out-dir>/train.jsonl and <out-dir>/train_eval.jsonl, each row a single
{"messages": [system, user, assistant]} object — the format Unsloth and TRL
consume directly.

THE POINT OF RECoT: the teacher was shown the gold label, the student is not.
The user turn written here is the INFERENCE turn (USER_TEMPLATE), never the
teacher turn that carried the label and trigger. A student trained on the
teacher's own input learns to expect the answer in its prompt and collapses at
inference. `_check_no_leakage` enforces this.

The system prompt is embedded per row, following wind/ and cards/ — verbose,
but it is what the chat template expects, and the student must train on exactly
the prompt it will meet at inference.

--labels-only strips the reasoning, leaving just the YAML answer. That is the
control for the paper's own thesis: is the manual deconstruction what makes the
dataset valuable, or do the labels alone suffice? Same rows, same split, same
seed, so the two runs differ only in whether the trace is present.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prompts import FALLACY_LABELS, USER_TEMPLATE, get_prompt  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--input", required=True, help="Validated traces from teacher.py")
ap.add_argument("--out-dir", default="data/sft")
ap.add_argument("--eval-frac", type=float, default=0.10,
                help="Held-out fraction for in-training eval, stratified by label.")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--labels-only", action="store_true",
                help="Assistant turn is the YAML answer alone, no reasoning. "
                     "The ablation that tests whether the traces earn their cost.")
ap.add_argument("--keep-contaminated", action="store_true",
                help="Keep the rows EXCLUDED_IDS drops. For reproducing the "
                     "unfiltered pool only.")
args = ap.parse_args()

# Rows removed from training, found by normalising away punctuation and case.
# Zanartu et al. deduplicated with exact matching plus BERT-embedding cosine
# similarity and manual review; these survived because the quote characters
# differ. Verified by character diff, not by the normalisation alone.
#
# Leaked into an eval split (>=98% character similarity, whole text):
#   train-0605  -> test-0079   differs by one capital and a trailing period
#   train-1390  -> val-0382    differs by curly quotes vs a period
#   train-1599  -> val-0113    differs by one apostrophe
# Same text, contradictory gold labels — the teacher wrote opposing traces:
#   train-0223  single cause   vs  train-0611  misrepresentation
#
# val-0108/val-0386 are the same conflict inside val. Left alone: correcting
# someone else's eval annotation is not ours to do, and it is reported as label
# noise instead.
EXCLUDED_IDS = {
    "train-0605", "train-1390", "train-1599",   # leaked into val/test
    "train-0223", "train-0611",                 # conflicting gold labels
}

in_path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
rows = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
if not rows:
    sys.exit(f"No traces in {in_path}")

variants = {r.get("prompt_variant") for r in rows}
if len(variants) != 1:
    sys.exit(f"Traces mix prompt variants {variants} — rebuild from one teacher run")
system_prompt = get_prompt(variants.pop())

_YAML_RE = re.compile(r"```(?:yaml|yml)?\s*\n?fallacy:.*?```", re.DOTALL | re.IGNORECASE)


def assistant_turn(r: dict) -> str:
    """The full trace, or just its YAML answer under --labels-only."""
    if not args.labels_only:
        return r["response"].strip()
    m = _YAML_RE.search(r["response"])
    if m:
        return m.group(0).strip()
    return f"```yaml\nfallacy: {r['true_fallacy']}\n```"


def _check_no_leakage(messages: list[dict], gold: str) -> None:
    """The student's prompt must not contain the answer, in any form."""
    prompt_text = "\n".join(m["content"] for m in messages if m["role"] != "assistant")
    if gold in prompt_text.split("### CODEBOOK:")[-1].split("### INSTRUCTIONS:")[0]:
        return  # label names legitimately appear in the codebook
    for marker in ("True Label", "true label above", "RECoT", "arrives at exactly"):
        if marker in prompt_text:
            raise SystemExit(f"LEAK: teacher-only text {marker!r} reached the student prompt")


# teacher.py appends in completion order, which is nondeterministic. Sort by
# itemId first so the split is a pure function of (itemId set, seed) and does
# not depend on how the trace file happened to get written — otherwise
# regenerating traces silently reshuffles train/eval, and the RECoT and
# labels-only arms could end up on different splits.
if any("itemId" not in r for r in rows):
    sys.exit("Traces missing itemId — cannot produce a reproducible split")
rows.sort(key=lambda r: r["itemId"])

if not args.keep_contaminated:
    present = {r["itemId"] for r in rows} & EXCLUDED_IDS
    rows = [r for r in rows if r["itemId"] not in EXCLUDED_IDS]
    print(f"excluded {len(present)} contaminated rows: {sorted(present)}")
    if missing_ids := EXCLUDED_IDS - present:
        print(f"  note: {sorted(missing_ids)} not present in this trace file")

built = []
for r in rows:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_TEMPLATE.format(text=r["text"])},
        {"role": "assistant", "content": assistant_turn(r)},
    ]
    _check_no_leakage(messages, r["true_fallacy"])
    built.append((r["true_fallacy"], {"messages": messages}, r["itemId"]))

# Stratified split, so rare labels (fake experts n=44, false choice n=48 in the
# full train pool) are not swallowed whole by the eval slice.
by_label: dict[str, list] = defaultdict(list)
for lab, obj, item_id in built:
    by_label[lab].append((obj, item_id))
rng = random.Random(args.seed)
train, evalset = [], []
for lab in sorted(by_label):
    items = by_label[lab][:]
    rng.shuffle(items)
    k = int(round(len(items) * args.eval_frac))
    evalset.extend(items[:k])
    train.extend(items[k:])
rng.shuffle(train)
rng.shuffle(evalset)

out_dir.mkdir(parents=True, exist_ok=True)
for name, data in (("train", train), ("train_eval", evalset)):
    p = out_dir / f"{name}.jsonl"
    with open(p, "w") as f:
        for obj, _ in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"wrote {p}  ({len(data)} rows, {p.stat().st_size / 1e6:.1f} MB)")

# The uploaded rows are {"messages": ...} only, so the split cannot be audited
# from them. The manifest records which itemId went where, making the split
# independently checkable against data/train/train.jsonl without the traces.
manifest = {
    "seed": args.seed,
    "eval_frac": args.eval_frac,
    "source": in_path.name,
    "rule": "sort by itemId, group by label, shuffle each group with seed, "
            "take round(n*eval_frac) per label into eval",
    "counts": {"train": len(train), "train_eval": len(evalset)},
    "train": sorted(i for _, i in train),
    "train_eval": sorted(i for _, i in evalset),
}
mp = out_dir / "split_manifest.json"
mp.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"wrote {mp}  ({len(evalset)} eval ids recorded)")

missing = set(FALLACY_LABELS) - set(by_label)
print(f"\nmode: {'labels-only' if args.labels_only else 'full RECoT traces'}")
print(f"system prompt: {len(system_prompt)} chars, embedded per row")
print(f"label spread : {dict(Counter(lab for lab, _, _ in built))}")
if missing:
    print(f"WARNING: no traces for {sorted(missing)}")
