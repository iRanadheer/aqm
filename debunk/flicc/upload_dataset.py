# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub", "python-dotenv"]
# ///
"""Push the built SFT splits to the Hub so ft/train.py can load them.

    uv run upload_dataset.py                    # private (default)
    uv run upload_dataset.py --public

Uploads both arms of the ablation into one dataset repo:

    recot/train.jsonl        recot/train_eval.jsonl        full teacher traces
    labels/train.jsonl       labels/train_eval.jsonl       YAML answer only

`ft/train.py --recot/--no-recot` selects between them. They come from the same
teacher pass and the same stratified split, so a difference between two trained
models is attributable to the reasoning and nothing else.

Defaults to PRIVATE, unlike the cards/wind dataset repos, because these traces
are unpublished. Pass --public deliberately.

The cards and wind chapters never had an upload script — their Hub datasets were
pushed by hand — so this exists to make the flicc pipeline reproducible end to
end rather than to follow an existing pattern.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
# Keys live in the monorepo root .env; nearer paths win, matching infer.py.
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")
load_dotenv(ROOT.parent.parent / ".env")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--repo", default="iRanadheer/flicc-fallacy-sft")
ap.add_argument("--recot-dir", default="data/sft")
ap.add_argument("--labels-dir", default="data/sft-labels")
ap.add_argument("--public", action="store_true", help="Create/keep the repo public.")
ap.add_argument("--dry-run", action="store_true", help="Report what would upload, then stop.")
args = ap.parse_args()


def resolve(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


pairs = [("recot", resolve(args.recot_dir)), ("labels", resolve(args.labels_dir))]
files: list[tuple[str, Path]] = []
for prefix, d in pairs:
    if not d.exists():
        sys.exit(f"Missing {d} — run build_sft.py first")
    for split in ("train", "train_eval"):
        f = d / f"{split}.jsonl"
        if not f.exists():
            sys.exit(f"Missing {f}")
        files.append((f"{prefix}/{split}.jsonl", f))
    m = d / "split_manifest.json"
    if m.exists():
        files.append((f"{prefix}/split_manifest.json", m))

print(f"repo: {args.repo}  ({'public' if args.public else 'private'})")
counts = {}
for dest, src in files:
    n = 0 if dest.endswith(".json") else sum(1 for _ in src.open())
    counts[dest] = n
    print(f"  {dest:26s} {n:5d} rows  {src.stat().st_size / 1e6:6.1f} MB")

# The two arms must be row-aligned, else the ablation compares different data.
if counts["recot/train.jsonl"] != counts["labels/train.jsonl"] or \
        counts["recot/train_eval.jsonl"] != counts["labels/train_eval.jsonl"]:
    sys.exit("recot/ and labels/ row counts differ — rebuild both from one teacher run")

card = f"""---
license: mit
task_categories: [text-classification]
language: [en]
tags: [climate, misinformation, fallacy, flicc, recot]
---

# FLICC fallacy detection — SFT data

Reasoning traces for training a student model on the twelve FLICC reasoning
fallacies of Zanartu et al. 2024 (doi.org/10.1038/s41598-024-76139-w).

## Layout

| path | rows | assistant turn |
|---|---:|---|
| `recot/train.jsonl` | {counts['recot/train.jsonl']} | full deconstruction trace + YAML |
| `recot/train_eval.jsonl` | {counts['recot/train_eval.jsonl']} | full deconstruction trace + YAML |
| `labels/train.jsonl` | {counts['labels/train.jsonl']} | YAML answer only |
| `labels/train_eval.jsonl` | {counts['labels/train_eval.jsonl']} | YAML answer only |

Both arms come from one teacher pass over the same rows with the same stratified
split (seed 42), so they differ only in whether the reasoning is present. That
is the ablation for the source paper's own thesis: is the manual deconstruction
what makes the dataset valuable, or do the labels alone suffice?

`split_manifest.json` in each folder records the seed, the rule, and which
itemId went to which split, so the split is checkable without these files.

## Decontamination

5 of the 1,796 training rows are excluded here. Normalising away punctuation and
case, then confirming by character diff, three training rows are >=98% identical
to an evaluation row:

| training row | eval row | difference |
|---|---|---|
| `train-0605` | `test-0079` | one capital letter, a trailing period |
| `train-1390` | `val-0382` | curly quotes vs a period |
| `train-1599` | `val-0113` | one apostrophe |

Two further rows, `train-0223` and `train-0611`, are the same sentence carrying
different gold labels (`single cause` and `misrepresentation`), so the teacher
produced contradictory traces for one input.

The same conflict exists inside the validation split (`val-0108` /`val-0386`,
`false equivalence` vs `single cause`) and is left untouched — correcting
someone else's evaluation annotation is not appropriate; it is reported as label
noise instead.

## Provenance

Source rows are the FLICC training split, vendored verbatim from
github.com/fzanart/FLICC. **Only the training split** — no val or test row
appears here.

Traces were generated by a teacher LLM shown the gold label and asked to produce
reasoning arriving at it without revealing it was told. Traces whose final label
did not match gold were discarded, not repaired.

The system prompt is embedded per row and is the same prompt the student meets
at inference. The user turn is the plain classification turn — it never contains
the gold label or the teacher trigger.

## Codebook

The definitions in the system prompt are **not** the paper's Table 1. They were
rewritten from the training split, because Table 1 as published does not fully
specify the annotation practice. On held-out test data this was worth +0.147
macro F1 on Claude Opus 4.7 and +0.191 on GPT-5.5 over the same prompt carrying
Table 1 verbatim. See `docs/decisions.md` in the source repo.
"""

if args.dry_run:
    print("\n--- dataset card ---")
    print(card)
    sys.exit(0)

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN not set (looked in .env at flicc/, debunk/, and repo root)")
api = HfApi(token=token)
api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
for dest, src in files:
    api.upload_file(path_or_fileobj=str(src), path_in_repo=dest,
                    repo_id=args.repo, repo_type="dataset")
    print(f"  uploaded {dest}")
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                repo_id=args.repo, repo_type="dataset")
print(f"\nhttps://huggingface.co/datasets/{args.repo}")
