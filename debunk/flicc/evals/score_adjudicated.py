"""Compare original and review-rule test accuracy using all saved expert labels.

A reviewed prediction is correct if it belongs to the expert's final set;
elsewhere it retains the original gold. Empty sets admit no prediction.
Macro F1 is checked only to verify the reproduced DeBERTa checkpoint.
"""

import argparse
from pathlib import Path
import sys

from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prompts import FALLACY_LABELS, extract_raw_label  # noqa: E402
from build_multilabel_gold import build, read  # noqa: E402

LABELS = list(FALLACY_LABELS)
WRONG = "__unparsed__"
DATA = ROOT / "data/adjudication"
DISPLAY = {
    "c3ds-flicc-qwen3-5-9b-deconstruct-revised": "FLICC-Qwen3.5-9B",
    "openai-gpt-5-5-deconstruct-revised": "GPT-5.5",
    "c3ds-flicc-qwen3-5-27b-deconstruct-revised": "FLICC-Qwen3.5-27B",
    "anthropic-claude-opus-4-7-deconstruct-revised": "Claude Opus 4.7",
    "c3ds-flicc-qwen3-5-2b-deconstruct-revised": "FLICC-Qwen3.5-2B",
    "c3ds-flicc-qwen3-5-4b-deconstruct-revised": "FLICC-Qwen3.5-4B",
    "c3ds-flicc-qwen3-5-0-8b-deconstruct-revised": "FLICC-Qwen3.5-0.8B",
    "qwen-qwen3-5-27b-deconstruct-revised-structured": "Qwen3.5-27B (base)",
    "qwen-qwen3-5-9b-deconstruct-revised-structured": "Qwen3.5-9B (base)",
    "qwen-qwen3-5-4b-deconstruct-revised-structured": "Qwen3.5-4B (base)",
    "qwen-qwen3-5-2b-deconstruct-revised-structured": "Qwen3.5-2B (base)",
    "qwen-qwen3-5-0-8b-deconstruct-revised-structured": "Qwen3.5-0.8B (base)",
    "anthropic-claude-opus-4-7-deconstruct-published": "Claude Opus 4.7 (published prompt)",
    "anthropic-claude-opus-4-7-bare": "Claude Opus 4.7 (bare prompt)",
    "openai-gpt-5-5-deconstruct-published": "GPT-5.5 (published prompt)",
    "openai-gpt-5-5-bare": "GPT-5.5 (bare prompt)",
    "fzanartu-flicc-deberta-v2-xlarge": "DeBERTa-v2-xlarge (reproduced)",
}


def assess(rows, expert):
    ids = [r["itemId"] for r in rows]
    if len(ids) != len(set(ids)) or not set(expert) <= set(ids):
        raise ValueError("Duplicate run IDs or missing reviewed items")
    gold = [r["true_fallacy"] for r in rows]
    if any(label not in LABELS for label in gold):
        raise ValueError("Unknown original gold label")
    pred = [extract_raw_label(str(r["response"])) if "response" in r
            else r["pred_label"] for r in rows]
    pred = [p if p in LABELS else WRONG for p in pred]
    original = [p == g for p, g in zip(pred, gold)]
    membership = []
    for r, p, g in zip(rows, pred, gold):
        if r["itemId"] in expert:
            review = expert[r["itemId"]]
            if g != review["gold_label"] or r["text"] != review["content"]:
                raise ValueError(f"Run does not match the review: {r['itemId']}")
            membership.append(p in review["gold_labels"])
        else:
            membership.append(p == g)
    errors = sum(not ok for ok in original)
    reviewed = sum(not ok and i in expert for i, ok in zip(ids, original))
    rescued = sum(not before and after for before, after in zip(original, membership))
    lost = sum(before and not after for before, after in zip(original, membership))
    return dict(n=len(rows), f1=f1_score(gold, pred, labels=LABELS,
                                       average="macro", zero_division=0),
                original=sum(original) / len(rows),
                member=sum(membership) / len(rows), errors=errors,
                reviewed=reviewed, rescued=rescued, lost=lost)


def replace_test_table(document, table):
    start = "## test split (n = 256)\n"
    end = "## Reading these numbers"
    if document.count(start) != 1 or document.count(end) != 1:
        raise ValueError("Cannot locate the unique test-table section in results.md")
    before, rest = document.split(start, 1)
    _, after = rest.split(end, 1)
    return before + table + "\n" + end + after


def demo():
    a, b = LABELS[:2]
    examples = read(ROOT / "data/results/test/c3ds-flicc-qwen3-5-27b-deconstruct-revised.jsonl")
    sample = next(r for r in examples if extract_raw_label(str(r["response"])) in LABELS)
    p = extract_raw_label(str(sample["response"]))
    x = dict(sample, itemId="x", true_fallacy=a if p != a else b)
    e = {"x": {"gold_label": x["true_fallacy"], "content": x["text"],
               "gold_labels": [p, b if p != b else a]}}
    assert assess([x], e)["member"] == 1
    e["x"]["gold_labels"] = []
    assert assess([x], e)["member"] == 0
    assert assess([x], e)["original"] == 0
    x.pop("response")
    x["pred_label"] = p
    assert assess([x], e)["member"] == 0  # saved DeBERTa label, empty expert set
    assert replace_test_table("a\n## test split (n = 256)\nold\n## Reading these numbers\nb",
                              "## test split (n = 256)\nnew\n").count("new") == 1
    print("self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="Write markdown table (otherwise print only)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        demo()
        raise SystemExit

    expected = build(read(DATA / "flicc-disagreement-review-export.jsonl"),
                     read(DATA / "test_disagreements_key.jsonl"),
                     read(DATA / "test_disagreements.jsonl"),
                     {r["label"] for r in read(DATA / "labels.jsonl")})
    stored = read(DATA / "multilabel_gold.jsonl")
    if stored != expected:
        raise ValueError("Derived gold file differs from the raw review; rebuild it")
    expert = {r["item_id"]: r for r in stored}
    rows = []
    for stem, name in DISPLAY.items():
        path = ROOT / "data/results/test" / f"{stem}.jsonl"
        result = assess(read(path), expert)
        if result["n"] != 256:
            raise ValueError(f"Unexpected test size: {path}")
        if stem == "fzanartu-flicc-deberta-v2-xlarge":
            if abs(result["f1"] - 0.7332474105800544) > 1e-6:
                raise ValueError("DeBERTa no longer reproduces published test result")
            kind = "reproduced baseline"
        elif stem.startswith("qwen-qwen"):
            kind = "base"
        elif stem.startswith("c3ds-flicc"):
            kind = "fine-tuned"
        else:
            kind = "frontier"
        rows.append((name, kind, result))
    rows.extend([
        ("GPT-4-0125 (Zanartu et al.)", "published baseline", {"f1": 0.300, "original": 0.320}),
        ("Gemini-pro (Zanartu et al.)", "published baseline", {"f1": 0.110, "original": 0.210}),
    ])
    rows.sort(key=lambda x: -x[2]["original"])
    lines = ["## test split (n = 256)", "",
             "Both score columns are **accuracy** on the same 256 items.",
             "Original accuracy uses single-label gold. Review-rule accuracy uses",
             "the expert's full saved labels on the 93 reviewed items: a match to",
             "any saved label counts; an empty set accepts none. Other items retain",
             "the original gold. The DeBERTa run reproduced the published 0.73",
             "macro F1 (0.733 here); the table shows its accuracy, not its F1.", "",
             "| Model | Kind | Original accuracy | Review-rule accuracy | Change (pp) | Errors reviewed | Rescued / lost |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, kind, r in rows:
        if "member" in r:
            lines.append(f"| {name} | {kind} | {r['original']:.3f} | "
                         f"{r['member']:.3f} | {(r['member']-r['original'])*100:+.1f} | "
                         f"{r['reviewed']}/{r['errors']} | {r['rescued']}/{r['lost']} |")
        else:
            lines.append(f"| {name} | {kind} | {r['original']:.3f} | — | — | — | — |")
    lines += ["",
              "Every model with a review-rule score uses the same 93 items,",
              "selected from the 27B's errors (93/93 covered).",
              "Coverage of each other model's errors is shown in the table; in",
              "particular, DeBERTa has 39/66 covered. Unreviewed errors still use",
              "the original gold, so changes are not a fair adjusted ranking.",
              "The expert confirmed that UI limitations sometimes required clicking",
              "`neither` to enter corrected labels; all saved labels were retained."]
    table = "\n".join(lines) + "\n"
    print(table, end="")
    if args.out:
        if args.out.name == "results.md":
            args.out.write_text(replace_test_table(args.out.read_text(), table))
        else:
            args.out.write_text(table)
