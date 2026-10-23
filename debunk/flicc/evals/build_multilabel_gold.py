"""Build expert label sets from the review export and the withheld slot key.

The final `expert_labels` field is authoritative even when `expert_choice` is
"neither": that choice can carry zero, one, or several valid labels. The expert
confirmed that UI limitations required clicking "neither" to enter a corrected
set, even if the set includes one or both displayed options.
Run from anywhere: python evals/build_multilabel_gold.py [--selfcheck].
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/adjudication"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def labels(raw, valid):
    if not isinstance(raw, str):
        raise ValueError(f"Missing expert_labels: {raw!r}")
    if raw.strip().lower() == "none":
        return []
    parts = [p.strip() for p in re.split(r"[;,]", raw)]
    if not all(parts) or len(set(parts)) != len(parts) or any(p not in valid for p in parts):
        raise ValueError(f"Invalid expert_labels: {raw!r}")
    return sorted(parts)


def build(export, keys, blind, valid):
    def index(rows, field):
        result = {r[field]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError(f"Duplicate {field}")
        return result

    key = index(keys, "itemId")
    items = index(blind, "itemId")
    if len(export) != len(key) or len(items) != len(key):
        raise ValueError("Export, key and blind set have different sizes")
    out = []
    seen = set()
    for r in export:
        item = r["item_id"]
        if item in seen or item not in key or item not in items:
            raise ValueError(f"Duplicate or unknown item: {item}")
        seen.add(item)
        k, b = key[item], items[item]
        a, bb = r["option_a_label"], r["option_b_label"]
        gold_slot = k["gold_slot"]
        if (gold_slot not in {"a", "b"} or a == bb or
            b["a"]["label"] != a or b["b"]["label"] != bb or
            b["content"] != r["content"] or
            (a if gold_slot == "a" else bb) != k["gold_label"] or
            (bb if gold_slot == "a" else a) != k["model_label"] or
            k[f"{gold_slot}_source"] != "gold" or
            k[f"{'b' if gold_slot == 'a' else 'a'}_source"] == "gold"):
            raise ValueError(f"Slot/key/blind mismatch: {item}")
        chosen = r["expert_choice"]
        if chosen not in {"a", "b", "both", "neither"}:
            raise ValueError(f"Unknown choice: {item}: {chosen!r}")
        selected = labels(r["expert_labels"], valid)
        expected = {"a": [a], "b": [bb], "both": sorted([a, bb])}
        if chosen != "neither" and selected != expected[chosen]:
            raise ValueError(f"Choice and final labels disagree: {item}")
        raw = json.loads(r["expert_annotation_raw"])["label"]
        raw_labels = sorted(raw) if isinstance(raw, list) else labels(raw, valid)
        if (r["expert_label_saved"] != r["expert_labels"] or
            raw_labels != selected):
            raise ValueError(f"Saved/raw label mismatch: {item}")
        out.append({
            "item_id": item, "content": r["content"],
            "gold_labels": selected, "expert_choice": chosen,
            "gold_label": k["gold_label"], "model_label": k["model_label"],
            "model_was_right": k["model_label"] in selected,
            "gold_was_right": k["gold_label"] in selected,
        })
    return out


def demo():
    valid = {"gold", "model", "extra"}
    k = {"itemId": "x", "gold_slot": "b", "gold_label": "gold",
         "model_label": "model", "a_source": "model", "b_source": "gold"}
    b = {"itemId": "x", "content": "text", "a": {"label": "model"},
         "b": {"label": "gold"}}
    r = {"item_id": "x", "content": "text", "option_a_label": "model",
         "option_b_label": "gold", "expert_choice": "neither",
         "expert_labels": "model, extra", "expert_label_saved": "model, extra",
         "expert_annotation_raw": '{"label":"model, extra"}'}
    assert build([r], [k], [b], valid)[0]["gold_labels"] == ["extra", "model"]
    assert build([r], [k], [b], valid)[0]["model_was_right"]
    r["expert_choice"] = "a"
    try:
        build([r], [k], [b], valid)
    except ValueError:
        pass
    else:
        raise AssertionError("Click/text disagreement went undetected")
    r["expert_labels"] = r["expert_label_saved"] = "model"
    r["expert_annotation_raw"] = '{"label":"model"}'
    assert build([r], [k], [b], valid)[0]["model_was_right"]
    r["expert_choice"] = "neither"
    r["expert_labels"] = r["expert_label_saved"] = "None"
    r["expert_annotation_raw"] = '{"label":"None"}'
    assert build([r], [k], [b], valid)[0]["gold_labels"] == []
    print("self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path, default=DATA / "flicc-disagreement-review-export.jsonl")
    ap.add_argument("--key", type=Path, default=DATA / "test_disagreements_key.jsonl")
    ap.add_argument("--blind", type=Path, default=DATA / "test_disagreements.jsonl")
    ap.add_argument("--out", type=Path, default=DATA / "multilabel_gold.jsonl")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        demo()
    else:
        valid = {r["label"] for r in read(DATA / "labels.jsonl")}
        out = build(read(args.export), read(args.key), read(args.blind), valid)
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
        print(f"{len(out)} items; labels per item: {dict(sorted(Counter(len(r['gold_labels']) for r in out).items()))}")
        print(f"gold admitted: {sum(r['gold_was_right'] for r in out)}/{len(out)}")
        print(f"model admitted: {sum(r['model_was_right'] for r in out)}/{len(out)}")
