"""Run Zanartu et al.'s published FLICC DeBERTa on the 256 test items.

CPU is sufficient. Install in a venv before running (see command in reply).
Pin the published model revision; verify its original score before saving
predictions. Match item IDs by exact text against the authors' test CSV.
"""

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CSV = ROOT.parents[2] / "FLICC/Data/fallacy_test.csv"
REFERENCE = DATA / "results/test/c3ds-flicc-qwen3-5-27b-deconstruct-revised.jsonl"
REVIEW = DATA / "adjudication/multilabel_gold.jsonl"
OUT = DATA / "results/test/fzanartu-flicc-deberta-v2-xlarge.jsonl"
SUMMARY = OUT.with_suffix(".summary.json")
MODEL = "fzanartu/flicc"
REVISION = "857fad67e906bc61985ea8cd814eeb334c7e1f31"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def match_test(csv_rows, reference):
    by_text = {r["text"]: r for r in reference}
    if len(csv_rows) != 256 or len(by_text) != 256 or len({r["text"] for r in csv_rows}) != 256:
        raise ValueError("Expected 256 unique test passages in both files")
    if {r["text"] for r in csv_rows} != set(by_text):
        raise ValueError("Authors' test passages do not match our test passages")
    if any(r["label"] != by_text[r["text"]]["true_fallacy"] for r in csv_rows):
        raise ValueError("Authors' gold labels differ from our test labels")
    return [by_text[r["text"]]["itemId"] for r in csv_rows]


def selfcheck():
    c = [{"text": "one", "label": "a"}, {"text": "two", "label": "b"}]
    r = [{"itemId": "id2", "text": "two", "true_fallacy": "b"},
         {"itemId": "id1", "text": "one", "true_fallacy": "a"}]
    try:
        match_test(c, r)
    except ValueError as e:
        assert "256" in str(e)
    else:
        raise AssertionError("Incomplete test split accepted")
    with CSV.open(newline="") as f:
        assert len(match_test(list(csv.DictReader(f)), read_jsonl(REFERENCE))) == 256
    print("self-check ok: exact text and gold label match for 256 items")


def main():
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    with CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    reference = read_jsonl(REFERENCE)
    ids = match_test(rows, reference)
    review = {r["item_id"]: r for r in read_jsonl(REVIEW)}
    if len(review) != 93 or not set(review) <= set(ids):
        raise ValueError("Expert review does not match these test items")
    for r, item_id in zip(rows, ids):
        if item_id in review and (r["text"] != review[item_id]["content"] or
                                  r["label"] != review[item_id]["gold_label"]):
            raise ValueError(f"Expert review mismatch: {item_id}")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    from sys import path
    path.insert(0, str(ROOT))
    from prompts import FALLACY_LABELS
    if [model.config.id2label[i] for i in range(12)] != list(FALLACY_LABELS):
        raise ValueError("Checkpoint labels do not match the 12-label codebook")
    classify = pipeline("text-classification", model=model, tokenizer=tokenizer,
                        device=-1)
    outputs = []
    for start in range(0, len(rows), 4):
        outputs.extend(classify([r["text"] for r in rows[start:start + 4]],
                                batch_size=4, top_k=None))
        done = min(start + 4, len(rows))
        if done % 32 == 0 or done == len(rows):
            print(f"Inferred {done}/{len(rows)} items", flush=True)
    probabilities = [{p["label"]: p["score"] for p in scores} for scores in outputs]
    if any(set(p) != set(FALLACY_LABELS) for p in probabilities):
        raise ValueError("Model did not return scores for all 12 labels")
    predictions = [max(p, key=p.get) for p in probabilities]
    original_f1 = f1_score([r["label"] for r in rows], predictions,
                           labels=list(FALLACY_LABELS), average="macro")
    original_acc = accuracy_score([r["label"] for r in rows], predictions)
    print(f"Original test: macro F1 {original_f1:.4f}, accuracy {original_acc:.4f}")
    if abs(original_f1 - 0.7332474105800544) > 0.005:
        raise ValueError("Published macro F1 not reproduced; predictions not saved")

    from prompts import extract_raw_label
    reference_by_id = {r["itemId"]: r for r in reference}
    records = []
    for i, r, p, scores in zip(ids, rows, predictions, probabilities):
        expert_labels = review[i]["gold_labels"] if i in review else None
        original_correct = p == r["label"]
        records.append({
            "itemId": i, "text": r["text"], "true_fallacy": r["label"],
            "pred_label": p, "confidence": scores[p],
            "probabilities": scores,
            "27b_pred_label": extract_raw_label(str(reference_by_id[i]["response"])),
            "reviewed": i in review, "expert_labels": expert_labels,
            "original_correct": original_correct,
            "expert_set_correct": p in expert_labels if expert_labels is not None else original_correct,
        })
    overlap = sum(r["reviewed"] and not r["original_correct"] for r in records)
    deberta_errors = sum(not r["original_correct"] for r in records)
    rescued = sum(r["reviewed"] and not r["original_correct"] and r["expert_set_correct"]
                  for r in records)
    lost = sum(r["reviewed"] and r["original_correct"] and not r["expert_set_correct"]
               for r in records)
    adjusted = sum(r["expert_set_correct"] for r in records)
    summary = {
        "model": MODEL, "revision": REVISION, "n": len(records),
        "original_macro_f1": original_f1, "original_accuracy": original_acc,
        "per_class": classification_report([r["label"] for r in rows], predictions,
                                           labels=list(FALLACY_LABELS), output_dict=True,
                                           zero_division=0),
        "deberta_errors": deberta_errors, "27b_errors": len(review),
        "error_overlap": overlap, "deberta_errors_outside_review": deberta_errors - overlap,
        "rescued_by_review": rescued, "lost_after_review": lost,
        "expert_set_correct": adjusted, "expert_set_accuracy": adjusted / len(records),
    }
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"DeBERTa errors: {deberta_errors}; overlap with 27B's 93 errors: {overlap}")
    print(f"Reviewed DeBERTa errors accepted: {rescued}; original hits rejected: {lost}")
    print(f"Expert-set membership accuracy: {adjusted}/256 = {adjusted / 256:.3f}")
    print(f"Saved per-item predictions: {OUT}\nSaved full metrics: {SUMMARY}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    selfcheck() if args.selfcheck else main()
