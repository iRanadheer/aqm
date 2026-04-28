"""
Prepare CARDS training splits in OpenAI fine-tune (messages) format.

Inputs:
    data/training_recot_opus.jsonl   - teacher RECoT data (text, true_claims, response, model)

Outputs:
    data/cards_train.jsonl              - SFT messages, RECoT (90% stratified)
    data/cards_train_eval.jsonl         - SFT messages, RECoT (10% stratified, early-stopping mirror)
    data/cards_train_norecot.jsonl      - same 90% rows, <think> stripped, no CoT trigger
    data/cards_train_eval_norecot.jsonl - same 10% rows, <think> stripped, no CoT trigger

Eval splits (cards_val.jsonl, cards_test.jsonl) are frozen canonical artifacts
checked into data/. They are not regenerated here.

Usage:
    python prepare_splits.py
"""

import json
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from prompts import slim_system_instruction, cot_trigger

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RANDOM_STATE = 42


def build_sft_record(text, response, use_recot=True):
    """Wrap (text, response) as an OpenAI chat fine-tune record.

    use_recot=False strips <think>...</think> and drops the CoT trigger from
    the user message — same boundary, just no reasoning shown.
    """
    if use_recot:
        user_content = f"### Text:\n{text}\n\n{cot_trigger}"
    else:
        response = strip_reasoning(response)
        user_content = f"### Text:\n{text}"
    return {"messages": [
        {"role": "system", "content": slim_system_instruction},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]}


def strip_reasoning(response):
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response


def parse_claims_from_response(response):
    """Pull category codes from a teacher response for stratification."""
    after_think = response.split("</think>")[-1] if "</think>" in response else response
    match = re.search(r"categories:\s*\n((?:\s*-\s*.+\n?)+)", after_think)
    if match:
        return sorted(re.findall(r"-\s*([\d_]+)", match.group(1)))
    return ["0_0_0"]


def prepare_train_splits():
    path = os.path.join(DATA_DIR, "training_recot_opus.jsonl")
    with open(path) as f:
        raw = [json.loads(line) for line in f]
    print(f"Loaded {len(raw)} RECoT training samples from training_recot_opus.jsonl")

    primary_labels = [parse_claims_from_response(r["response"])[0] for r in raw]

    counts = pd.Series(primary_labels).value_counts()
    rare = set(counts[counts < 2].index)
    strat_keys = ["_rare_" if l in rare else l for l in primary_labels]

    train_idx, eval_idx = train_test_split(
        range(len(raw)),
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=strat_keys,
    )

    for name, indices in [("cards_train", train_idx), ("cards_train_eval", eval_idx)]:
        for suffix, use_recot in [("", True), ("_norecot", False)]:
            records = [
                build_sft_record(raw[i]["text"], raw[i]["response"], use_recot=use_recot)
                for i in indices
            ]
            out_path = os.path.join(DATA_DIR, f"{name}{suffix}.jsonl")
            with open(out_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            print(f"  {name}{suffix}.jsonl: {len(records)} samples")


if __name__ == "__main__":
    print("=" * 60)
    print("Preparing CARDS training splits (90/10 stratified)")
    print("=" * 60)
    prepare_train_splits()
    print("\nDone. Files written to:", DATA_DIR)
