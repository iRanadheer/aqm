"""
Prepare CARDS dataset splits.

Inputs:
    data/training_recot_opus.jsonl       - teacher RECoT data (text, true_claims, response, model)
    data/training_recot_opus_chat.jsonl  - teacher chat-format RECoT data (teacher.py --chat)
    data/congress_test.csv               - source for eval splits (2,051 congressional rows)

Outputs (all written to data/):
    cards_train.jsonl               - SFT messages, RECoT (90% stratified)
    cards_train_eval.jsonl          - SFT messages, RECoT (10% stratified, early-stopping mirror)
    cards_train_norecot.jsonl       - same 90% rows, <think> stripped
    cards_train_eval_norecot.jsonl  - same 10% rows, <think> stripped
    cards_train_chat.jsonl          - SFT messages, chat format (90% stratified)
    cards_train_eval_chat.jsonl    - SFT messages, chat format (10% stratified)
    cards_val.jsonl                 - {id, text, true_claims} (30% stratified of congress_test)
    cards_test.jsonl                - {id, text, true_claims} (70% stratified of congress_test)

No CoT trigger anywhere — user messages are bare; the system prompt alone
pins the output format.

All splits use random_state=42 — deterministic given the inputs.

Usage:
    python prepare_splits.py             # build both train and test splits
    python prepare_splits.py --train     # train splits only
    python prepare_splits.py --test     # test splits only (cards_val + cards_test)
    python prepare_splits.py --chat     # chat train splits only
"""

import argparse
import json
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from prompts import (
    slim_chat_system_instruction,
    slim_system_instruction,
    slim_system_instruction_norecot,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RANDOM_STATE = 42


def build_sft_record(text, response, use_recot=True):
    """Wrap (text, response) as an OpenAI chat fine-tune record.

    use_recot=False strips <think>...</think> AND swaps the system prompt
    to the no-RECoT variant (no `<think>` directive in OUTPUT FORMAT) so
    system and target agree. The user message is identical for both.
    """
    if use_recot:
        sys_prompt = slim_system_instruction
    else:
        sys_prompt = slim_system_instruction_norecot
        response = strip_reasoning(response)
    return {"messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"### Text:\n{text}"},
        {"role": "assistant", "content": response},
    ]}


def build_chat_sft_record(text, response):
    """Wrap (text, response) as a chat-variant SFT record.

    The user message uses the same `### Text:` template as the API variant
    — one common input convention; only system prompt and response differ.
    """
    return {"messages": [
        {"role": "system", "content": slim_chat_system_instruction},
        {"role": "user", "content": f"### Text:\n{text}"},
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


def parse_true_claims(val):
    """Parse true_claims — handles numpy-style space-separated lists."""
    return sorted(re.findall(r"[\d_]+", str(val)))


def _norm_text(t):
    """Whitespace-collapsed, lowercased text — for contamination matching.

    Benchmark and train-pool copies of the same passage differ in whitespace
    and quoting, so exact matching misses them.
    """
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def load_benchmark_texts():
    """Normalized texts of cards_val + cards_test, for train decontamination."""
    texts = set()
    for name in ("cards_val.jsonl", "cards_test.jsonl"):
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"  WARNING: {name} not found — decontamination check incomplete")
            continue
        with open(path) as f:
            for line in f:
                texts.add(_norm_text(json.loads(line)["text"]))
    return texts


def load_api_teacher_rows():
    path = os.path.join(DATA_DIR, "training_recot_opus.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def stratified_train_eval_indices(raw):
    """The canonical 90/10 partition of the API teacher rows.

    Stratified on the first category code of the API response; rare labels
    (count<2) bucketed as `_rare_`. Seed 42. Every train variant (RECoT,
    no-RECoT, chat) derives its row partition from THIS function so all
    variants train/early-stop on exactly the same rows.
    """
    primary_labels = [parse_claims_from_response(r["response"])[0] for r in raw]
    counts = pd.Series(primary_labels).value_counts()
    rare = set(counts[counts < 2].index)
    strat_keys = ["_rare_" if l in rare else l for l in primary_labels]
    return train_test_split(
        range(len(raw)),
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=strat_keys,
    )


def prepare_train_splits():
    """Build cards_train{,_eval}{,_norecot}.jsonl from training_recot_opus.jsonl.

    The same indices are reused for the no-RECoT mirror so that both
    variants share an identical row partition.
    """
    raw = load_api_teacher_rows()
    print(f"Loaded {len(raw)} RECoT training samples from training_recot_opus.jsonl")

    train_idx, eval_idx = stratified_train_eval_indices(raw)
    bench = load_benchmark_texts()

    for name, indices in [("cards_train", train_idx), ("cards_train_eval", eval_idx)]:
        # Decontaminate AFTER the split: rows whose normalized text appears in
        # val/test are dropped; the partition of all other rows is unchanged.
        kept = [i for i in indices if _norm_text(raw[i]["text"]) not in bench]
        dropped = len(indices) - len(kept)
        for suffix, use_recot in [("", True), ("_norecot", False)]:
            records = [
                build_sft_record(raw[i]["text"], raw[i]["response"], use_recot=use_recot)
                for i in kept
            ]
            out_path = os.path.join(DATA_DIR, f"{name}{suffix}.jsonl")
            with open(out_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            msg = f"  {name}{suffix}.jsonl: {len(records)} samples"
            if dropped:
                msg += f" ({dropped} dropped: text appears in val/test)"
            print(msg)


def prepare_chat_train_splits():
    """Build cards_train_chat{,_eval}.jsonl from training_recot_opus_chat.jsonl.

    The 90/10 partition is NOT recomputed — it is inherited from the API
    teacher file via stratified_train_eval_indices(), joined by text. Chat
    and API variants therefore train/early-stop on exactly the same rows.
    """
    chat_path = os.path.join(DATA_DIR, "training_recot_opus_chat.jsonl")
    if not os.path.exists(chat_path):
        print(f"  WARNING: {chat_path} not found — run `python teacher.py --chat` first")
        return
    with open(chat_path) as f:
        chat_raw = [json.loads(line) for line in f]
    chat_by_text = {r["text"]: r["response"] for r in chat_raw}
    print(f"Loaded {len(chat_raw)} chat RECoT training samples from training_recot_opus_chat.jsonl")

    api_raw = load_api_teacher_rows()
    train_idx, eval_idx = stratified_train_eval_indices(api_raw)
    bench = load_benchmark_texts()

    for name, indices in [("cards_train_chat", train_idx), ("cards_train_eval_chat", eval_idx)]:
        records, missing, dropped = [], 0, 0
        for i in indices:
            text = api_raw[i]["text"]
            # Same decontamination as the API splits — keeps partitions identical.
            if _norm_text(text) in bench:
                dropped += 1
                continue
            response = chat_by_text.get(text)
            if response is None:
                missing += 1
                continue
            records.append(build_chat_sft_record(text, response))
        out_path = os.path.join(DATA_DIR, f"{name}.jsonl")
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        msg = f"  {name}.jsonl: {len(records)} samples"
        if dropped:
            msg += f" ({dropped} dropped: text appears in val/test)"
        if missing:
            msg += f" (WARNING: {missing} rows missing from chat teacher file — rerun teacher.py --chat)"
        print(msg)


def load_final_claims_overrides():
    """Load the post-review label set used by the published Nature paper.

    `data/mapping/final_claims_dict.json` is keyed by raw `congress_test`
    text and stores the revised label set as a numpy-array repr string
    (space-separated, e.g. ``"['6_1_3' '4_2_2' '4_1_5']"``). It differs
    from the seed `true_claims` column on ~5% of items; published metrics
    were computed against this revised set, so we promote it as canonical
    for the eval splits. Falls back silently to seed labels for any text
    not present in the override.
    """
    path = os.path.join(DATA_DIR, "mapping", "final_claims_dict.json")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — using seed true_claims only")
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for text, raw in d.items():
        if not isinstance(raw, str):
            continue
        try:
            out[text] = sorted(eval(raw.replace(" ", ", ")))
        except Exception:
            continue
    print(f"  Loaded {len(out)} final-label overrides from mapping/final_claims_dict.json")
    return out


def prepare_test_splits():
    """Build cards_val.jsonl (30%) and cards_test.jsonl (70%) from congress_test.csv.

    Stratified on the first seed `true_claims` code; rare labels (count<2)
    bucketed as `_rare_`. After the partition is fixed, labels are promoted
    from `mapping/final_claims_dict.json` (the Nature paper's revised set)
    where an override exists. With seed 42 + the override file this
    reproduces the canonical 615 / 1,436 split byte-for-byte.
    """
    src = os.path.join(DATA_DIR, "congress_test.csv")
    df = pd.read_csv(src)
    print(f"Loaded {len(df)} rows from congress_test.csv")

    df["tc"] = df["true_claims"].apply(parse_true_claims)
    df["primary_label"] = df["tc"].apply(lambda x: x[0])

    vc = df["primary_label"].value_counts()
    rare = set(vc[vc < 2].index)
    df["strat_key"] = df["primary_label"].apply(
        lambda x: "_rare_" if x in rare else x
    )

    df_val, df_test = train_test_split(
        df,
        test_size=0.7,
        random_state=RANDOM_STATE,
        stratify=df["strat_key"],
    )

    final_overrides = load_final_claims_overrides()

    for name, d in [("cards_val", df_val), ("cards_test", df_test)]:
        records = []
        n_overridden = 0
        for _, row in d.iterrows():
            seed_labels = row["tc"]
            final_labels = final_overrides.get(row["text"])
            if final_labels is not None and final_labels != seed_labels:
                labels = final_labels
                n_overridden += 1
            else:
                labels = seed_labels
            records.append({
                "id": row["id"],
                "text": row["text"],
                "true_claims": labels,
            })
        out_path = os.path.join(DATA_DIR, f"{name}.jsonl")
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        zero_pct = sum(r["true_claims"] == ["0_0_0"] for r in records) / len(records) * 100
        multi_pct = sum(len(r["true_claims"]) > 1 for r in records) / len(records) * 100
        print(
            f"  {name}.jsonl: {len(records)} rows "
            f"({zero_pct:.1f}% zero, {multi_pct:.1f}% multi-label, "
            f"{n_overridden} labels promoted from final_claims_dict)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", action="store_true", help="Build train splits only.")
    parser.add_argument("--test", action="store_true", help="Build test splits only (cards_val + cards_test).")
    parser.add_argument("--chat", action="store_true", help="Build chat train splits only.")
    args = parser.parse_args()

    do_train = args.train or not (args.train or args.test or args.chat)
    do_test = args.test or not (args.train or args.test or args.chat)
    do_chat = args.chat or not (args.train or args.test or args.chat)

    if do_train:
        print("=" * 60)
        print("Train splits (90/10 stratified, seed 42)")
        print("=" * 60)
        prepare_train_splits()

    if do_chat:
        print("\n" + "=" * 60)
        print("Chat train splits (90/10 stratified, seed 42)")
        print("=" * 60)
        prepare_chat_train_splits()

    if do_test:
        print("\n" + "=" * 60)
        print("Test splits (30/70 stratified, seed 42)")
        print("=" * 60)
        prepare_test_splits()

    print("\nDone. Files written to:", DATA_DIR)
