# /// script
# dependencies = [
#   "unsloth",
#   "datasets",
#   "trl>=0.12.0",
#   "huggingface_hub[hf_transfer]",
#   "transformers>=5.2.0",
#   "flash-linear-attention",
#   "trackio",
#   "tensorboard",
# ]
# ///
"""
Joint SFT on HF Jobs: Qwen3.5-9B trained on (CARDS train + Wind train),
early-stopped on (CARDS train_eval + Wind train_eval).

val/test of either project are deliberately not loaded.

Outputs (both public on HF Hub):
  - iRanadheer/CARDS-Wind-Qwen3.5-9B-lora    (LoRA adapter)
  - iRanadheer/CARDS-Wind-Qwen3.5-9B         (merged BF16)
"""
import os
import sys
import time
import subprocess

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
subprocess.run(["nvidia-smi"], check=False)

import torch
print(f"Torch {torch.__version__}  CUDA={torch.cuda.is_available()}  "
      f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig

HF_USERNAME    = "iRanadheer"
CARDS_REPO     = f"{HF_USERNAME}/cards_sft_dataset"
WIND_REPO      = f"{HF_USERNAME}/wind-opposition-sft"
VARIANT        = "CARDS-Wind-Qwen3.5-9B"
LORA_REPO      = f"{HF_USERNAME}/{VARIANT}-lora"
MERGED_REPO    = f"{HF_USERNAME}/{VARIANT}"
BASE_MODEL     = "Qwen/Qwen3.5-9B"
MAX_SEQ_LENGTH = 8192
OUTPUT_DIR     = VARIANT
HF_TOKEN       = os.environ["HF_TOKEN"]

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print("[1/6] Loading model...")
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print(f"  loaded in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 2. Load + combine datasets
# ---------------------------------------------------------------------------
print("\n[2/6] Loading + combining datasets (CARDS + Wind)...")
t0 = time.time()
cards_train = load_dataset(CARDS_REPO, data_files="cards_train.jsonl",      split="train", token=HF_TOKEN).select_columns(["messages"])
cards_eval  = load_dataset(CARDS_REPO, data_files="cards_train_eval.jsonl", split="train", token=HF_TOKEN).select_columns(["messages"])
wind_train  = load_dataset(WIND_REPO,  data_files="train/train.jsonl",      split="train", token=HF_TOKEN).select_columns(["messages"])
wind_eval   = load_dataset(WIND_REPO,  data_files="train/train_eval.jsonl", split="train", token=HF_TOKEN).select_columns(["messages"])
print(f"  CARDS train={len(cards_train)} eval={len(cards_eval)} | "
      f"Wind train={len(wind_train)} eval={len(wind_eval)}")

train_dataset = concatenate_datasets([cards_train, wind_train]).shuffle(seed=42)
eval_dataset  = concatenate_datasets([cards_eval,  wind_eval]).shuffle(seed=42)
print(f"  joint train={len(train_dataset)} eval={len(eval_dataset)}")

def apply_template(examples):
    out = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True,
        )
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        out.append(text)
    return {"text": out}

train_dataset = train_dataset.map(apply_template, batched=True, remove_columns=["messages"])
eval_dataset  = eval_dataset.map(apply_template,  batched=True, remove_columns=["messages"])
print(f"  templated in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 3. Configure trainer
# ---------------------------------------------------------------------------
print("\n[3/6] Configuring trainer...")
config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    push_to_hub=True,
    hub_model_id=LORA_REPO,
    hub_private_repo=False,

    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_length=MAX_SEQ_LENGTH,

    logging_steps=5,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=6,

    eval_strategy="steps",
    eval_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    warmup_steps=10,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    weight_decay=0.01,
    seed=42,
    bf16=True,

    report_to="trackio",
    run_name="cards-wind-9b",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
)

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
print("\n[4/6] Training...")
t0 = time.time()
res = trainer.train()
print(f"  done in {(time.time() - t0) / 60:.1f} min  train_loss={res.metrics.get('train_loss')}")
try:
    print(f"  final eval_loss={trainer.evaluate().get('eval_loss')}")
except Exception as e:
    print(f"  eval failed: {e}")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 5. Push LoRA adapter
# ---------------------------------------------------------------------------
print(f"\n[5/6] Pushing LoRA adapter to {LORA_REPO}...")
try:
    trainer.push_to_hub()
    print(f"  https://huggingface.co/{LORA_REPO}")
except Exception as e:
    print(f"  push_to_hub failed: {e}; uploading folder manually...")
    trainer.save_model(OUTPUT_DIR)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(LORA_REPO, private=False, exist_ok=True)
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=LORA_REPO, repo_type="model",
                      commit_message=f"Upload {VARIANT} LoRA")

# ---------------------------------------------------------------------------
# 6. Merge LoRA + push full BF16 model
# ---------------------------------------------------------------------------
print(f"\n[6/6] Merging LoRA + pushing full model to {MERGED_REPO}...")
try:
    merged = f"{OUTPUT_DIR}_merged"
    model.save_pretrained_merged(merged, tokenizer)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(MERGED_REPO, private=False, exist_ok=True)
    api.upload_folder(folder_path=merged, repo_id=MERGED_REPO, repo_type="model",
                      commit_message=f"Upload merged {VARIANT}")
    print(f"  https://huggingface.co/{MERGED_REPO}")
except Exception as e:
    print(f"  merge+push failed: {e}")

print(f"\nDone! ({VARIANT})")
