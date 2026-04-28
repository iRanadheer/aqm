# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth",
#   "datasets",
#   "trl>=0.12.0",
#   "huggingface_hub[hf_transfer]",
#   "transformers>=5.2.0",
#   "flash-linear-attention",
#   "tilelang",  # required on Hopper (H100/H200): triton >= 3.4 has a bug in gated chunk_bwd_dqkwg (#640). Do NOT include on Ampere/Ada — tilelang fails to probe CUDA there.
#   "tensorboard",
# ]
# ///
"""
Joint SFT: Qwen3.6-27B trained on (CARDS train + Wind train) and early-stopped
on (CARDS train_eval + Wind train_eval). val/test of either project are NEVER
loaded here.

Usage (on RunPod / vast.ai pod):
    huggingface-cli login
    tmux new -d -s train "uv run train_27b_joint.py 2>&1 | tee train.log"
    tmux attach -t train
"""

import os
import sys
import time
import subprocess

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run(["nvidia-smi"], check=False)

import torch
print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig

HF_USERNAME    = "iRanadheer"
CARDS_REPO     = f"{HF_USERNAME}/cards_sft_dataset"
WIND_REPO      = f"{HF_USERNAME}/wind-opposition-sft"
VARIANT        = "CARDS-Wind-Qwen3.6-27B"
LORA_REPO      = f"{HF_USERNAME}/{VARIANT}-lora"
MERGED_REPO    = f"{HF_USERNAME}/{VARIANT}"
BASE_MODEL     = "Qwen/Qwen3.6-27B"
MAX_SEQ_LENGTH = 8192
OUTPUT_DIR     = VARIANT

HF_TOKEN = os.environ.get("HF_TOKEN")

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print("[1/5] Loading Qwen3.6-27B...")
start = time.time()

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
print(f"Model loaded in {time.time() - start:.1f}s")

# ---------------------------------------------------------------------------
# 2. Load + combine datasets (CARDS + Wind)
# ---------------------------------------------------------------------------
print("\n[2/5] Loading datasets...")
start = time.time()

cards_train = load_dataset(CARDS_REPO, data_files="cards_train.jsonl",      split="train", token=HF_TOKEN)
cards_eval  = load_dataset(CARDS_REPO, data_files="cards_train_eval.jsonl", split="train", token=HF_TOKEN)
wind_train  = load_dataset(WIND_REPO,  data_files="train/train.jsonl",      split="train", token=HF_TOKEN)
wind_eval   = load_dataset(WIND_REPO,  data_files="train/train_eval.jsonl", split="train", token=HF_TOKEN)

print(f"  CARDS train: {len(cards_train)},  CARDS eval: {len(cards_eval)}")
print(f"  Wind  train: {len(wind_train)},  Wind  eval: {len(wind_eval)}")

# Keep only messages column so concatenation is clean across datasets
cards_train = cards_train.select_columns(["messages"])
cards_eval  = cards_eval.select_columns(["messages"])
wind_train  = wind_train.select_columns(["messages"])
wind_eval   = wind_eval.select_columns(["messages"])

train_dataset = concatenate_datasets([cards_train, wind_train]).shuffle(seed=42)
eval_dataset  = concatenate_datasets([cards_eval,  wind_eval]).shuffle(seed=42)
print(f"  Joint train: {len(train_dataset)},  Joint eval: {len(eval_dataset)}")

def apply_template(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        texts.append(text)
    return {"text": texts}

train_dataset = train_dataset.map(apply_template, batched=True, remove_columns=["messages"])
eval_dataset  = eval_dataset.map(apply_template,  batched=True, remove_columns=["messages"])

print(f"Sample (first 200 chars): {train_dataset[0]['text'][:200]}")
print(f"Dataset ready in {time.time() - start:.1f}s")

# ---------------------------------------------------------------------------
# 3. Configure trainer
# ---------------------------------------------------------------------------
print("\n[3/5] Configuring trainer...")

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

    report_to=["tensorboard"],
    run_name="CARDS-Wind-Qwen3.6-27B",
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
print("\n[4/5] Training...")
start = time.time()
train_result = trainer.train()
train_time = time.time() - start

print(f"\nTraining completed in {train_time / 60:.1f} minutes")
train_loss = train_result.metrics.get("train_loss")
if train_loss:
    print(f"  Final train loss: {train_loss:.4f}")

print("\nRunning final evaluation...")
try:
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss")
    if eval_loss:
        print(f"  Final eval loss: {eval_loss:.4f}")
except Exception as e:
    print(f"  Eval failed: {e}")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 5. Push LoRA adapter
# ---------------------------------------------------------------------------
print(f"\n[5/6] Pushing LoRA adapter to {LORA_REPO}...")
try:
    trainer.push_to_hub()
    print(f"  LoRA adapter saved: https://huggingface.co/{LORA_REPO}")
except Exception as e:
    print(f"  push_to_hub failed: {e}")
    print("  Saving locally and uploading manually...")
    trainer.save_model(OUTPUT_DIR)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(LORA_REPO, private=False, exist_ok=True)
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=LORA_REPO, repo_type="model",
                      commit_message=f"Upload {VARIANT} LoRA adapter")
    print(f"  LoRA adapter saved: https://huggingface.co/{LORA_REPO}")

# ---------------------------------------------------------------------------
# 6. Merge LoRA into base and push full model
# ---------------------------------------------------------------------------
print(f"\n[6/6] Merging LoRA and pushing full model to {MERGED_REPO}...")
try:
    merged_dir = f"{OUTPUT_DIR}_merged"
    model.save_pretrained_merged(merged_dir, tokenizer)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(MERGED_REPO, private=False, exist_ok=True)
    api.upload_folder(folder_path=merged_dir, repo_id=MERGED_REPO, repo_type="model",
                      commit_message=f"Upload merged {VARIANT} model")
    print(f"  Merged model saved: https://huggingface.co/{MERGED_REPO}")
except Exception as e:
    print(f"  Merge+push failed: {e}")

print(f"\nDone! Joint training complete ({VARIANT}).")
