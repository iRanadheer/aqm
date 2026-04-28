# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl>=0.12.0",
#     "huggingface_hub[hf_transfer]",
#     "tensorboard",
#     "transformers>=5.2.0",
#     "flash-linear-attention",
# ]
# ///
# Note: On Hopper (H200) with Triton >= 3.4.0, flash-linear-attention's
# gated_chunk_bwd_dqkwg is buggy and requires the `tilelang` backend.
# On Ampere (A100) and Ada (L40S), tilelang cannot probe CUDA and breaks
# training. Leave tilelang OUT of the PEP 723 header; install conditionally
# on the target GPU if needed (see README / training notes).

"""
Wind Opposition Qwen3.5-4B SFT Training

Examples:
  python ft/train_qwen_sft.py
  python ft/train_qwen_sft.py --model 2b
  python ft/train_qwen_sft.py --no-merge         # skip the merge step, push LoRA adapter only
"""

import argparse
import os
import sys
import subprocess
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
MODELS = {
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "2b": "Qwen/Qwen3.5-2B",
    "4b": "Qwen/Qwen3.5-4B",
    "9b": "Qwen/Qwen3.5-9B",
    "27b": "Qwen/Qwen3.5-27B",
}

DATASETS = {
    "wind": {
        "repo": "iRanadheer/wind-opposition-sft",
        "variant_prefix": "Windy-Qwen3.5",
    },
    "aerowcards": {
        "repo": "iRanadheer/aerowcards",
        "variant_prefix": "Aerowcards-Qwen3.5",
    },
}

parser = argparse.ArgumentParser(description="Qwen SFT training on HF dataset")
parser.add_argument("--dataset", type=str, default="wind", choices=DATASETS.keys(),
                    help="Dataset key (default: wind). Determines HF dataset repo + output variant prefix.")
parser.add_argument("--model", type=str, default="4b", choices=MODELS.keys(),
                    help="Model size (default: 4b)")
parser.add_argument("--batch-size", type=int, default=2, help="Per device batch size (default: 2)")
parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
parser.add_argument("--no-merge", dest="merge_and_push", action="store_false", default=True,
                    help="Skip merging LoRA into base model (push adapter only)")
args = parser.parse_args()

MODEL_SIZE = args.model
BASE_MODEL = MODELS[MODEL_SIZE]
DATASET_CFG = DATASETS[args.dataset]
# e.g., "Qwen/Qwen3.5-4B" → "4B"
MODEL_SHORT = BASE_MODEL.split("/")[-1].replace("Qwen3.5-", "")
VARIANT = f"{DATASET_CFG['variant_prefix']}-{MODEL_SHORT}"

print(f"{'='*60}")
print(f"  Dataset: {args.dataset} ({DATASET_CFG['repo']})")
print(f"  Variant: {VARIANT}")
print(f"  Base model: {BASE_MODEL}")
print(f"  Batch size: {args.batch_size}, Grad accum: {args.grad_accum}, Effective: {args.batch_size * args.grad_accum}")
print(f"  Merge and push: {args.merge_and_push}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
subprocess.run(["nvidia-smi"], check=False)

subprocess.run([
    "uv", "pip", "install",
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu128",
    "--reinstall",
    "--python", sys.executable,
], check=True)

import torch
print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import login
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

HF_USERNAME = "iRanadheer"
DATASET_REPO = DATASET_CFG["repo"]
# LoRA adapter: suffixed with -lora. Merged full model: plain VARIANT.
LORA_REPO = f"{HF_USERNAME}/{VARIANT}-lora"
MERGED_REPO = f"{HF_USERNAME}/{VARIANT}"
MODEL_REPO = LORA_REPO  # SFTTrainer pushes the LoRA adapter during training
MAX_SEQ_LENGTH = 8192


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print(f"[1/5] Loading {BASE_MODEL}...")
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
# 2. Load dataset
# ---------------------------------------------------------------------------
print(f"\n[2/5] Loading dataset...")
start = time.time()

train_dataset = load_dataset(DATASET_REPO, data_files="train/train.jsonl", split="train")
eval_dataset = load_dataset(DATASET_REPO, data_files="train/train_eval.jsonl", split="train")
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

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
eval_dataset = eval_dataset.map(apply_template, batched=True, remove_columns=["messages"])

print(f"Sample (first 200 chars): {train_dataset[0]['text'][:200]}")
print(f"Dataset ready in {time.time() - start:.1f}s")

# ---------------------------------------------------------------------------
# 3. Configure trainer
# ---------------------------------------------------------------------------
print(f"\n[3/5] Configuring trainer (variant={VARIANT})...")

OUTPUT_DIR = f"{VARIANT}-lora"

config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    push_to_hub=True,
    hub_model_id=MODEL_REPO,
    hub_private_repo=False,

    num_train_epochs=3,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
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
    run_name=f"{VARIANT}-lora",
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
print(f"\n[4/5] Training ({VARIANT})...")
start = time.time()
train_result = trainer.train()
train_time = time.time() - start

print(f"\nTraining completed in {train_time / 60:.1f} minutes")
train_loss = train_result.metrics.get("train_loss")
if train_loss:
    print(f"  Final train loss: {train_loss:.4f}")

print("\nRunning final evaluation...")
eval_loss = None
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
# 5. Save and push
# ---------------------------------------------------------------------------
print(f"\n[5/5] Pushing best checkpoint to Hub...")
try:
    trainer.push_to_hub()
    print(f"\nModel saved: https://huggingface.co/{MODEL_REPO}")
except Exception as e:
    print(f"  push_to_hub failed: {e}")
    print("  Saving locally and uploading manually...")
    trainer.save_model(OUTPUT_DIR)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(MODEL_REPO, private=False, exist_ok=True)
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=MODEL_REPO, repo_type="model",
                      commit_message=f"Upload {VARIANT} model")
    print(f"\nModel saved: https://huggingface.co/{MODEL_REPO}")

if args.merge_and_push:
    print(f"\nMerging LoRA and pushing full model to {MERGED_REPO}...")
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

print(f"\nDone! Training complete ({VARIANT}).")
