# /// script
# requires-python = ">=3.10"
# dependencies = [
#   # Pins follow cards/ft/train_moe.py, which documents this dependency set
#   # after it was worked out the hard way. The header inherited from the older
#   # cards/ft/train.py fails on HF Jobs:
#   #   AttributeError: module 'torch' has no attribute 'float4_e2m1fn_x2'
#   # — it pinned no torch at all, so uv resolved one older than the Unsloth
#   # build expects (float4_e2m1fn_x2 arrived in torch 2.8).
#   #
#   #  - torch==2.10.0: the only pin that matters. Its plain-PyPI wheel bundles
#   #    CUDA, so NO extra-index-url is needed, and it ships triton 3.6, so no
#   #    manual triton pin either. Inside unsloth's torch<2.11 cap.
#   #  - flash-linear-attention deliberately OMITTED: it drags in triton>=3.4 ->
#   #    tilelang -> "No CUDA in container". Dense Qwen3.5 does not need it.
#   #  - everything else left loose so uv resolves within unsloth's own ranges.
#   "unsloth==2026.5.9",
#   "torch==2.10.0",
#   "torchvision==0.25.0",
#   "transformers>=5.2.0",
#   "trl>=0.12.0",
#   "datasets",
#   "huggingface_hub[hf_transfer]",
#   "tensorboard",
# ]
# ///
"""FLICC SFT trainer — student model for twelve-way fallacy classification.

  uv run ft/train.py --base-model Qwen/Qwen3.5-4B
  uv run ft/train.py --base-model Qwen/Qwen3.5-4B --no-recot
  uv run ft/train.py --base-model Qwen/Qwen3.5-9B --lora-only

--recot (default) trains on the full deconstruction traces; --no-recot trains
on the same rows with the assistant turn reduced to the YAML answer. Both arms
live in one Hub dataset and come from one teacher pass, so the comparison
isolates the reasoning — the ablation for the source paper's claim that manual
deconstruction is what makes the dataset valuable.

Pushes a LoRA adapter to <hf_user>/<variant>-lora and, unless --lora-only,
merges and pushes the full BF16 model to <hf_user>/<variant>.

Recipe, hyperparameters, and the resumable checkpoint strategy follow
cards/ft/train.py — the same LoRA rank/alpha, optimiser, and schedule that
produced the CARDS and Wind students. Deliberately no --joint or --combined:
this taxonomy shares no label space with those chapters.

Runs identically on HF Jobs, Vast.ai, RunPod, or any cu128-capable GPU host.
Note: wind/docs/methodology.md records HF Jobs a100-large failing silently and
L40S hitting dependency issues for this workload; RunPod H200 was the fallback.
"""

import argparse
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

HF_USERNAME = os.environ.get("HF_USERNAME", "iRanadheer")
# The SFT dataset lives under iRanadheer (public) regardless of the namespace
# the model is pushed to — following cards/ft/train.py::COMBINED_REPO. Deriving
# it from HF_USERNAME instead sends an HF_USERNAME=C3DS run looking for
# C3DS/flicc-fallacy-sft and it dies at dataset load. Override with
# FLICC_DATASET if a copy really does live elsewhere.
FLICC_REPO = os.environ.get("FLICC_DATASET", "iRanadheer/flicc-fallacy-sft")

ap = argparse.ArgumentParser(description="FLICC fallacy SFT trainer")
ap.add_argument("--base-model", required=True,
                help="HF model id, e.g. Qwen/Qwen3.5-4B")
ap.add_argument("--recot", dest="recot", action="store_true", default=True)
ap.add_argument("--no-recot", dest="recot", action="store_false",
                help="Train on labels only — the control arm.")
ap.add_argument("--max-seq", type=int, default=8192,
                help="System prompt is ~1.6k tokens and traces ~500, so 8192 "
                     "is ample; lower only if VRAM-bound.")
ap.add_argument("--lora-only", dest="push_merged", action="store_false", default=True,
                help="Skip merging + pushing the full BF16 model.")
ap.add_argument("--variant", default=None, help="Override derived variant name")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch-size", type=int, default=1)
ap.add_argument("--grad-accum", type=int, default=8,
                help="Keep batch_size*grad_accum constant to preserve the "
                     "effective batch of the recipe.")
ap.add_argument("--lora-rank", type=int, default=16)
ap.add_argument("--lora-alpha", type=int, default=16)
args = ap.parse_args()

# House convention, e.g. iRanadheer/CARDS-Wind-LFM2.5-8B-A1B: STUDY-BaseModel,
# base model spelled exactly as upstream. RECoT is the headline arm and carries
# no tag; only the control is marked.
model_short = args.base_model.split("/")[-1]          # e.g. Qwen3.5-4B
base_name = f"FLICC-{model_short}"                     # e.g. FLICC-Qwen3.5-4B
variant = args.variant or (base_name if args.recot else f"{base_name}-norecot")

LORA_REPO = f"{HF_USERNAME}/{variant}-lora"
MERGED_REPO = f"{HF_USERNAME}/{variant}"
OUTPUT_DIR = variant
data_folder = "recot" if args.recot else "labels"

print("=" * 60)
print(f"  Variant:     {variant}")
print(f"  Base model:  {args.base_model}")
print(f"  RECoT:       {args.recot}  (dataset folder: {data_folder}/)")
print(f"  Dataset:     {FLICC_REPO}")
print(f"  Max seq:     {args.max_seq}")
print(f"  Push merged: {args.push_merged}")
print("=" * 60)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
subprocess.run(["nvidia-smi"], check=False)

import torch
print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

from huggingface_hub import HfApi, login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

from unsloth import FastLanguageModel  # must precede trl/transformers/peft
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# 1. Load model + LoRA
# ---------------------------------------------------------------------------
print("\n[1/5] Loading base model ...")
t = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.base_model,
    max_seq_length=args.max_seq,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print(f"  loaded in {time.time() - t:.1f}s")


# ---------------------------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------------------------
print("\n[2/5] Loading dataset ...")
t = time.time()
train_ds = load_dataset(FLICC_REPO, data_files=f"{data_folder}/train.jsonl",
                        split="train", token=hf_token)
eval_ds = load_dataset(FLICC_REPO, data_files=f"{data_folder}/train_eval.jsonl",
                       split="train", token=hf_token)
print(f"  train: {len(train_ds)}, eval: {len(eval_ds)}")


def apply_template(examples):
    # enable_thinking matches the arm: True for RECoT (target contains
    # <think>...</think>), False for labels-only (target is YAML alone, so no
    # auto-injected <think> — train and inference stay aligned).
    out = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
            enable_thinking=args.recot,
        )
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        out.append(text)
    return {"text": out}


train_ds = train_ds.map(apply_template, batched=True, remove_columns=["messages"])
eval_ds = eval_ds.map(apply_template, batched=True, remove_columns=["messages"])
print(f"  ready in {time.time() - t:.1f}s")
print(f"  sample: {train_ds[0]['text'][:200]}...")


# ---------------------------------------------------------------------------
# 3. Configure trainer
# ---------------------------------------------------------------------------
print("\n[3/5] Configuring trainer ...")
# Multimodal Qwen ships a Processor rather than a Tokenizer; unwrap to the inner
# text tokenizer for fields TRL reads directly, and pass the real EOS to bypass
# TRL's "<EOS_TOKEN>" sentinel default.
text_tok = tokenizer if hasattr(tokenizer, "encode") else tokenizer.tokenizer

config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    push_to_hub=True,
    hub_model_id=LORA_REPO,
    hub_private_repo=False,
    # Push the full latest checkpoint (optimizer/scheduler/RNG state) so an
    # interrupted job resumes exactly on a fresh host.
    hub_strategy="checkpoint",
    eos_token=text_tok.eos_token,

    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    max_length=args.max_seq,

    logging_steps=5,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=None,   # keep every local checkpoint; nothing pruned before upload

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
    run_name=variant,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=config,
)


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
print("\n[4/5] Training ...")
t = time.time()
result = trainer.train()
print(f"  done in {(time.time() - t) / 60:.1f} min")
if (loss := result.metrics.get("train_loss")):
    print(f"  final train loss: {loss:.4f}")

# NO final trainer.evaluate() here. The trainer already evaluates every
# eval_steps, and load_best_model_at_end has just swapped the best checkpoint
# back in — evaluating again after that weight swap killed the 9B run on
# Blackwell with "CUDA error: an illegal memory access was encountered", after
# 606/606 steps had completed cleanly. The number it printed was redundant with
# the eval_loss already in the training log, and the abort cost the merge step.
# Do not reintroduce it, and never call torch.cuda.synchronize() in a handler
# for a CUDA fault: the context is already poisoned, so it converts a survivable
# error into a process abort.


# ---------------------------------------------------------------------------
# 5. Push LoRA (always); merge + push (optional)
# ---------------------------------------------------------------------------
print(f"\n[5/5] Pushing LoRA -> {LORA_REPO} ...")
api = HfApi()
try:
    trainer.push_to_hub()
except Exception as e:
    print(f"  push_to_hub failed ({e}); falling back to manual upload")
    trainer.save_model(OUTPUT_DIR)
    api.create_repo(LORA_REPO, private=False, exist_ok=True)
    api.upload_folder(folder_path=OUTPUT_DIR, repo_id=LORA_REPO, repo_type="model",
                      commit_message=f"Upload {variant} LoRA adapter")
print(f"  https://huggingface.co/{LORA_REPO}")

if args.push_merged:
    print(f"\nMerging + pushing full model -> {MERGED_REPO} ...")
    merged_dir = f"{OUTPUT_DIR}_merged"
    try:
        model.save_pretrained_merged(merged_dir, tokenizer)
        api.create_repo(MERGED_REPO, private=False, exist_ok=True)
        api.upload_folder(folder_path=merged_dir, repo_id=MERGED_REPO, repo_type="model",
                          commit_message=f"Upload merged {variant} model")
        print(f"  https://huggingface.co/{MERGED_REPO}")
    except Exception as e:
        print(f"  merge+push failed: {e}")

print(f"\nDone — {variant}")
print(f"Evaluate with: python infer.py --backend vllm --model {MERGED_REPO} "
      f"--prompt deconstruct-revised --input data/test/test.jsonl")
