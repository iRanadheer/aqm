# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth",
#   "datasets",
#   "trl>=0.12.0",
#   "transformers>=5.2.0",
#   "huggingface_hub[hf_transfer]",
#   "flash-linear-attention",
#   "triton<3.4",  # Hopper backward path is incorrect on >=3.4 (FLA #640); pre-3.4 is correct
#   "tensorboard",
# ]
#
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# index-strategy = "unsafe-best-match"
# ///
"""
CARDS SFT trainer — single entrypoint for every variant.

  uv run train.py --base-model Qwen/Qwen3.5-4B
  uv run train.py --base-model Qwen/Qwen3.5-4B --no-recot
  uv run train.py --base-model Qwen/Qwen3.6-27B --joint
  uv run train.py --base-model Qwen/Qwen3.6-27B --combined
  uv run train.py --base-model Qwen/Qwen3.5-4B --lora-only

Single-task: trains on iRanadheer/cards_sft_dataset only.
Joint:      adds iRanadheer/wind-opposition-sft (CARDS+Wind from the same backbone).
Combined:   trains on iRanadheer/cards-wind-qwen-chat — API + chat formats for
            both CARDS and Wind in one model; the system prompt selects the
            output format at inference. Produces CARDS-Wind-<model> (the
            API-only predecessors carry an -API suffix).

Pushes a LoRA adapter to <hf_user>/<variant>-lora AND merges + pushes the
full BF16 model to <hf_user>/<variant>. Pass --lora-only to skip the merge.

Runs identically on HF Jobs, Vast.ai, RunPod, or any cu128-capable GPU host.
"""

import argparse
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
HF_USERNAME = os.environ.get("HF_USERNAME", "iRanadheer")
CARDS_REPO = f"{HF_USERNAME}/cards_sft_dataset"
WIND_REPO = f"{HF_USERNAME}/wind-opposition-sft"
# Combined API+chat dataset lives under iRanadheer (public) regardless of the
# namespace the model is pushed to (e.g. HF_USERNAME=C3DS).
COMBINED_REPO = "iRanadheer/cards-wind-qwen-chat"

ap = argparse.ArgumentParser(description="CARDS / CARDS+Wind SFT trainer")
ap.add_argument("--base-model", required=True,
                help="HF model id, e.g. Qwen/Qwen3.5-4B or Qwen/Qwen3.6-27B")
ap.add_argument("--joint", action="store_true",
                help="Joint CARDS + Wind training (default: CARDS only)")
ap.add_argument("--combined", action="store_true",
                help="Train on the combined API+chat CARDS+Wind dataset "
                     f"({COMBINED_REPO}); one model serves both output formats.")
ap.add_argument("--recot", dest="recot", action="store_true", default=True)
ap.add_argument("--no-recot", dest="recot", action="store_false")
ap.add_argument("--max-seq", type=int, default=8192)
ap.add_argument("--lora-only", dest="push_merged", action="store_false", default=True,
                help="Skip merging + pushing the full BF16 model (LoRA adapter only)")
ap.add_argument("--variant", default=None,
                help="Override variant name (default: derived from model + flags)")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch-size", type=int, default=1,
                help="per_device_train_batch_size. Keep batch_size*grad_accum "
                     "constant to preserve the effective batch (recipe).")
ap.add_argument("--grad-accum", type=int, default=8,
                help="gradient_accumulation_steps.")
ap.add_argument("--lora-rank", type=int, default=16,
                help="LoRA rank r. Effective update scale is alpha/r.")
ap.add_argument("--lora-alpha", type=int, default=16,
                help="LoRA alpha. Lower = less perturbation of the base prior "
                     "(useful for strong-base scales where vanilla SFT regresses).")
args = ap.parse_args()

if args.combined and args.joint:
    ap.error("--combined and --joint are mutually exclusive")
if args.combined and not args.recot:
    ap.error("--combined data is RECoT-only (every row targets <think> + YAML); drop --no-recot")


# ---------------------------------------------------------------------------
# Variant naming + dataset selection
# ---------------------------------------------------------------------------
model_short = args.base_model.split("/")[-1]                      # e.g. Qwen3.5-4B

if args.variant:
    variant = args.variant
elif args.combined:
    variant = f"CARDS-Wind-{model_short}"                          # e.g. CARDS-Wind-Qwen3.6-27B
elif args.joint:
    variant = f"CARDS-Wind-{model_short}-API"                      # legacy API-only joint line
else:
    recot_tag = "recot" if args.recot else "norecot"
    variant = f"cards_{model_short.lower().replace('-', '_')}_{recot_tag}"

LORA_REPO = f"{HF_USERNAME}/{variant}-lora"
MERGED_REPO = f"{HF_USERNAME}/{variant}"
OUTPUT_DIR = variant

cards_train_file = "cards_train.jsonl" if args.recot else "cards_train_norecot.jsonl"
cards_eval_file = "cards_train_eval.jsonl" if args.recot else "cards_train_eval_norecot.jsonl"


print("=" * 60)
print(f"  Variant:    {variant}")
print(f"  Base model: {args.base_model}")
print(f"  Combined:   {args.combined}  Joint: {args.joint}  RECoT: {args.recot}")
print(f"  Max seq:    {args.max_seq}")
print(f"  Push merged:{args.push_merged}  (use --lora-only to skip)")
print("=" * 60)


# ---------------------------------------------------------------------------
# Environment + auth
# ---------------------------------------------------------------------------
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
from datasets import concatenate_datasets, load_dataset
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
# 2. Load dataset(s)
# ---------------------------------------------------------------------------
print("\n[2/5] Loading dataset ...")
t = time.time()

if args.combined:
    # cards/ and wind/ files are each already API+chat combined.
    parts = {
        split: [
            load_dataset(COMBINED_REPO, data_files=f"{folder}/{split}.jsonl",
                         split="train", token=hf_token).select_columns(["messages"])
            for folder in ("cards", "wind")
        ]
        for split in ("train", "train_eval")
    }
    train_ds = concatenate_datasets(parts["train"]).shuffle(seed=42)
    eval_ds = concatenate_datasets(parts["train_eval"]).shuffle(seed=42)
else:
    train_ds = load_dataset(CARDS_REPO, data_files=cards_train_file, split="train", token=hf_token)
    eval_ds = load_dataset(CARDS_REPO, data_files=cards_eval_file, split="train", token=hf_token)

    if args.joint:
        wind_train = load_dataset(WIND_REPO, data_files="train/train.jsonl", split="train", token=hf_token)
        wind_eval = load_dataset(WIND_REPO, data_files="train/train_eval.jsonl", split="train", token=hf_token)
        train_ds = concatenate_datasets([
            train_ds.select_columns(["messages"]),
            wind_train.select_columns(["messages"]),
        ]).shuffle(seed=42)
        eval_ds = concatenate_datasets([
            eval_ds.select_columns(["messages"]),
            wind_eval.select_columns(["messages"]),
        ]).shuffle(seed=42)

print(f"  train: {len(train_ds)}, eval: {len(eval_ds)}")


def apply_template(examples):
    # enable_thinking matches the variant: True for RECoT (target includes
    # <think>...</think>), False for no-RECoT (target is YAML only — no
    # auto-injected <think>\n means train and inference are aligned).
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

# Multimodal Qwen ships a Processor (not a Tokenizer); unwrap to the inner
# text tokenizer for fields TRL reads directly. TRL >=0.18 defaults
# SFTConfig.eos_token to a literal "<EOS_TOKEN>" sentinel that fails on
# any processor whose vocab doesn't contain that string — pass the real
# EOS explicitly to bypass it.
text_tok = tokenizer if hasattr(tokenizer, "encode") else tokenizer.tokenizer
eos_token = text_tok.eos_token

config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    push_to_hub=True,
    hub_model_id=LORA_REPO,
    hub_private_repo=False,
    eos_token=eos_token,

    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    max_length=args.max_seq,

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

try:
    eval_result = trainer.evaluate()
    if (loss := eval_result.get("eval_loss")):
        print(f"  final eval loss: {loss:.4f}")
except Exception as e:
    print(f"  eval failed: {e}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


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
