# /// script
# requires-python = ">=3.10"
# dependencies = [
#   # Fully pinned, exact released versions — all verified to satisfy
#   # unsloth==2026.5.9's own constraints (checked against its PyPI requires-dist):
#   #     torch>=2.4.0,<2.11.0   trl>=0.18.2,!=0.19.0,<=0.24.0   transformers<=5.5.0,>=4.51.3
#   # Picks (newest in-range of each):
#   #  - torch==2.10.0: plain-PyPI wheel bundles CUDA (no extra-index) AND pins
#   #    triton==3.6.0, which HAS tl.make_tensor_descriptor -> Unsloth MoE grouped_gemm
#   #    kernel works with NO manual triton pin. (torch 2.12 is OUT of unsloth's range.)
#   #  - transformers==5.5.0: v5 required for qwen3_5_moe; 5.9 is OUT of unsloth's <=5.5 cap.
#   #  - trl==0.22.2: Unsloth's recommended pin; 1.5.x is OUT of range.
#   #  - flash-linear-attention deliberately OMITTED: transformers' qwen3_5_moe GatedDeltaNet
#   #    falls back to torch without it. Installing FLA triggered the
#   #    triton>=3.4 -> tilelang -> "No CUDA in container" crash chain. No FLA => no tilelang.
#   # Pin ONLY torch (the one that matters): torch==2.10.0 bundles CUDA from plain
#   # PyPI (no extra-index) AND ships triton 3.6 (has make_tensor_descriptor -> MoE
#   # kernel works), and is inside unsloth's torch<2.11 cap. Everything else is left
#   # UNPINNED so uv resolves it within unsloth==2026.5.9's own ranges automatically
#   # (trl, datasets, transformers caps live in unsloth's metadata) — no more guessing.
#   # transformers>=5.2 only sets a floor to force v5 (required for qwen3_5_moe).
#   # flash-linear-attention is omitted on purpose (avoids the tilelang/no-CUDA crash).
#   "unsloth==2026.5.9",
#   "torch==2.10.0",
#   "torchvision==0.25.0",
#   "transformers>=5.2.0",
#   "huggingface_hub[hf_transfer]",
#   "tensorboard",
# ]
# ///
"""
CARDS / CARDS+Wind SFT trainer for Qwen3.5 **MoE** backbones (A3B / A10B).

  uv run train_moe.py --base-model Qwen/Qwen3.5-35B-A3B
  uv run train_moe.py --base-model Qwen/Qwen3.5-35B-A3B --joint
  uv run train_moe.py --base-model Qwen/Qwen3.5-35B-A3B --no-recot

Why a separate file from train.py (the dense trainer)? MoE models need a
different load + LoRA path that would otherwise clutter the working dense path:

  1. FastModel (not FastLanguageModel) — required for MoE.
  2. Expert FFNs are FUSED tensors (mlp.experts.gate_up_proj / .down_proj),
     so the LoRA target_modules differ: "gate_up_proj"/"down_proj" instead of
     the dense "gate_proj"/"up_proj"/"down_proj". The dense names SILENTLY fail
     to attach LoRA to experts (unslothai/unsloth#4907 — only ~0.2% trainable),
     producing a near-worthless run. We assert trainable% as a guard.
  3. bf16 LoRA only — MoE 4-bit QLoRA is unsupported (bitsandbytes limitation).
  4. lora_alpha defaults to 2*r (MoE convention).

Everything else — RECoT/no-RECoT data, joint CARDS+Wind concat, variant naming,
LoRA push + merge → BF16 push — matches train.py so downstream (quantize.py,
infer.py, reports) is unchanged.

Runs on HF Jobs (h200 flavor recommended; bf16 LoRA for 35B-A3B needs ~74 GB).
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
CARDS_REPO = os.environ.get("CARDS_REPO", "C3DS/cards_sft_dataset")  # dataset lives under C3DS
WIND_REPO = f"{HF_USERNAME}/wind-opposition-sft"

ap = argparse.ArgumentParser(description="CARDS / CARDS+Wind MoE SFT trainer")
ap.add_argument("--base-model", required=True,
                help="HF MoE model id, e.g. Qwen/Qwen3.5-35B-A3B")
ap.add_argument("--joint", action="store_true",
                help="Joint CARDS + Wind training (default: CARDS only)")
ap.add_argument("--recot", dest="recot", action="store_true", default=True)
ap.add_argument("--no-recot", dest="recot", action="store_false")
ap.add_argument("--max-seq", type=int, default=4096,
                help="Max sequence length. CARDS rows are ~2k tokens; 4096 is "
                     "ample and far cheaper than the dense default of 8192.")
ap.add_argument("--lora-only", dest="push_merged", action="store_false", default=True,
                help="Skip merging + pushing the full BF16 model (LoRA adapter only)")
ap.add_argument("--variant", default=None,
                help="Override variant name (default: derived from model + flags)")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--lora-rank", type=int, default=16,
                help="LoRA rank r.")
ap.add_argument("--lora-alpha", type=int, default=None,
                help="LoRA alpha. Default: 2*rank (MoE convention).")
ap.add_argument("--optim", default="adamw_8bit",
                help="Optimizer. Use adamw_torch if adamw_8bit errors on the "
                     "target CUDA stack (reported on some newer containers).")
ap.add_argument("--min-trainable-pct", type=float, default=0.5,
                help="Abort if trainable params < this %% of total — guards "
                     "against LoRA silently missing the fused MoE experts.")
ap.add_argument("--moe-backend", default="grouped_mm",
                choices=["grouped_mm", "unsloth_triton", "native_torch"],
                help="MoE expert compute backend (UNSLOTH_MOE_BACKEND). grouped_mm: "
                     "torch._grouped_mm, default, fast on H100+ (needs triton>=3.4, "
                     "satisfied by torch==2.10.0's bundled triton 3.6). native_torch: "
                     "pure-torch for-loop, ~12x slower but kernel-free fallback if the "
                     "Triton compile ever fails.")
args = ap.parse_args()

# Select MoE backend BEFORE importing unsloth (read at import time).
os.environ["UNSLOTH_MOE_BACKEND"] = args.moe_backend

lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank * 2


# ---------------------------------------------------------------------------
# Variant naming + dataset selection (identical scheme to train.py)
# ---------------------------------------------------------------------------
model_short = args.base_model.split("/")[-1]                      # e.g. Qwen3.5-35B-A3B

if args.variant:
    variant = args.variant
elif args.joint:
    variant = f"CARDS-Wind-{model_short}"                          # e.g. CARDS-Wind-Qwen3.5-35B-A3B
else:
    recot_tag = "recot" if args.recot else "norecot"
    variant = f"cards_{model_short.lower().replace('-', '_').replace('.', '_')}_{recot_tag}"

LORA_REPO = f"{HF_USERNAME}/{variant}-lora"
MERGED_REPO = f"{HF_USERNAME}/{variant}"
OUTPUT_DIR = variant

cards_train_file = "cards_train.jsonl" if args.recot else "cards_train_norecot.jsonl"
cards_eval_file = "cards_train_eval.jsonl" if args.recot else "cards_train_eval_norecot.jsonl"


print("=" * 60)
print(f"  Variant:    {variant}")
print(f"  Base model: {args.base_model}  (MoE path)")
print(f"  Joint:      {args.joint}  RECoT: {args.recot}")
print(f"  LoRA:       r={args.lora_rank} alpha={lora_alpha} lr={args.lr} epochs={args.epochs}")
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

from unsloth import FastModel  # MoE path: FastModel, not FastLanguageModel
from datasets import concatenate_datasets, load_dataset
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# 1. Load model + LoRA  (MoE-specific)
# ---------------------------------------------------------------------------
print("\n[1/5] Loading base MoE model ...")
t = time.time()

model, tokenizer = FastModel.from_pretrained(
    model_name=args.base_model,
    max_seq_length=args.max_seq,
    load_in_4bit=False,   # MoE 4-bit QLoRA unsupported (bitsandbytes limitation)
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

# Fused-expert LoRA targets. "gate_up_proj"/"down_proj" hit the MoE experts;
# "gate_proj"/"up_proj" (dense names) would silently miss them (#4907).
model = FastModel.get_peft_model(
    model,
    r=args.lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# --- Guard: did LoRA actually attach to the fused experts? -----------------
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
pct = 100.0 * trainable / total
print(f"  trainable params: {trainable:,} / {total:,} ({pct:.3f}%)")
if pct < args.min_trainable_pct:
    raise SystemExit(
        f"FATAL: only {pct:.3f}% trainable (< {args.min_trainable_pct}%). "
        f"LoRA likely did NOT attach to the fused MoE experts "
        f"(unslothai/unsloth#4907). Check target_modules / unsloth version "
        f"before spending GPU hours."
    )
print(f"  loaded in {time.time() - t:.1f}s")


# ---------------------------------------------------------------------------
# 2. Load dataset(s)   (identical to train.py)
# ---------------------------------------------------------------------------
print("\n[2/5] Loading dataset ...")
t = time.time()

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
# 3. Configure trainer   (identical to train.py)
# ---------------------------------------------------------------------------
print("\n[3/5] Configuring trainer ...")

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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=args.lr,
    max_length=args.max_seq,

    logging_steps=5,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,   # 35B checkpoints are large; keep fewer than dense

    eval_strategy="steps",
    eval_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    warmup_steps=10,
    lr_scheduler_type="cosine",
    optim=args.optim,
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
    print("  NOTE: merging fused-expert MoE LoRA is the least-proven step for "
          "this architecture; if it fails, the -lora adapter above is still saved.")
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
