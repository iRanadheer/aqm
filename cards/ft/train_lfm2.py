# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth",
#   "datasets",
#   "trl>=0.12.0",
#   "transformers>=5.9.0",  # lfm2_moe arch ships in 5.9+
#   "huggingface_hub[hf_transfer]",
#   "tensorboard",
# ]
#
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# index-strategy = "unsafe-best-match"
# override-dependencies = ["transformers>=5.9.0"]
# ///
"""
LFM2.5-8B-A1B SFT trainer — combined API+chat dataset, ChatML thinking.

  uv run train_lfm2.py                                   # LFM2.5-8B-A1B

Why a separate script (vs ft/train_gemma.py / ft/train.py):
  * LiquidAI/LFM2.5-8B-A1B is a HYBRID MoE (`lfm2_moe`): 8.3B total / 1.5B
    active, 32 experts (4/tok), first 2 layers dense, 18 short-conv (LIV) +
    6 GQA attention layers. Loads via unsloth FastModel.
  * MoE LoRA targets (verified against the safetensors module names):
        q_proj, k_proj, v_proj, out_proj   (attention)
        in_proj                            (LIV conv mixing)
        w1, w2, w3                         (expert FFNs + the 2 dense FFNs)
    The router/gate (`feed_forward.gate`) is deliberately LEFT OUT — Liquid's
    own guidance: "it's not a good idea to fine-tune the router layer", and
    unsloth freezes it by default. Targeting w1/w2/w3 engages unsloth's MoE
    Split-LoRA across all 32 experts.
  * Reasoning is LFM2-native: an inline <think>{reasoning}</think> block in the
    ChatML assistant turn (the model's own thinking format), then the visible
    YAML. We hand-build the target string (like train_gemma.py) so the thinking
    is guaranteed present in the SFT target; a render guard asserts it.
  * Liquid-recommended HPs: lr 2e-4, lora_alpha 32 (= 2x rank).

Target format (one [system, user, assistant] row):

  <|im_start|>system\n{system}<|im_end|>\n
  <|im_start|>user\n{user}<|im_end|>\n
  <|im_start|>assistant\n<think>\n{reasoning}\n</think>{yaml}<|im_end|>\n

Inference: run with the model's chat template (thinking on); parse the visible
YAML after </think>.

Reuses iRanadheer/cards-wind-qwen-chat gemma4/*.jsonl — the `messages` rows are
model-agnostic (cards+wind, API+chat combined, think-blocks repaired).
Pushes a LoRA adapter, the merged BF16 model, and (best-effort) a Q4_0 GGUF for
llama.cpp serving on the A2 cards.
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
COMBINED_REPO = "iRanadheer/cards-wind-qwen-chat"

# Verified leaf module names (safetensors header of LFM2.5-8B-A1B). `gate` is the
# router and is intentionally absent → frozen.
LFM2_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]

ap = argparse.ArgumentParser(description="LFM2.5-8B-A1B combined SFT trainer (ChatML thinking)")
ap.add_argument("--base-model", default="LiquidAI/LFM2.5-8B-A1B",
                help="LFM2 HF model id (default: LiquidAI/LFM2.5-8B-A1B)")
ap.add_argument("--max-seq", type=int, default=8192,
                help="8192 (consistent with the Gemma runs); max observed row 5209 tok.")
ap.add_argument("--lora-only", dest="push_merged", action="store_false", default=True,
                help="Skip merging + pushing the full BF16 model (LoRA adapter only)")
ap.add_argument("--gguf", action="store_true", default=False,
                help="Also export a best-effort Q4_0 GGUF (separate llama.cpp serving step).")
ap.add_argument("--variant", default=None,
                help="Override variant name (default: CARDS-Wind-LFM2.5-8B-A1B)")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4, help="Liquid-recommended LFM2 LR.")
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--grad-accum", type=int, default=2)
ap.add_argument("--lora-rank", type=int, default=16)
ap.add_argument("--lora-alpha", type=int, default=32,
                help="Liquid recipe: alpha = 2x rank.")
ap.add_argument("--max-steps", type=int, default=-1,
                help="Cap steps for a cheap dry-run (e.g. 5) before the full run.")
args = ap.parse_args()

if "lfm2" not in args.base_model.lower():
    ap.error("this script is LFM2-specific; use ft/train_gemma.py or ft/train.py for other models")


# ---------------------------------------------------------------------------
# Variant naming
# ---------------------------------------------------------------------------
model_short = args.base_model.split("/")[-1]                  # e.g. LFM2.5-8B-A1B
variant = args.variant or f"CARDS-Wind-{model_short}"

LORA_REPO = f"{HF_USERNAME}/{variant}-lora"
MERGED_REPO = f"{HF_USERNAME}/{variant}"
OUTPUT_DIR = variant

print("=" * 60)
print(f"  Variant:    {variant}")
print(f"  Base model: {args.base_model}")
print(f"  Dataset:    {COMBINED_REPO} (combined API+chat)")
print(f"  Max seq:    {args.max_seq}  Epochs: {args.epochs}  LR: {args.lr}")
print(f"  Batch:      {args.batch_size} x accum {args.grad_accum}")
print(f"  LoRA:       r={args.lora_rank} alpha={args.lora_alpha}  targets={LFM2_TARGETS}")
print(f"  Router:     gate left untargeted (frozen)")
print(f"  Push merged:{args.push_merged}  GGUF:{args.gguf}")
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

from unsloth import FastModel  # must precede trl/transformers/peft
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset


# ---------------------------------------------------------------------------
# 1. Load model + LoRA  (FastModel — lfm2_moe arch)
# ---------------------------------------------------------------------------
print("\n[1/5] Loading base model ...")
t = time.time()

model, tokenizer = FastModel.from_pretrained(
    model_name=args.base_model,
    max_seq_length=args.max_seq,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)
# Explicit target_modules (Liquid's documented way) — router/gate omitted so it
# stays frozen; w1/w2/w3 engage unsloth MoE Split-LoRA across the 32 experts.
model = FastModel.get_peft_model(
    model,
    target_modules=LFM2_TARGETS,
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print(f"  loaded in {time.time() - t:.1f}s")


# ---------------------------------------------------------------------------
# 2. Load combined dataset + ChatML thinking render
# ---------------------------------------------------------------------------
print("\n[2/5] Loading dataset ...")
t = time.time()

# gemma4/ = cards+wind, API+chat combined, think-blocks repaired. `messages` are
# model-agnostic, so they're reused verbatim for LFM2.
train_ds = load_dataset(COMBINED_REPO, data_files="gemma4/train.jsonl",
                        split="train", token=hf_token).shuffle(seed=42)
eval_ds = load_dataset(COMBINED_REPO, data_files="gemma4/train_eval.jsonl",
                       split="train", token=hf_token).shuffle(seed=42)
print(f"  train: {len(train_ds)}, eval: {len(eval_ds)}")


def render_lfm2_chatml(msgs):
    """ChatML render of one [system, user, assistant] row, keeping the assistant's
    inline <think>...</think> block (LFM2's native thinking format) in the target.

    All markers are verified atomic single tokens in the LFM2.5 vocab:
        <|im_start|>=124899  <|im_end|>=124900 (eos)  <|startoftext|>=124894 (bos)
        <think>=124901  </think>=124902
    So the inline <think>/<\/think> already in the data ARE the model's native
    reasoning delimiters — no token rewriting needed (unlike Gemma's channels).
    No <bos> emitted here — add_bos_token is False, so the probe below prepends it.
    """
    sys_c, user_c, asst_c = (m["content"] for m in msgs)
    if "<think>" not in asst_c or "</think>" not in asst_c:
        raise ValueError("assistant target has no <think> block — combined data is RECoT-only")
    return (
        "<|im_start|>system\n" + sys_c.strip() + "<|im_end|>\n"
        "<|im_start|>user\n" + user_c.strip() + "<|im_end|>\n"
        "<|im_start|>assistant\n" + asst_c.strip() + "<|im_end|>\n"
    )


def apply_template(examples):
    return {"text": [render_lfm2_chatml(msgs) for msgs in examples["messages"]]}


train_ds = train_ds.map(apply_template, batched=True, remove_columns=["messages"])
eval_ds = eval_ds.map(apply_template, batched=True, remove_columns=["messages"])
print(f"  ready in {time.time() - t:.1f}s")

# Render guard: refuse to train if reasoning vanished from the target.
sample = train_ds[0]["text"]
assert "<think>" in sample and "</think>" in sample, "thought block missing"
assert len(sample.split("<think>", 1)[1].split("</think>", 1)[0]) > 50, \
    "reasoning stripped from SFT target"
assert "```yaml" in sample.split("</think>", 1)[1], "visible YAML answer missing"
print(f"  render guard OK; sample: {sample[:160]!r}...")

# BOS probe: verify the tokenizer auto-adds <bos>; prepend ourselves if not.
_text_tok = tokenizer if hasattr(tokenizer, "encode") else tokenizer.tokenizer
_probe = _text_tok("probe", add_special_tokens=True).input_ids
if _text_tok.bos_token_id is not None and (not _probe or _probe[0] != _text_tok.bos_token_id):
    print("  tokenizer does NOT auto-add <bos> — prepending it to every text")
    bos = _text_tok.bos_token
    train_ds = train_ds.map(lambda ex: {"text": [bos + t for t in ex["text"]]}, batched=True)
    eval_ds = eval_ds.map(lambda ex: {"text": [bos + t for t in ex["text"]]}, batched=True)
else:
    print("  tokenizer auto-adds <bos> — texts stay bos-free")


# ---------------------------------------------------------------------------
# 3. Configure trainer
# ---------------------------------------------------------------------------
print("\n[3/5] Configuring trainer ...")

config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    push_to_hub=True,
    hub_model_id=LORA_REPO,
    hub_private_repo=False,
    hub_strategy="checkpoint",
    eos_token="<|im_end|>",

    num_train_epochs=args.epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    max_length=args.max_seq,

    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=4,

    eval_strategy="steps",
    eval_steps=50,
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

# Loss only on the assistant turn — the codebook system prompt (~85% of tokens,
# identical every row) and user text are context, not targets.
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
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
# 5. Push LoRA (always); merge + push (optional); GGUF (best-effort)
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

if args.gguf:
    print(f"\nExporting Q4_0 GGUF (for llama.cpp / A2 serving) ...")
    try:
        model.save_pretrained_gguf(f"{OUTPUT_DIR}_gguf", tokenizer,
                                   quantization_method="q4_0")
        api.create_repo(f"{MERGED_REPO}-GGUF", private=False, exist_ok=True)
        api.upload_folder(folder_path=f"{OUTPUT_DIR}_gguf", repo_id=f"{MERGED_REPO}-GGUF",
                          repo_type="model", commit_message=f"Q4_0 GGUF {variant}")
        print(f"  https://huggingface.co/{MERGED_REPO}-GGUF")
    except Exception as e:
        print(f"  GGUF export failed ({e}); convert separately from the merged repo")

print(f"\nDone — {variant}")
