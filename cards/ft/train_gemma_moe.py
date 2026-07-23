# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth",
#   "datasets",
#   "trl>=0.12.0",
#   "transformers>=5.10.2",  # gemma4 arch
#   "huggingface_hub[hf_transfer]",
#   "tensorboard",
# ]
#
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# index-strategy = "unsafe-best-match"
# override-dependencies = ["transformers>=5.10.2"]
# ///
"""
Gemma 4 MoE SFT trainer — for google/gemma-4-26B-A4B-it (and the 31B/12B dense).

  uv run train_gemma_moe.py                                   # gemma-4-26B-A4B-it
  uv run train_gemma_moe.py --base-model google/gemma-4-31B-it
  uv run train_gemma_moe.py --max-steps 5 --lora-only         # cheap dry-run

Built for running directly on a GPU box (Vast.ai H200): set HF_TOKEN in the env,
`uv run` this file. Pushes a LoRA adapter + merged BF16 model to the Hub.

Why this vs ft/train_gemma.py:
  * gemma-4-26B-A4B is a MULTIMODAL MoE (Gemma4ForConditionalGeneration):
      - vision_tower.* / embed_vision.*  -> FROZEN (finetune_vision_layers=False)
      - self_attn.{q,k,v,o}_proj         -> adapted (attention)
      - mlp.{gate,up,down}_proj          -> adapted (shared/dense expert)
      - experts.{gate_up,down}_proj      -> adapted (128 routed experts, grouped;
                                            unsloth MoE Split-LoRA handles these)
      - router.{proj,scale,per_expert_scale} -> FROZEN (never LoRA the router)
    The finetune_* flags below are unsloth's gemma4 selector; finetune_mlp_modules
    drives both the shared MLP and the routed experts via the gemma4 MoE path,
    and leaves the router untouched. (Verify with the trainable-params print +
    a --max-steps 5 dry-run before committing the full run.)
  * eval_steps bumped 25 -> 150 (the 12B run wasted ~3.5h on over-frequent eval).

RECoT reasoning uses Gemma's NATIVE thought channel (identical render to
ft/train_gemma.py); inference must run enable_thinking=True and parse after
<channel|>. Trains on iRanadheer/cards-wind-qwen-chat gemma4/*.jsonl.
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

ap = argparse.ArgumentParser(description="Gemma 4 MoE combined SFT trainer (native thinking channel)")
ap.add_argument("--base-model", default="google/gemma-4-26B-A4B-it",
                help="Gemma 4 HF model id (default: google/gemma-4-26B-A4B-it)")
ap.add_argument("--max-seq", type=int, default=8192)
ap.add_argument("--lora-only", dest="push_merged", action="store_false", default=True,
                help="Skip merging + pushing the full BF16 model (LoRA adapter only)")
ap.add_argument("--variant", default=None,
                help="Override variant name (default: CARDS-Wind-Gemma4-<size>)")
ap.add_argument("--epochs", type=int, default=2,
                help="2 is the sweet spot — epoch 3 plateaued on the LFM2/12B runs.")
ap.add_argument("--lr", type=float, default=1e-4,
                help="Gemma bases are strong; gentle LR (1e-4) per the 12B/31B recipe.")
ap.add_argument("--batch-size", type=int, default=2)
ap.add_argument("--grad-accum", type=int, default=4)
ap.add_argument("--lora-rank", type=int, default=16)
ap.add_argument("--lora-alpha", type=int, default=16,
                help="Lower than rank = gentler perturbation of a strong base.")
ap.add_argument("--max-steps", type=int, default=-1,
                help="Cap steps for a cheap dry-run (e.g. 5) before the full run.")
args = ap.parse_args()

if "gemma" not in args.base_model.lower():
    ap.error("this script is Gemma-4-specific; use ft/train_lfm2.py or ft/train.py for other models")


# ---------------------------------------------------------------------------
# Variant naming
# ---------------------------------------------------------------------------
model_short = args.base_model.split("/")[-1]                  # gemma-4-26B-A4B-it
if args.variant:
    variant = args.variant
else:
    size = model_short.replace("gemma-4-", "").replace("-it", "")   # 26B-A4B
    variant = f"CARDS-Wind-Gemma4-{size}"

LORA_REPO = f"{HF_USERNAME}/{variant}-lora"
MERGED_REPO = f"{HF_USERNAME}/{variant}"
OUTPUT_DIR = variant

print("=" * 60)
print(f"  Variant:    {variant}")
print(f"  Base model: {args.base_model}")
print(f"  Dataset:    {COMBINED_REPO} (combined API+chat)")
print(f"  Max seq:    {args.max_seq}  Epochs: {args.epochs}  LR: {args.lr}")
print(f"  Batch:      {args.batch_size} x accum {args.grad_accum}")
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

from unsloth import FastModel  # must precede trl/transformers/peft
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# 1. Load model + LoRA  (FastModel — gemma4 MoE arch)
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
# finetune_* flags = unsloth's gemma4 layer selector. Vision tower frozen;
# attention + MLP (shared expert AND the 128 routed experts via the gemma4 MoE
# Split-LoRA path) adapted; the router is left frozen by this selector.
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
# Print the LoRA coverage — on the dry-run, confirm the routed experts are
# adapted and `router.proj` is NOT in the trainable set.
try:
    model.print_trainable_parameters()
except Exception as e:
    print(f"  (print_trainable_parameters unavailable: {e})")
# Diagnostic: which base modules actually received LoRA. Confirm the routed
# experts ARE adapted and the router is NOT — the whole point of the MoE recipe.
_lora = [n for n, _ in model.named_parameters() if ".lora_" in n]
_leaves = sorted({n.split(".lora_")[0].split(".")[-1] for n in _lora})
print(f"  LoRA-adapted leaf modules: {_leaves}")
print(f"  experts adapted: {any('experts' in n for n in _lora)} (want True)")
print(f"  router adapted:  {any('router' in n for n in _lora)} (want False)")
print(f"  vision adapted:  {any('vision' in n for n in _lora)} (want False)")
print(f"  loaded in {time.time() - t:.1f}s")


# ---------------------------------------------------------------------------
# 2. Load combined dataset + native-channel render
# ---------------------------------------------------------------------------
print("\n[2/5] Loading dataset ...")
t = time.time()

train_ds = load_dataset(COMBINED_REPO, data_files="gemma4/train.jsonl",
                        split="train", token=hf_token).shuffle(seed=42)
eval_ds = load_dataset(COMBINED_REPO, data_files="gemma4/train_eval.jsonl",
                       split="train", token=hf_token).shuffle(seed=42)
print(f"  train: {len(train_ds)}, eval: {len(eval_ds)}")


def render_gemma_native(msgs):
    """Hand-built Gemma 4 thinking-mode render of one [system, user, assistant] row.
    Reasoning -> native thought channel; YAML after it is the visible answer.
    Markers are verified single special tokens (<|turn|>=105, <turn|>=106,
    <|channel>=100, <channel|>=101, <|think|>=98)."""
    sys_c, user_c, asst_c = (m["content"] for m in msgs)
    if "</think>" not in asst_c:
        raise ValueError("assistant target has no <think> block — combined data is RECoT-only")
    reasoning = asst_c.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    visible = asst_c.split("</think>", 1)[1].strip()
    return (
        "<|turn>system\n<|think|>\n" + sys_c.strip() + "<turn|>\n"
        "<|turn>user\n" + user_c.strip() + "<turn|>\n"
        "<|turn>model\n<|channel>thought\n" + reasoning + "\n<channel|>" + visible + "<turn|>\n"
    )


def apply_template(examples):
    return {"text": [render_gemma_native(msgs) for msgs in examples["messages"]]}


train_ds = train_ds.map(apply_template, batched=True, remove_columns=["messages"])
eval_ds = eval_ds.map(apply_template, batched=True, remove_columns=["messages"])
print(f"  ready in {time.time() - t:.1f}s")

# Render guard
sample = train_ds[0]["text"]
assert "<|channel>thought\n" in sample and "<channel|>" in sample, "thought channel missing"
assert len(sample.split("<|channel>thought\n", 1)[1].split("<channel|>", 1)[0]) > 50, \
    "reasoning stripped from SFT target"
assert "```yaml" in sample.split("<channel|>", 1)[1], "visible YAML answer missing"
print(f"  render guard OK; sample: {sample[:160]!r}...")

# BOS probe
_text_tok = tokenizer if hasattr(tokenizer, "encode") else tokenizer.tokenizer
_probe = _text_tok("probe", add_special_tokens=True).input_ids
if _text_tok.bos_token_id is not None and _probe[0] != _text_tok.bos_token_id:
    print("  tokenizer does NOT auto-add <bos> — prepending it to every text")
    bos = _text_tok.bos_token
    train_ds = train_ds.map(lambda ex: {"text": [bos + t for t in ex["text"]]}, batched=True)
    eval_ds = eval_ds.map(lambda ex: {"text": [bos + t for t in ex["text"]]}, batched=True)
else:
    print("  tokenizer auto-adds <bos> — texts stay bos-free (unsloth convention)")


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
    eos_token="<turn|>",

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
    eval_steps=150,
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

# Loss only on the model turn (thought channel + YAML).
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
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
