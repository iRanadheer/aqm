# /// script
# requires-python = ">=3.10"
# dependencies = [
#   # Same pins as ft/train.py — the merge has to load the base model the same
#   # way the adapter was trained against.
#   "unsloth==2026.5.9",
#   "torch==2.10.0",
#   "torchvision==0.25.0",
#   "transformers>=5.2.0",
#   "peft",
#   "huggingface_hub[hf_transfer]",
# ]
# ///
"""Merge a trained LoRA adapter into its base model and push the result.

    uv run ft/merge_push.py --base-model Qwen/Qwen3.5-9B --variant FLICC-Qwen3.5-9B

Recovers the merged model when training finished but the merge did not. That is
not hypothetical: the 9B run completed all 606 steps (train loss 0.1861), then
the redundant post-training trainer.evaluate() raised a CUDA illegal memory
access on Blackwell and aborted the process before step 5. The adapter survived
because hub_strategy="checkpoint" had been pushing throughout; only the merge
was lost. train.py no longer does that final evaluate, so this script exists to
rescue runs made before the fix.

Reads  <hf_user>/<variant>-lora   (adapter, already on the Hub)
Writes <hf_user>/<variant>        (merged BF16 model)

No training, no dataset, no optimizer — just load, merge, push. Minutes, not
hours, and it needs a GPU only because Unsloth's loader wants one.
"""

import argparse
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

HF_USERNAME = os.environ.get("HF_USERNAME", "iRanadheer")

ap = argparse.ArgumentParser(description="Merge a LoRA adapter and push the full model")
ap.add_argument("--base-model", required=True, help="e.g. Qwen/Qwen3.5-9B")
ap.add_argument("--variant", required=True, help="e.g. FLICC-Qwen3.5-9B")
ap.add_argument("--adapter", default=None, help="Default: <hf_user>/<variant>-lora")
ap.add_argument("--max-seq", type=int, default=8192)
args = ap.parse_args()

ADAPTER_REPO = args.adapter or f"{HF_USERNAME}/{args.variant}-lora"
MERGED_REPO = f"{HF_USERNAME}/{args.variant}"

print("=" * 60)
print(f"  base    : {args.base_model}")
print(f"  adapter : {ADAPTER_REPO}")
print(f"  merged  : {MERGED_REPO}")
print("=" * 60)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from huggingface_hub import HfApi, login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

from unsloth import FastLanguageModel

# Load the ADAPTER repo directly — Unsloth resolves the base from its
# adapter_config.json and returns a model that carries save_pretrained_merged.
#
# Do NOT do this instead:
#     model, tok = FastLanguageModel.from_pretrained(base)
#     model = PeftModel.from_pretrained(model, adapter)   # plain peft
#     model.merge_and_unload().save_pretrained(dir)
# That is what the first version of this script did, and it produced a 9B whose
# every weight was prefixed `language_model.` — Unsloth wraps Qwen3.5 with that
# submodule, and plain save_pretrained preserves the wrapper. vLLM then refuses
# it outright:
#     ValueError: There is no module or parameter named 'language_model'
#                 in Qwen3_5Model
# save_pretrained_merged strips the wrapper and shards the output, matching what
# ft/train.py writes for every other size.
print(f"\n[1/2] Loading {ADAPTER_REPO} (base resolved from adapter_config) ...")
t = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_REPO,
    max_seq_length=args.max_seq,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)
print(f"  loaded in {time.time() - t:.1f}s")

print(f"\n[2/2] Merging and pushing -> {MERGED_REPO} ...")
t = time.time()
merged_dir = f"{args.variant}_merged"
model.save_pretrained_merged(merged_dir, tokenizer)
api = HfApi(token=hf_token)
api.create_repo(MERGED_REPO, private=False, exist_ok=True)
api.upload_folder(folder_path=merged_dir, repo_id=MERGED_REPO, repo_type="model",
                  commit_message=f"Merged {args.variant} (adapter from {ADAPTER_REPO})")
print(f"  done in {(time.time() - t) / 60:.1f} min")
print(f"  https://huggingface.co/{MERGED_REPO}")
print("Verify vLLM can load it before evaluating — a bad merge only shows up at serve time.")
