# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=5.2.0",
#   "peft",
#   "accelerate",
#   "huggingface_hub[hf_transfer]",
# ]
#
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# index-strategy = "unsafe-best-match"
# ///
"""
Merge a pushed LoRA checkpoint into its base model and push the result.

Exists for runs cancelled after convergence: training pushes adapter
checkpoints to the Hub every save (hub_strategy=every_save), so the best
checkpoint can be merged without finishing the remaining epochs.

  uv run merge_lora.py \
      --base Qwen/Qwen3.6-27B \
      --lora C3DS/CARDS-Wind-Qwen3.6-27B-lora --revision <commit> \
      --dst C3DS/CARDS-Wind-Qwen3.6-27B

Pick the revision by mapping eval_loss logs to checkpoint commits
("Training in progress, step N").
"""

import argparse
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--revision", default=None, help="LoRA repo commit/branch (default: main)")
ap.add_argument("--dst", required=True)
ap.add_argument("--private", action="store_true")
args = ap.parse_args()

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[1/4] Loading base {args.base} (bf16, {device}) ...")
t = time.time()
model = AutoModelForCausalLM.from_pretrained(
    args.base, torch_dtype=torch.bfloat16, device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(args.base)
print(f"  {time.time()-t:.0f}s")

print(f"[2/4] Applying adapter {args.lora} @ {args.revision or 'main'} ...")
t = time.time()
model = PeftModel.from_pretrained(model, args.lora, revision=args.revision)
model = model.merge_and_unload()
print(f"  {time.time()-t:.0f}s")

out_dir = "merged"
print(f"[3/4] Saving to {out_dir}/ ...")
t = time.time()
model.save_pretrained(out_dir, safe_serialization=True)
tokenizer.save_pretrained(out_dir)
print(f"  {time.time()-t:.0f}s")

print(f"[4/4] Pushing -> {args.dst} ...")
t = time.time()
api = HfApi()
api.create_repo(args.dst, private=args.private, exist_ok=True)
api.upload_folder(
    folder_path=out_dir, repo_id=args.dst, repo_type="model",
    commit_message=f"Merge {args.lora}@{args.revision or 'main'} into {args.base}",
)
print(f"  {time.time()-t:.0f}s")
print(f"Done: https://huggingface.co/{args.dst}")
