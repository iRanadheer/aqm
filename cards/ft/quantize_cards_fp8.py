# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git@main",
#     "transformers>=5.2.0",
#     "tokenizers>=0.21",
#     "torch",
#     "huggingface_hub[hf_transfer]",
#     "accelerate",
# ]
#
# [tool.uv]
# override-dependencies = [
#     "transformers>=5.2.0",
# ]
# ///

"""
CARDS Qwen3.5 FP8 Quantization (post-training, no retraining).

Quantizes all three merged CARDS models (4B, 9B, 27B) to FP8_DYNAMIC
using llm-compressor and uploads them (public) to the C3DS org.

Runs end-to-end on a single big GPU — designed for an H100/H200/B200
with ≥80 GB VRAM so the 27B fits without multi-GPU sharding.

  Source repos:  C3DS/CARDS-Qwen3.5-{4B,9B,27B}
  Output repos:  C3DS/CARDS-Qwen3.5-{4B,9B,27B}-FP8

Examples:
  uv run quantize_cards_fp8.py                 # all three, push public
  uv run quantize_cards_fp8.py --no-push       # local only
  uv run quantize_cards_fp8.py --only 27b      # single size
"""

import argparse
import gc
import os
import shutil
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODELS = {
    "4b":  "C3DS/CARDS-Qwen3.5-4B",
    "9b":  "C3DS/CARDS-Qwen3.5-9B",
    "27b": "C3DS/CARDS-Qwen3.5-27B",
}

parser = argparse.ArgumentParser(description="FP8 quantization for CARDS Qwen3.5 models")
parser.add_argument("--only", choices=MODELS.keys(), default=None,
                    help="Quantize only one size (default: all three)")
parser.add_argument("--no-push", dest="push", action="store_false", default=True,
                    help="Skip Hub upload (default: push public)")
parser.add_argument("--keep-local", action="store_true",
                    help="Keep the local output dir after pushing")
args = parser.parse_args()

SIZES = [args.only] if args.only else ["4b", "9b", "27b"]

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import login, HfApi
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

print(f"{'='*60}")
print(f"  Sizes:   {', '.join(SIZES)}")
print(f"  Push:    {args.push} (public)")
print(f"  GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM:    {total_gb:.1f} GB")
print(f"{'='*60}\n")


def quantize_one(size: str) -> None:
    SIZE = size.upper()
    SRC  = MODELS[size]
    OUT  = f"CARDS-Qwen3.5-{SIZE}-FP8"
    REPO = f"C3DS/{OUT}"

    print(f"\n{'#'*60}")
    print(f"#  {SRC}  ->  {REPO}")
    print(f"{'#'*60}")

    # ----- load -----
    print(f"\n[1/3] Loading {SRC} ...")
    t = time.time()
    tokenizer = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForCausalLM.from_pretrained(
        SRC,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  loaded in {time.time() - t:.1f}s")

    # ----- quantize -----
    # FP8_DYNAMIC: per-channel FP8 weights (static), per-token FP8 activations (dynamic).
    # No calibration data needed. lm_head stays in bf16.
    print(f"\n[2/3] Quantizing (FP8_DYNAMIC) ...")
    t = time.time()
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )
    oneshot(model=model, recipe=recipe)
    print(f"  quantized in {time.time() - t:.1f}s")

    # ----- save -----
    print(f"\n[3/3] Saving to ./{OUT} ...")
    t = time.time()
    model.save_pretrained(OUT, save_compressed=True)
    tokenizer.save_pretrained(OUT)
    print(f"  saved in {time.time() - t:.1f}s")

    # ----- push -----
    if args.push:
        print(f"\nPushing public -> {REPO} ...")
        t = time.time()
        api = HfApi()
        api.create_repo(REPO, private=False, exist_ok=True)
        api.upload_folder(
            folder_path=OUT,
            repo_id=REPO,
            repo_type="model",
            commit_message=f"Upload FP8 quantized CARDS-Qwen3.5-{SIZE}",
        )
        print(f"  pushed in {time.time() - t:.1f}s")
        print(f"  https://huggingface.co/{REPO}")

    # ----- free memory + disk for the next size -----
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if args.push and not args.keep_local:
        shutil.rmtree(OUT, ignore_errors=True)
        print(f"  cleaned local ./{OUT}")


overall_start = time.time()
for size in SIZES:
    try:
        quantize_one(size)
    except Exception as e:
        print(f"\n!! {size} failed: {e}")
        raise

print(f"\n{'='*60}")
print(f"  All done in {(time.time() - overall_start) / 60:.1f} min")
print(f"{'='*60}")
if args.push:
    for size in SIZES:
        print(f"  https://huggingface.co/C3DS/CARDS-Qwen3.5-{size.upper()}-FP8")
print("\nServe any of them with:")
print("  vllm serve C3DS/CARDS-Qwen3.5-27B-FP8 --max-model-len 4096")
