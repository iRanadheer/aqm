# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git@main",
#   "transformers>=5.10.2",  # gemma4_unified arch first ships in 5.10
#   "tokenizers>=0.21",
#   "torch",
#   "accelerate",
#   "huggingface_hub[hf_transfer]",
# ]
#
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# index-strategy = "unsafe-best-match"
# override-dependencies = ["transformers>=5.10.2"]
# ///
"""
Post-training FP8_DYNAMIC quantizer.

  uv run quantize.py --src iRanadheer/CARDS-Wind-Qwen3.6-27B
  uv run quantize.py --src <merged_repo> --dst-org C3DS --no-push
  SRC_MODEL=... DST_ORG=... uv run quantize.py     # env-driven also fine

Per-channel FP8 weights, per-token FP8 activations. No calibration. lm_head
stays in bf16. Within ±0.002 sF1 of BF16 on CARDS — effectively lossless.

Output repo: <dst-org>/<src-name>-FP8.
"""

import argparse
import gc
import os
import shutil
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.environ.get("SRC_MODEL"),
                help="Source model HF id (or set SRC_MODEL)")
ap.add_argument("--dst-org", default=os.environ.get("DST_ORG", "C3DS"),
                help="Destination HF org (default: C3DS)")
ap.add_argument("--no-push", dest="push", action="store_false", default=True)
ap.add_argument("--keep-local", action="store_true",
                help="Keep local quantized dir after pushing")
args = ap.parse_args()

if not args.src:
    ap.error("Provide --src or set SRC_MODEL")

OUT = f"{args.src.split('/')[-1]}-FP8"
REPO = f"{args.dst_org}/{OUT}"

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import HfApi, login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print(f"  Source: {args.src}")
print(f"  Output: ./{OUT}")
print(f"  Repo:   {REPO}  (push={args.push})")
if torch.cuda.is_available():
    print(f"  GPU:    {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
print("=" * 60)


print(f"\n[1/3] Loading {args.src} ...")
t = time.time()
tokenizer = AutoTokenizer.from_pretrained(args.src)
try:
    model = AutoModelForCausalLM.from_pretrained(args.src, dtype=torch.bfloat16, device_map="auto")
except ValueError:
    # Unified/multimodal archs (e.g. gemma4_unified) don't map to CausalLM.
    from transformers import AutoModelForMultimodalLM
    model = AutoModelForMultimodalLM.from_pretrained(args.src, dtype=torch.bfloat16, device_map="auto")
    print("  (loaded via AutoModelForMultimodalLM)")
print(f"  loaded in {time.time() - t:.1f}s")

print("\n[2/3] Quantizing (FP8_DYNAMIC) ...")
t = time.time()
# Vision/audio/projector ignores are no-ops on text-only models; on unified
# models they keep the non-text towers in bf16 (vLLM expects this layout).
oneshot(model=model, recipe=QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC",
    ignore=["lm_head", "re:.*vision.*", "re:.*audio.*", "re:.*projector.*"],
))
print(f"  quantized in {time.time() - t:.1f}s")

print(f"\n[3/3] Saving to ./{OUT} ...")
t = time.time()
model.save_pretrained(OUT, save_compressed=True)
tokenizer.save_pretrained(OUT)
print(f"  saved in {time.time() - t:.1f}s")

if args.push:
    print(f"\nPushing -> {REPO} ...")
    t = time.time()
    api = HfApi()
    api.create_repo(REPO, private=False, exist_ok=True)
    api.upload_folder(folder_path=OUT, repo_id=REPO, repo_type="model",
                      commit_message=f"Upload FP8 quantized {args.src.split('/')[-1]}")
    print(f"  pushed in {time.time() - t:.1f}s")
    print(f"  https://huggingface.co/{REPO}")

del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

if args.push and not args.keep_local:
    shutil.rmtree(OUT, ignore_errors=True)
    print(f"  cleaned local ./{OUT}")

print(f"\nDone. Serve with:")
print(f"  vllm serve {REPO} --max-model-len 4096 --trust-remote-code --language-model-only")
