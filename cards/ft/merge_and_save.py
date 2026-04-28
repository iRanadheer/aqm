"""
Merge LoRA adapter (checkpoint-600) into Qwen3.6-27B base and save merged bf16 model.

Usage on pod:
    python merge_and_save.py
Then upload via:
    hf upload iRanadheer/CARDS-Qwen3.6-27B /workspace/CARDS-Qwen3.6-27B-merged . --repo-type=model
"""
import os
import sys
import time
import shutil

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from unsloth import FastLanguageModel

ADAPTER_DIR = "/workspace/CARDS-Qwen3.6-27B/checkpoint-600"
MERGED_DIR = "/workspace/CARDS-Qwen3.6-27B-merged"
MAX_SEQ_LENGTH = 4096

print(f"[1/3] Loading adapter + base from {ADAPTER_DIR}...")
start = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
    load_in_16bit=True,
)
print(f"  loaded in {time.time()-start:.1f}s")
print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

print(f"\n[2/3] Merging LoRA into base and saving to {MERGED_DIR}...")
start = time.time()
model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method="merged_16bit",
)
print(f"  merged+saved in {time.time()-start:.1f}s")

src_ct = os.path.join(ADAPTER_DIR, "chat_template.jinja")
if os.path.exists(src_ct) and not os.path.exists(os.path.join(MERGED_DIR, "chat_template.jinja")):
    shutil.copy(src_ct, os.path.join(MERGED_DIR, "chat_template.jinja"))
    print(f"  copied chat_template.jinja")

print(f"\n[3/3] Done. Merged model at {MERGED_DIR}")
print("Contents:")
for f in sorted(os.listdir(MERGED_DIR)):
    size = os.path.getsize(os.path.join(MERGED_DIR, f))
    print(f"  {f}  ({size/1e9:.2f} GB)" if size > 1e8 else f"  {f}  ({size/1e6:.2f} MB)")
