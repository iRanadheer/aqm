# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "llmcompressor>=0.8.0",
#   "compressed-tensors",
#   "transformers>=5.10.2",   # gemma4_unified arch first ships in 5.10
#   "tokenizers>=0.21",
#   "torch",
#   "torchvision",   # gemma4_unified image/video processors import torchvision.transforms.v2
#   "pillow",        # Gemma4UnifiedProcessor is @requires(backends=("vision",))
#   "accelerate",
#   "datasets",
#   "huggingface_hub",
# ]
#
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# index-strategy = "unsafe-best-match"
# override-dependencies = ["transformers>=5.10.2"]
# ///
"""
W4A16 (int4) quantizer for the CARDS line — Ampere-servable.

Companion to ft/quantize.py (FP8_DYNAMIC). FP8 needs SM89+; the A2 cards are
GA107/SM86, so FP8 has no hardware path there. int4 W4A16 does: both AWQ and
GPTQ land on vLLM's Marlin kernels, which need only SM80+. W4A16 is also the
only 4-bit format that supports runtime LoRA adapters in vLLM (bitsandbytes
does not), which matters for the multi-adapter serving plan.

Runs GPTQ then AWQ from the SAME calibration set and pushes each as it
finishes. GPTQ goes first on purpose: it is layer-wise and architecture-
agnostic, so it should work on any arch. AWQ needs architecture-aware
mappings for its rescaling transform and there is no upstream Gemma 4 AWQ
example, so it is the one that might fail — by then GPTQ is already pushed.

  uv run ft/quantize_w4a16.py                              # 12B, both methods
  uv run ft/quantize_w4a16.py --method gptq
  uv run ft/quantize_w4a16.py --no-push --num-calib 32     # smoke test

Calibration is IN-DOMAIN by design. A CARDS model spends ~85% of every
sequence inside a byte-identical system prompt, so calibrating on a generic
corpus would fit scales to activations the model never sees at serve time.
We use the exact training rows, rendered through the model's own chat template.

IGNORE LIST — verified against the checkpoint's safetensors header, not
guessed. The 12B is the ENCODER-FREE `gemma4_unified` variant: it has no
vision_tower/audio_tower, but it does have `model.vision_embedder.patch_dense`,
which IS a Linear. Upstream's Gemma 4 example ignores tower names only and
would quantize that layer in 4 bits. Real non-language Linears here:
    model.vision_embedder.patch_dense
    model.embed_vision.embedding_projection
    model.embed_audio.embedding_projection
`lm_head` is a no-op entry (embeddings are tied, no separate tensor) but is
kept for portability to untied checkpoints such as the Qwen line.
"""

import argparse
import gc
import os
import shutil
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--src", default=os.environ.get("SRC_MODEL", "C3DS/CARDS-Wind-Gemma4-12B"))
ap.add_argument("--calib-repo", default=os.environ.get("CALIB_REPO", "iRanadheer/cards-wind-qwen-chat"))
ap.add_argument("--calib-file", default=os.environ.get("CALIB_FILE", "gemma4/train.jsonl"),
                help="Must match the data the model was trained on.")
ap.add_argument("--dst-org", default=os.environ.get("DST_ORG", "C3DS"))
ap.add_argument("--method", choices=["gptq", "awq", "both"], default=os.environ.get("METHOD", "both"))
ap.add_argument("--num-calib", type=int, default=int(os.environ.get("NUM_CALIB", 512)))
ap.add_argument("--max-seq", type=int, default=int(os.environ.get("MAX_SEQ", 2048)))
ap.add_argument("--no-push", dest="push", action="store_false", default=True)
ap.add_argument("--keep-local", action="store_true")
args = ap.parse_args()

SRC_NAME = args.src.split("/")[-1]
# GPTQ first — see module docstring.
METHODS = ["gptq", "awq"] if args.method == "both" else [args.method]

IGNORE = [
    "lm_head",
    "re:.*vision_embedder.*",   # patch_dense is a Linear on the unified arch
    "re:.*embed_vision.*",      # embedding_projection is a Linear
    "re:.*embed_audio.*",       # embedding_projection is a Linear
    "re:.*vision_tower.*",      # present on the tower-based variants
    "re:.*audio_tower.*",
]

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

print("=" * 70)
print(f"  Source     : {args.src}")
print(f"  Calibration: {args.calib_repo}::{args.calib_file}")
print(f"               {args.num_calib} samples @ max_seq {args.max_seq}")
print(f"  Methods    : {' -> '.join(m.upper() for m in METHODS)}   (W4A16)")
print(f"  Push       : {args.push} -> {args.dst_org}/  (public)")
print("=" * 70)

from huggingface_hub import HfApi, login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
elif args.push:
    sys.exit("HF_TOKEN not set but pushing was requested.")

import torch
import transformers

print(f"\ntorch {torch.__version__} | transformers {transformers.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} ({p.total_memory / 1e9:.1f} GB, SM{p.major}{p.minor})")
else:
    print("WARNING: no CUDA device — calibration will be unusably slow.")

import llmcompressor
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from transformers import AutoModelForImageTextToText, AutoProcessor

print(f"llmcompressor {getattr(llmcompressor, '__version__', 'unknown')}")


def load_model_and_processor():
    """Returns (model, tokenizer, processor|None).

    Gemma 4 loads through the image-text-to-text head (per upstream's example;
    the repo ships processor_config.json, which confirms it). Falls back to
    CausalLM so the same script serves the text-only Qwen line.

    The tokenizer is loaded SEPARATELY and is what calibration uses — this is
    a text-only calibration, so it must not depend on the multimodal processor
    resolving. The processor is loaded best-effort and only used so the output
    repo keeps its multimodal config; if its vision backends are unavailable we
    save the tokenizer alone rather than failing the whole run.
    """
    from transformers import AutoTokenizer
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.src, dtype=torch.bfloat16)
        print("  model loaded via AutoModelForImageTextToText")
    except Exception as e:
        print(f"  ImageTextToText load failed ({type(e).__name__}: {e}); trying CausalLM")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.src, dtype=torch.bfloat16)
        print("  model loaded via AutoModelForCausalLM")

    tok = AutoTokenizer.from_pretrained(args.src)
    try:
        proc = AutoProcessor.from_pretrained(args.src)
        print("  processor loaded (multimodal config will be preserved)")
    except Exception as e:
        proc = None
        print(f"  WARNING: processor unavailable ({type(e).__name__}); saving tokenizer only")
    return model, tok, proc


def build_calibration(tok):
    ds = load_dataset(args.calib_repo, data_files=args.calib_file, split="train", token=hf_token)
    print(f"  {len(ds)} rows available in {args.calib_file}")
    ds = ds.shuffle(seed=42).select(range(min(args.num_calib, len(ds))))

    def render(row):
        # Full turn including the assistant response: calibration should see the
        # generation-time activation distribution, not the prompt alone.
        return {"text": tok.apply_chat_template(row["messages"], tokenize=False)}

    def tokenize(row):
        return tok(row["text"], padding=False, max_length=args.max_seq,
                   truncation=True, add_special_tokens=False)

    ds = ds.map(render)
    ds = ds.map(tokenize, remove_columns=ds.column_names, desc="tokenizing")
    lens = sorted(len(x) for x in ds["input_ids"])
    print(f"  {len(ds)} samples | tokens: median {lens[len(lens) // 2]}, max {lens[-1]}")
    return ds


def recipe_for(method):
    """Each method follows its own canonical upstream recipe.

    GPTQ uses symmetric W4A16 (-> vLLM gptq_marlin). AWQ is a TWO-stage recipe:
    AWQModifier applies the activation-aware rescaling transform, then
    QuantizationModifier does the actual rounding, asymmetric (-> awq_marlin).
    Both kernels require only SM80+, so both run on the A2s.
    """
    if method == "gptq":
        return GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE,
                            dampening_frac=0.01)
    return [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=IGNORE),
    ]


def sanity_generate(model, tok):
    """Cheap smoke test: a broken quantization produces obvious garbage here."""
    try:
        dispatch_model(model)
        ids = tok("The main argument against climate policy is", return_tensors="pt").input_ids
        out = model.generate(ids.to(model.device), max_new_tokens=40, do_sample=False)
        print("  SAMPLE:", tok.decode(out[0], skip_special_tokens=True)[:300].replace("\n", " "))
    except Exception as e:
        print(f"  (sanity generation skipped: {type(e).__name__}: {e})")


def dir_size_gb(path):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(path) for f in fs) / 1e9


CARD = """---
license: gemma
base_model: {src}
tags: [compressed-tensors, {ml}, w4a16, int4, cards, climate]
---

# {out}

int4 (W4A16) quantization of [`{src}`](https://huggingface.co/{src}) via
`llm-compressor`, using **{m}**.

## Why int4

Built to serve on **NVIDIA A2** (GA107, SM86). FP8 needs SM89+, so the sibling
FP8 checkpoint has no hardware path on Ampere. int4 W4A16 runs on vLLM's Marlin
kernels (SM80+), and is the only 4-bit format in vLLM that also supports runtime
LoRA adapters.

## Calibration

- **{n} samples** from [`{cr}`](https://huggingface.co/datasets/{cr}) (`{cf}`)
- max_seq_length {ms}, rendered through the model's own chat template
- In-domain by design: this model spends most of every sequence inside a fixed
  system prompt, so a generic calibration corpus would not match serve-time
  activations.

`lm_head`, the tied embedding table, and the vision/audio projection layers
(including `vision_embedder.patch_dense` on the encoder-free unified arch) are
left in BF16.

## Serving

```bash
vllm serve {repo} --max-model-len 4096 --trust-remote-code
```

## Status

Provisional name. A sibling built with the other quantization method from
identical calibration data is published alongside it; the two are being
compared on the CARDS test split before one is adopted.
"""

results = {}
for method in METHODS:
    out = f"{SRC_NAME}-{method.upper()}-w4a16"
    repo = f"{args.dst_org}/{out}"
    print("\n" + "=" * 70)
    print(f"  {method.upper()}  ->  {repo}")
    print("=" * 70)

    try:
        t0 = time.time()
        print(f"\n[1/5] Loading {args.src} ...")
        model, tok, proc = load_model_and_processor()
        print(f"  loaded in {time.time() - t0:.0f}s")

        print("\n[2/5] Building calibration set ...")
        calib = build_calibration(tok)

        print(f"\n[3/5] Quantizing ({method.upper()}) ...")
        t = time.time()
        oneshot(model=model, dataset=calib, recipe=recipe_for(method),
                max_seq_length=args.max_seq, num_calibration_samples=len(calib))
        print(f"  quantized in {time.time() - t:.0f}s")

        print("\n[4/5] Sanity generation ...")
        sanity_generate(model, tok)

        print(f"\n[5/5] Saving ./{out} ...")
        model.save_pretrained(out, save_compressed=True)
        tok.save_pretrained(out)
        if proc is not None:
            proc.save_pretrained(out)
        size = dir_size_gb(out)
        print(f"  {size:.2f} GB on disk")
        results[method] = f"{size:.2f} GB"

        if args.push:
            print(f"\nPushing -> {repo} ...")
            api = HfApi()
            api.create_repo(repo, private=False, exist_ok=True)
            with open(os.path.join(out, "README.md"), "w") as f:
                f.write(CARD.format(src=args.src, out=out, repo=repo, m=method.upper(),
                                    ml=method, n=len(calib), cr=args.calib_repo,
                                    cf=args.calib_file, ms=args.max_seq))
            api.upload_folder(folder_path=out, repo_id=repo, repo_type="model",
                              commit_message=f"{method.upper()} W4A16 of {SRC_NAME} "
                                             f"({len(calib)} in-domain calibration samples)")
            print(f"  https://huggingface.co/{repo}")

    except Exception as e:
        # Never let a later method's failure discard an earlier method's result.
        import traceback
        print(f"\n!!! {method.upper()} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        results[method] = f"FAILED ({type(e).__name__})"

    for v in ("model", "tok", "proc", "calib"):
        globals().pop(v, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if args.push and not args.keep_local:
        shutil.rmtree(out, ignore_errors=True)

print("\n" + "=" * 70)
print("  DONE")
for m in METHODS:
    print(f"    {m.upper():<5} {results.get(m, 'not run')}")
print("\n  Next: serve each on an A2, run infer.py with the SAME flags that")
print("  produced cards-wind-gemma4-12b-bf16.jsonl, then:")
print("    python evals/significance.py pair test <slug> cards-wind-gemma4-12b-bf16")
print("=" * 70)
