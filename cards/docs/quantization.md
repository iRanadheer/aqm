# Quantization (`ft/quantize.py`)

Post-training FP8_DYNAMIC quantization via `llmcompressor`. PEP-723.

```
uv run ft/quantize.py --src <merged-repo> [--dst-org <org>] [--no-push]
```

## Recipe

Per-channel FP8 weights, per-token FP8 activations, `lm_head` left in bf16.
No calibration data needed.

## Output

Quantized model is saved with `save_compressed=True` and uploaded to
`<dst-org>/<src-name>-FP8`. Local copy is cleaned by default after push
(`--keep-local` to keep it).

## Why FP8_DYNAMIC

It's the simplest scheme that's lossless on this task — within sampling
noise of the BF16 source across the eval set. No need for calibration data
or a longer recipe.
