# Serving (`serve/start_cards_vllm.sh`)

Env-driven vLLM launcher. `MODEL` and `PREC` (precision) select which Hub
repo to serve.

```
MODEL=4b PREC=fp8 ./serve/start_cards_vllm.sh
```

`--served-model-name` is fixed to the BF16 name regardless of precision,
so client code (`infer.py`) doesn't need to change when swapping FP8↔BF16.

## Flags that matter

- `--enable-prefix-caching`: every CARDS request shares the long system
  prompt; prefix caching turns that into a one-time cost per worker.
- `--max-num-seqs`: the default is too high for long-context KV cache on
  one GPU; lower if you see thrashing.
- `--max-model-len`: 8192 is the training max; reduce if memory-tight.
- `--language-model-only`: required for Qwen text-only fine-tunes that
  ship with the multimodal config wrapper.
