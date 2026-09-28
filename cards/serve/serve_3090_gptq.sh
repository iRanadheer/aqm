#!/usr/bin/env bash
# Serve the int4 (W4A16 / GPTQ) CARDS-Wind Gemma 4 12B on a single RTX 3090.
#
#   ./serve_3090_gptq.sh          # creates the venv on first run, then serves
#
# Single card, so no data parallelism — see serve_a2_gptq.sh for the 3x A2 build.
#
# WHY transformers IS PINNED TO <5.15  (this is the pin that matters)
#   Gemma 4 is heterogeneous: 8 full-attention layers use head_dim 512 /
#   1 kv_head, the other 40 sliding layers use 256 / 8. transformers 5.15.0
#   (released 2026-08-10) added the per-layer "heterogeneity" config machinery,
#   which raises AmbiguousGlobalPerLayerAttributeError on config.head_dim.
#   That exception does NOT inherit from AttributeError, so vLLM's
#   `getattr(hf_text_config, "head_dim", 0)` in
#   model_arch_config_convertor.get_head_size() propagates it instead of
#   falling back to the default, and the server dies before loading weights.
#
#   vllm 0.25.1 declares only `transformers>=5.5.3` — no upper bound — so a
#   fresh install silently pulls 5.15.0 and breaks. The A2 box works purely
#   because its venv was built before 5.15 shipped. Every vllm release to date
#   has this bug; the fix (vllm PR #51757) merged 2026-08-13, AFTER 0.27.1
#   (2026-08-11), so no published release contains it.
#   Once vllm >=0.27.2 ships, both pins can go.
#
#   Do NOT "fix" this with allow_global_per_layer_attribute_access=True: that
#   returns the global head_dim (256) for ALL layers, silently mis-sizing KV
#   for the 8 full-attention layers that actually use 512.
#
# Why --max-num-seqs is 12 here but 6 on the A2:
#   KV costs 336 KiB/token for this model. Weights are ~7.35 GiB.
#     A2   15.0 GiB -> ~6.1 GiB KV  -> ~19k tokens -> ~4.6x @ 4096  (measured: 18,693)
#     3090 24.0 GiB -> ~14.6 GiB KV -> ~46k tokens -> ~11x  @ 4096
#   Same SM86 as the A2 so the Marlin int4 path is identical, but at 936 GB/s
#   vs 200 GB/s the 3090 decodes ~4.7x faster.
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/cards-vllm}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.14.1}"   # last release before 5.15

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "Creating venv at $VENV (vllm==$VLLM_VERSION, transformers==$TRANSFORMERS_VERSION) ..."
  mkdir -p "$(dirname "$VENV")"
  if command -v uv >/dev/null; then
    uv venv --python 3.12 "$VENV"
    # transformers pinned in the SAME resolve, so vllm can't drag 5.15 back in
    VIRTUAL_ENV="$VENV" uv pip install "vllm==$VLLM_VERSION" "transformers==$TRANSFORMERS_VERSION"
  else
    "$(command -v python3.12 || command -v python3)" -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install "vllm==$VLLM_VERSION" "transformers==$TRANSFORMERS_VERSION"
  fi
fi

source "$VENV/bin/activate"
TF_HAVE="$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo '?')"
echo "venv: $VENV | python $(python -V 2>&1 | cut -d' ' -f2) | vllm $(vllm --version 2>&1 | head -1) | transformers $TF_HAVE"

# An existing venv may predate this pin, or something may have upgraded it.
case "$TF_HAVE" in
  5.1[5-9].*|5.[2-9][0-9].*|[6-9].*)
    echo "ERROR: transformers $TF_HAVE breaks Gemma 4 heterogeneous configs (see header)."
    echo "       Fix: pip install 'transformers==$TRANSFORMERS_VERSION'"
    exit 1 ;;
esac

CUDA_VISIBLE_DEVICES="${GPU:-0}" vllm serve "${MODEL:-C3DS/CARDS-Wind-Gemma4-12B-GPTQ-w4a16}" \
  --served-model-name "${SERVED_NAME:-CARDS-Wind-Gemma4-12B-GPTQ-w4a16}" \
  --host 0.0.0.0 \
  --port "${PORT:-8002}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
  --max-num-seqs "${MAX_NUM_SEQS:-12}" \
  --enable-prefix-caching \
  --trust-remote-code
