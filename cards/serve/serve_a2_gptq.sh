#!/usr/bin/env bash
# Serve the int4 (W4A16 / GPTQ) CARDS-Wind Gemma 4 12B on a single A2 card.
#
#   ./serve_a2_gptq.sh              # GPU 2, port 8002
#   GPU=3 PORT=8003 ./serve_a2_gptq.sh
#
# One card, one process. Run it more than once on different GPUs/ports if you
# want more throughput — infer.py --base-url picks which one to hit.
#
# Deliberately NOT --data-parallel-size 3: three ranks repacking 4-bit weights
# for Marlin at load time, concurrently on 200 GB/s cards, blew past vLLM's
# 600s engine-ready timeout. Separate single-GPU processes start fine.
#
# GPUs 0/1 run a Qwen embedding model on port 8000 — leave them alone.
# --host 0.0.0.0 is required to reach this over Tailscale.
#
# Sizing: KV costs 336 KiB/token for this model (8 full-attention layers at
# 1 kv_head x 512, plus 40 sliding layers at 8 kv_heads x 256; vLLM does not
# exploit the sliding-window cap). Weights are ~7.35 GiB, so on a 15 GiB A2
# that leaves ~6.1 GiB KV = ~19k tokens = ~4.6x concurrency at 4096.
# Measured: 18,693 tokens. Hence --max-num-seqs 6, not 12.
set -euo pipefail

VENV="${VENV:-/home/projects/open-climate-index/.venv}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

CUDA_VISIBLE_DEVICES="${GPU:-2}" vllm serve "${MODEL:-C3DS/CARDS-Wind-Gemma4-12B-GPTQ-w4a16}" \
  --served-model-name "${SERVED_NAME:-CARDS-Wind-Gemma4-12B-GPTQ-w4a16}" \
  --host 0.0.0.0 \
  --port "${PORT:-8002}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
  --max-num-seqs "${MAX_NUM_SEQS:-6}" \
  --enable-prefix-caching \
  --trust-remote-code
