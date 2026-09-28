#!/usr/bin/env bash
# Serve a FLICC model with vLLM, tuned for throughput on a single RTX PRO 6000.
#
#   ./serve_vllm.sh                              # C3DS/FLICC-Qwen3.5-0.8B
#   ./serve_vllm.sh C3DS/FLICC-Qwen3.5-9B        # any other model
#
# Every flag below was checked against `vllm serve --help=all` on this box
# (vLLM 0.27.1) — none are assumed.
#
# The workload is 713 classification calls (457 val + 256 test) whose system
# prompt is byte-identical every time, so prefix caching is the flag that
# matters most here, not raw batch size.
set -euo pipefail

MODEL="${1:-C3DS/FLICC-Qwen3.5-0.8B}"
PORT="${PORT:-8000}"

source /venv/main/bin/activate

# HF_TOKEN is read from the environment, never baked into this file. Only
# needed if a repo is private; export it before running if so.
: "${HF_TOKEN:=}"
[ -n "$HF_TOKEN" ] && export HF_TOKEN

echo "model : $MODEL"
echo "port  : $PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --async-scheduling \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384

# Why each throughput flag:
#
#   --enable-prefix-caching     The ~1550-token system prompt is identical on
#                               all 713 requests. Cached once, every subsequent
#                               request skips re-prefilling it. Biggest single
#                               win for this workload. Default is None (off).
#   --async-scheduling          vLLM's own help: "helps to avoid gaps in GPU
#                               utilization, leading to better latency and
#                               throughput."
#   --enable-chunked-prefill    Lets long prefills interleave with decode
#                               instead of stalling the batch.
#   --max-num-seqs 512          0.8B weights are ~1.6 GB of a 96 GB card, so
#                               almost all memory is KV cache — run wide.
#                               Lower this for the 9B/27B.
#   --max-num-batched-tokens    16384 = 2x max-model-len, so a full-length
#                               prefill never has to be split more than needed.
#   --gpu-memory-utilization    0.95 vs the 0.92 default; more KV cache. Back
#                               off if a larger model OOMs at load.
#   --max-model-len 8192        Matches the training max_seq. Rows are ~2k
#                               tokens, so there is no truncation risk.
#
# Flags deliberately NOT used: --swap-space and --max-seq-len-to-capture do not
# exist in 0.27.1 (verified), and tensor parallelism is pointless on one GPU.
