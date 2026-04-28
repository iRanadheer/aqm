#!/bin/bash
# Start vLLM server for CARDS-Qwen3.5 on RTX PRO 6000 (Blackwell).
#
# Usage:
#   ./start_cards_vllm.sh              # default: 4B FP8
#   MODEL=9b  PREC=fp8  ./start_cards_vllm.sh
#   MODEL=27b PREC=bf16 ./start_cards_vllm.sh
#
# Env vars:
#   MODEL  = 4b | 9b | 27b          (default: 4b)
#   PREC   = fp8 | bf16             (default: fp8)
#   PORT   = server port            (default: 8080)

set -euo pipefail

MODEL="${MODEL:-4b}"
PREC="${PREC:-fp8}"
PORT="${PORT:-8080}"

case "$MODEL" in
  4b)  SIZE="4B"  ;;
  9b)  SIZE="9B"  ;;
  27b) SIZE="27B" ;;
  *) echo "Bad MODEL: $MODEL (want 4b|9b|27b)"; exit 1 ;;
esac

case "$PREC" in
  fp8)  REPO="C3DS/CARDS-Qwen3.5-${SIZE}-FP8" ;;
  bf16) REPO="C3DS/CARDS-Qwen3.5-${SIZE}"     ;;
  *) echo "Bad PREC: $PREC (want fp8|bf16)"; exit 1 ;;
esac

SERVED_NAME="C3DS/CARDS-Qwen3.5-${SIZE}"

echo "Serving $REPO  (served-model-name=$SERVED_NAME)  on port $PORT"
echo

# Blackwell (B200 / RTX PRO 6000) optimized flags:
#   --kv-cache-dtype fp8          : Blackwell-native FP8 KV cache -> 2x concurrency
#   --enable-prefix-caching       : share cached codebook across requests (big CARDS win)
#   --max-num-batched-tokens 131072: prefill throughput with long shared system prompt
#   --max-num-seqs 1024           : high concurrent request ceiling
#   --gpu-memory-utilization 0.90 : 96GB headroom for KV cache
exec vllm serve "$REPO" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 8192 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 131072 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --disable-log-requests
