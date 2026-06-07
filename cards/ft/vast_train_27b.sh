#!/usr/bin/env bash
# Launch the combined (API+chat) Qwen3.6-27B SFT run ON the vast.ai H200 box.
#   /workspace/vast_train_27b.sh                  # full 3-epoch reference run
#   /workspace/vast_train_27b.sh --variant CARDS-Wind-Qwen3.6-27B-3ep
# Extra args are passed through to train.py.
# Run inside tmux/screen — this is an ~11-12h job.
set -euo pipefail

cd /workspace

# 1. Credentials — refuse to start without them (checkpoint pushes need HF_TOKEN).
if [ ! -f /workspace/.env_train ]; then
  echo "ERROR: /workspace/.env_train missing. Create it first:"
  echo "  printf 'HF_TOKEN=hf_...\nHF_USERNAME=C3DS\n' > /workspace/.env_train && chmod 600 /workspace/.env_train"
  exit 1
fi
set -a; source /workspace/.env_train; set +a
: "${HF_TOKEN:?HF_TOKEN not set in .env_train}"
export HF_USERNAME="${HF_USERNAME:-C3DS}"

# 2. GPU must be free (a leftover vLLM server will OOM the run an hour in).
USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$USED_MB" -gt 2000 ]; then
  echo "ERROR: GPU already has ${USED_MB} MiB in use. Stop the server first:"
  echo "  pkill -f 'vllm serve'"
  exit 1
fi

# 3. Launch. Defaults = the clean reference recipe: 3 epochs, batch 1 x accum 8,
#    lr 2e-4. hub_strategy=checkpoint makes the run resumable from the Hub.
mkdir -p /workspace/logs
LOG=/workspace/logs/train_$(date +%Y%m%d_%H%M).log
echo "[train] logging to $LOG"
exec uv run /workspace/train.py \
  --base-model Qwen/Qwen3.6-27B \
  --combined \
  "$@" \
  2>&1 | tee "$LOG"
