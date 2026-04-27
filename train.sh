#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-4562}"

python -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes 2 \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  train.py \
  --config configs/train/train.yaml
