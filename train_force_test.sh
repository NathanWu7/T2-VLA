#!/usr/bin/env bash
set -e

# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

# 只让当前进程看到 4,5,6,7 号 GPU
export CUDA_VISIBLE_DEVICES=2,3

# 切到工程根目录
cd /raid/qiweiw/workspace/T2-VLA

#uv run scripts/compute_norm_stats.py --config-name pi0_libero_force_low_mem_finetune

uv run scripts/train.py pi0_libero_force_low_mem_finetune \
  --exp-name force_test2 \
  --data.repo_id=NathanWu7/tabero_force \
  --resume
# --overwrite