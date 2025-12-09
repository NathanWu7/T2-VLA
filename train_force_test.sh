#!/usr/bin/env bash
set -e

# 只让当前进程看到 4,5,6,7 号 GPU
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 切到工程根目录
cd /raid/qiweiw/workspace/T2-VLA

uv run scripts/train.py pi0_libero_force_low_mem_finetune \
  --exp-name force_test2 \
  --data.repo_id=NathanWu7/tabero_force