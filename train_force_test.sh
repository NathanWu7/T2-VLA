#!/usr/bin/env bash
set -e

# 切到工程根目录
cd /raid/qiweiw/workspace/T2-VLA

uv run scripts/train.py pi0_libero_force_low_mem_finetune \
  --exp-name force_test2 \
  --data.repo_id=/raid/qiweiw/data/force_test2 \
  --data.assets.assets_dir=/raid/qiweiw/data/force_test2 \
  --data.assets.asset_id=force_test2