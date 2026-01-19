#!/usr/bin/env bash
#set -e

# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

# 只让当前进程看到 4,5,6,7 号 GPU
#export CUDA_VISIBLE_DEVICES=2,3

# 切到工程根目录（按需修改为你本机的 T2-VLA 路径）
#cd /raid/qiweiw/workspace/T2-VLA

###############################################################################
# 示例 1：老的 pi0_libero_force_low_mem_finetune（关节 + 力）
###############################################################################

# 计算归一化统计（只需要跑一次）
# uv run scripts/compute_norm_stats.py pi0_libero_force_low_mem_finetune
#uv run scripts/compute_norm_stats.py --config-name pi0_lora_tacforce_tabero
# 继续在 Tabero 力数据上训练（或恢复训练）
uv run scripts/compute_norm_stats.py --config-name pi0_lora_tacforce_tabero
uv run scripts/train.py pi0_lora_tacforce_tabero --exp-name=pi0_lora_tacforce_tabero_25 --overwrite
# 如需从头重跑同名实验，可以加上 --overwrite
#   --overwrite
uv run scripts/compute_norm_stats.py --config-name pi0_lora_tacall_tabero
uv run scripts/train.py pi0_lora_tacall_tabero --exp-name=pi0_lora_tacall_tabero_25 --overwrite
###############################################################################
# 示例 2：三路图像 + 13D 动作（无 tactile token）—— pi0_lora_tacimg_force
###############################################################################

# 计算归一化统计
# uv run scripts/compute_norm_stats.py pi0_lora_tacimg_force

# 训练
# uv run scripts/train.py pi0_lora_tacimg_force \
#   --exp-name tacimg_force_run1 \
#   --data.repo_id=NathanWu7/tabero \
#   --overwrite

###############################################################################
# 示例 3：两路图像 + 触觉力场 tactile + 13D 动作—— pi0_lora_tacfield_force
###############################################################################

# 计算归一化统计
# uv run scripts/compute_norm_stats.py pi0_lora_tacfield_force

# 训练
# uv run scripts/train.py pi0_lora_tacfield_force \
#   --exp-name tacfield_force_run1 \
#   --data.repo_id=NathanWu7/tabero \
#   --overwrite
