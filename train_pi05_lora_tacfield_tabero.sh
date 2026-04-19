#!/bin/bash
#
# 单跑：pi05_lora_tacfield_tabero
# - 先 compute_norm_stats
# - 再 train
#
# 用法示例：
#   sbatch train_pi05_lora_tacfield_tabero.sh
#   RUN_TAG=run1 sbatch train_pi05_lora_tacfield_tabero.sh
#   OVERWRITE=0 sbatch train_pi05_lora_tacfield_tabero.sh
#   WANDB_MODE=offline sbatch train_pi05_lora_tacfield_tabero.sh
#   # 指定节点（可选）
#   sbatch --nodelist=h20-6 train_pi05_lora_tacfield_tabero.sh
#

# ========= Slurm 资源申请（按你们集群规则修改）=========
#SBATCH --job-name=pi05_lora_tacfield_tabero
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 强制跑到指定节点（如需覆盖，也可以在 sbatch 时传 --nodelist=...）
#SBATCH --nodelist=h20-6
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/qiweiw/workspace/T2-VLA/slurm/%x-%j.out
#SBATCH --error=/home/qiweiw/workspace/T2-VLA/slurm/%x-%j.err
# ======================================================

set -euo pipefail
set -x

PROJECT_DIR="/home/qiweiw/workspace/T2-VLA"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/slurm"

CONFIG_NAME="pi05_lora_tacfield_tabero"
RUN_TAG="${RUN_TAG:-25}"
# 恢复训练默认不覆盖（overwrite=0），否则会把已有 checkpoint 目录清掉
OVERWRITE="${OVERWRITE:-0}"
# 是否恢复训练（默认 1）。恢复训练会从最后一个 checkpoint 继续跑。
RESUME="${RESUME:-1}"
# 是否重新计算 norm stats（默认 0）。如果你已经算过，就保持 0。
RUN_NORM_STATS="${RUN_NORM_STATS:-0}"
EXP_NAME="${CONFIG_NAME}_${RUN_TAG}"

echo "Running on host: $(hostname)"
echo "Working dir:     $(pwd)"
echo "Slurm job id:    ${SLURM_JOB_ID:-N/A}"
echo "Slurm nodelist:  ${SLURM_JOB_NODELIST:-N/A}"
echo "Config:          ${CONFIG_NAME}"
echo "Exp name:        ${EXP_NAME}"
echo "Overwrite:       ${OVERWRITE}"
echo "Resume:          ${RESUME}"
echo "Run norm stats:  ${RUN_NORM_STATS}"

# Hydra 报错更全
export HYDRA_FULL_ERROR=1
# JAX/XLA 显存预分配比例
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# ========= HuggingFace 缓存（降低 429 风险）=========
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"
# 默认禁用 xet，减少额外 API 请求（如需开启：export HF_HUB_DISABLE_XET=0）
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# ========= W&B 非交互策略 =========
# 不要把 WANDB_API_KEY 写进脚本（会泄露）。推荐在提交前 export：
#   export WANDB_API_KEY=xxxx
# 或 sbatch 注入：
#   sbatch --export=ALL,WANDB_API_KEY=xxxx ...
#
# 规则：
# - 若你提供了 WANDB_API_KEY 且未显式设置 WANDB_MODE，则默认 online（会上云）
# - 若你未提供 WANDB_API_KEY 且也未设置 WANDB_MODE，则默认 offline（避免交互登录导致失败）
if [[ -n "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then
  export WANDB_MODE=online
elif [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then
  export WANDB_MODE=offline
fi

# GPU 信息（没有也不致命）
nvidia-smi -L || true
nvidia-smi || true

# 依赖（有则跳过）
if [[ ! -d "${PROJECT_DIR}/.venv" ]]; then
  echo "No .venv found, running: uv sync --frozen --no-dev"
  uv sync --frozen --no-dev
fi

overwrite_flag=""
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_flag="--overwrite"
fi

resume_flag=""
if [[ "${RESUME}" == "1" ]]; then
  resume_flag="--resume"
fi

# 可选：给 train.py 追加任意参数（例如调大步数等）
# 用法：EXTRA_TRAIN_ARGS="--num_train_steps=50000" sbatch ...
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

if [[ "${RUN_NORM_STATS}" == "1" ]]; then
  echo "===== compute_norm_stats: ${CONFIG_NAME} ====="
  srun --ntasks=1 bash -lc "cd '${PROJECT_DIR}' && uv run scripts/compute_norm_stats.py --config-name '${CONFIG_NAME}'"
else
  echo "===== skip compute_norm_stats (RUN_NORM_STATS=${RUN_NORM_STATS}) ====="
fi

echo "===== train: ${CONFIG_NAME} (exp-name=${EXP_NAME}) ====="
srun --ntasks=1 bash -lc "cd '${PROJECT_DIR}' && uv run scripts/train.py '${CONFIG_NAME}' --exp-name='${EXP_NAME}' ${resume_flag} ${overwrite_flag} ${EXTRA_TRAIN_ARGS}"

echo "===== done: ${CONFIG_NAME} ====="
set +x


