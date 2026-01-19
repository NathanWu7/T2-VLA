#!/bin/bash

# ========= Slurm 资源申请（你需要按你们集群规则修改这块）=========
# 常见要改的：account / partition / GPU 数 / 时间 / CPU 数 / 内存
#SBATCH --job-name=t2vla_train
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/qiweiw/workspace/T2-VLA/slurm/%x-%j.out
#SBATCH --error=/home/qiweiw/workspace/T2-VLA/slurm/%x-%j.err
# 例如（按需取消注释并修改）：
#SBATCH --account=qiweiw
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL
# ==========================================================

set -euo pipefail
set -x

PROJECT_DIR="/home/qiweiw/workspace/T2-VLA"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/slurm"

CONFIG_NAME="pi05_tacforce_tabero"
EXP_NAME="pi05_tacforce_tabero_25"

echo "Running on host: $(hostname)"
echo "Working dir:     $(pwd)"
echo "Slurm job id:    ${SLURM_JOB_ID:-N/A}"
echo "Slurm nodelist:  ${SLURM_JOB_NODELIST:-N/A}"
echo "Config:          ${CONFIG_NAME}"
echo "Exp name:        ${EXP_NAME}"

# 建议保留：Hydra 报错更全
export HYDRA_FULL_ERROR=1
# 控制 JAX/XLA 预分配显存比例，避免一上来吃满整张卡
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# 如果计算节点没外网，你可以改成离线：
# export WANDB_MODE=offline
# export TRANSFORMERS_OFFLINE=1

# 打印 GPU 信息（没有 nvidia-smi 也不致命）
nvidia-smi -L || true
nvidia-smi || true

# 依赖安装策略（强烈建议你在登录节点先跑一次：uv sync --frozen --no-dev）
# 这里做“有则跳过”，避免每次作业都重新装。
if [[ ! -d "${PROJECT_DIR}/.venv" ]]; then
  echo "No .venv found, running: uv sync --frozen --no-dev"
  uv sync --frozen --no-dev
fi

# ========= 这里开始是真正训练命令（按 train2.sh 的写法）=========
echo "===== compute_norm_stats: ${CONFIG_NAME} ====="
srun --ntasks=1 bash -lc "cd '${PROJECT_DIR}' && uv run scripts/compute_norm_stats.py --config-name '${CONFIG_NAME}'"

echo "===== train: ${CONFIG_NAME} ====="
srun --ntasks=1 bash -lc "cd '${PROJECT_DIR}' && uv run scripts/train.py '${CONFIG_NAME}' --exp-name='${EXP_NAME}' --overwrite"
# ===============================================================


set +x
