#!/bin/bash
#
# 用 Slurm Job Array 并行跑 3 个实验（每个 array task 跑 1 个 config）
#
# 用法：
#   sbatch train3_array.sh
#   RUN_TAG=run1 sbatch train3_array.sh
#   OVERWRITE=0 sbatch train3_array.sh
#   # 限制同时最多跑 1 个（等价于顺序，但仍是 3 个独立任务）
#   sbatch --array=0-2%1 train3_array.sh
#
# 说明：
# - job array 会创建多个“独立”任务：日志独立、失败互不影响、调度器按资源可用性并行调度。
# - 每个任务跑 1 个 config，可并发执行。

# ========= Slurm 资源申请（你需要按你们集群规则修改这块）=========
#SBATCH --job-name=t2vla
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 强制跑到指定节点（如需覆盖，也可以在 sbatch 时传 --nodelist=...）SBATCH --nodelist=h20-2
#
# 这里的 GPU 数是“每个 array task”的资源申请。
# 当前设置为每个任务 8 张卡。
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#
# 注意：array 输出里用 %A(作业ID)_%a(array下标) 以区分三份日志
#SBATCH --output=/home/qiweiw/workspace/T2-VLA/slurm/%x-%A_%a.out
#SBATCH --error=/home/qiweiw/workspace/T2-VLA/slurm/%x-%A_%a.err
# ==========================================================

set -euo pipefail
set -x

PROJECT_DIR="/home/qiweiw/workspace/T2-VLA"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/slurm"

RUN_TAG="${RUN_TAG:-}"
# 恢复训练默认不覆盖（overwrite=0），否则会把已有 checkpoint 目录清掉
OVERWRITE="${OVERWRITE:-0}"
# 是否恢复训练（默认 1）。恢复训练会从最后一个 checkpoint 继续跑。
RESUME="${RESUME:-1}"
# 是否重新计算 norm stats（默认 0）。如果你已经算过，就保持 0。
RUN_NORM_STATS="${RUN_NORM_STATS:-0}"

# 训练配置约束：overwrite 与 resume 不能同时开启。
if [[ "${OVERWRITE}" == "1" && "${RESUME}" == "1" ]]; then
  echo "OVERWRITE=1 detected, forcing RESUME=0 to avoid conflict."
  RESUME="0"
fi

# ========= HuggingFace 下载/缓存设置（避免并行触发 429 限流）=========
# 1) 三个 task 共用同一个 HF cache（在共享文件系统上），命中后不会重复下载
# 2) 默认禁用 xet（有时会触发额外 token 请求），需要 xet 的话可显式设 HF_HUB_DISABLE_XET=0
# 3) 对“首次访问 HuggingFace”的阶段做加锁/错峰，避免 3 个任务同时打到 HF
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

# 默认禁用 xet，降低额外 API 请求；如需开启：export HF_HUB_DISABLE_XET=0
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# 每个 task 启动时错峰（秒），降低同时请求概率；可设为 0 关闭
STARTUP_STAGGER_SEC="${STARTUP_STAGGER_SEC:-20}"

CONFIGS=(
  "pi05_lora_tacimg_tabero"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
if (( TASK_ID < 0 || TASK_ID >= ${#CONFIGS[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} 越界，期望范围是 0..$(( ${#CONFIGS[@]} - 1 ))"
  exit 1
fi

CONFIG_NAME="${CONFIGS[${TASK_ID}]}"
EXP_NAME="${CONFIG_NAME}"
if [[ -n "${RUN_TAG}" ]]; then
  EXP_NAME="${CONFIG_NAME}_${RUN_TAG}"
fi

echo "Running on host:  $(hostname)"
echo "Working dir:      $(pwd)"
echo "Slurm job id:     ${SLURM_JOB_ID:-N/A}"
echo "Array job id:     ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Array task id:    ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Config:           ${CONFIG_NAME}"
echo "Exp name:         ${EXP_NAME}"
echo "Overwrite:        ${OVERWRITE}"
echo "Resume:           ${RESUME}"
echo "Run norm stats:   ${RUN_NORM_STATS}"

export HYDRA_FULL_ERROR=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

nvidia-smi -L || true
nvidia-smi || true

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

# 错峰启动（对 array 并行很有用）
if [[ "${STARTUP_STAGGER_SEC}" != "0" ]]; then
  sleep "$(( TASK_ID * STARTUP_STAGGER_SEC ))"
fi

# 用 flock 在共享文件系统上做一个“HF 访问锁”，避免三个任务同时触发 HF 限流
HF_LOCK_FILE="${HF_LOCK_FILE:-${HF_HOME}/hf_download.lock}"

if [[ "${RUN_NORM_STATS}" == "1" ]]; then
  echo "===== compute_norm_stats: ${CONFIG_NAME} ====="
  srun --ntasks=1 bash -lc "cd '${PROJECT_DIR}' && flock -x '${HF_LOCK_FILE}' -c \"uv run scripts/compute_norm_stats.py --config-name '${CONFIG_NAME}'\""
else
  echo "===== skip compute_norm_stats (RUN_NORM_STATS=${RUN_NORM_STATS}) ====="
fi

echo "===== train: ${CONFIG_NAME} (exp-name=${EXP_NAME}) ====="
srun --ntasks=1 bash -lc "cd '${PROJECT_DIR}' && uv run scripts/train.py '${CONFIG_NAME}' --exp-name='${EXP_NAME}' ${resume_flag} ${overwrite_flag}"

echo "===== done: ${CONFIG_NAME} ====="
set +x


