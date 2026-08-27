#!/usr/bin/env bash
# 本机训练脚本（不改仓库 train.sh）
# 数据：xiangxin0923/realworld_replayed_task820
#   img   → pi05_lora_tacimg_realworld_replayed_task820
#   field → pi05_lora_tacfield_realworld_replayed_task820
#
# 用法：
#   source /data/home/chenxiangyu/xiangxin/T2-VLA/train_xiangxin.sh
#   wandb_key
#   /data/home/chenxiangyu/xiangxin/T2-VLA/train_xiangxin.sh img 6
#   /data/home/chenxiangyu/xiangxin/T2-VLA/train_xiangxin.sh field 7
#
# 可选环境变量：
#   BATCH_SIZE=16
#   NUM_TRAIN_STEPS=30000
#   SKIP_NORM=1
#   OVERWRITE=1
#   RESUME=1

wandb_key() {
  local t2vla_wandb_key
  read -r -s -p "W&B API key: " t2vla_wandb_key
  printf "\n"
  export WANDB_API_KEY="${t2vla_wandb_key}"
  unset t2vla_wandb_key
  echo "W&B API key loaded in this tmux pane."
}

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "wandb_key is defined. Next: wandb_key"
  return 0
fi

set -euo pipefail

ROOT="/data/home/chenxiangyu/xiangxin"
T2VLA="${ROOT}/T2-VLA"
DATA="${ROOT}/dataset"

export HF_LEROBOT_HOME="${DATA}"
export OPENPI_DATA_HOME="${DATA}/openpi"
export ASSETS_BASE_DIR="${DATA}/assets"
export CHECKPOINT_BASE_DIR="${DATA}/checkpoints"
export T2VLA_WANDB_PROJECT="${T2VLA_WANDB_PROJECT:-t2-vla-tabero-xiangxin}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"
export LD_LIBRARY_PATH="${T2VLA}/.deps/ffmpeg/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

CLASH_SH="${HOME}/clash/clash.sh"
if [[ -f "${CLASH_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CLASH_SH}" open
  # shellcheck disable=SC1090
  source "${CLASH_SH}" on
  export GOOGLE_CLOUD_DISABLE_GRPC=true
fi

mkdir -p "${OPENPI_DATA_HOME}" "${ASSETS_BASE_DIR}" "${CHECKPOINT_BASE_DIR}"

case "${1:-}" in
  img|tacimg|sim)
    CONFIG_NAME="pi05_lora_tacimg_realworld_replayed_task820"
    REPO_ID="xiangxin0923/realworld_replayed_task820"
    ;;
  field|tacfield)
    CONFIG_NAME="pi05_lora_tacfield_realworld_replayed_task820"
    REPO_ID="xiangxin0923/realworld_replayed_task820"
    ;;
  *)
    echo "Usage: $0 img|field [CUDA_VISIBLE_DEVICES]"
    exit 1
    ;;
esac

if [[ -n "${2:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$2"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Must specify one GPU, e.g. $0 img 6"
  echo "Free GPUs right now should be passed explicitly."
  exit 1
fi

EXP_NAME="${EXP_NAME:-${CONFIG_NAME}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
OVERWRITE="${OVERWRITE:-1}"
RESUME="${RESUME:-0}"
SKIP_NORM="${SKIP_NORM:-0}"

cd "${T2VLA}"

echo "==== T2-VLA train_xiangxin ===="
echo "config:              ${CONFIG_NAME}"
echo "repo_id:             ${REPO_ID}"
echo "exp-name:            ${EXP_NAME}"
echo "batch-size:          ${BATCH_SIZE}"
echo "num-train-steps:     ${NUM_TRAIN_STEPS}"
echo "HF_LEROBOT_HOME:     ${HF_LEROBOT_HOME}"
echo "OPENPI_DATA_HOME:    ${OPENPI_DATA_HOME}"
echo "ASSETS_BASE_DIR:     ${ASSETS_BASE_DIR}"
echo "CHECKPOINT_BASE_DIR: ${CHECKPOINT_BASE_DIR}"
echo "project-name:        ${T2VLA_WANDB_PROJECT}"
echo "WANDB_MODE:          ${WANDB_MODE}"
echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES:-<all>}"
echo "XLA_MEM_FRACTION:    ${XLA_PYTHON_CLIENT_MEM_FRACTION}"

if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "W&B API key is loaded"
else
  echo "W&B API key is NOT loaded"
  echo "Do not wandb login on this server."
  echo "In this pane run:"
  echo "  source ${T2VLA}/train_xiangxin.sh"
  echo "  wandb_key"
  echo "then rerun this script."
  exit 1
fi

if [[ "${RESUME}" == "1" && "${OVERWRITE}" == "1" ]]; then
  echo "RESUME=1 and OVERWRITE=1 cannot be used together."
  exit 1
fi

overwrite_flag=""
resume_flag=""
if [[ "${RESUME}" == "1" ]]; then
  resume_flag="--resume"
elif [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_flag="--overwrite"
fi

if [[ "${SKIP_NORM}" != "1" ]]; then
  echo "===== compute_norm_stats ====="
  uv run scripts/compute_norm_stats.py \
    --config-name "${CONFIG_NAME}" \
    --assets-base-dir "${ASSETS_BASE_DIR}"
else
  echo "===== skip compute_norm_stats ====="
fi

echo "===== train ====="
uv run scripts/train.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --project-name="${T2VLA_WANDB_PROJECT}" \
  --assets-base-dir="${ASSETS_BASE_DIR}" \
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
  --batch-size="${BATCH_SIZE}" \
  --num-train-steps="${NUM_TRAIN_STEPS}" \
  ${overwrite_flag} \
  ${resume_flag}

echo "===== done ====="
echo "checkpoints: ${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"
echo "openpi weights cache: ${OPENPI_DATA_HOME}"
