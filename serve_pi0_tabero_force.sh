#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 一键启动 pi0_tabero_force 远程推理服务的脚本。
# 会从 Hugging Face 仓库拉取 checkpoint，并用 scripts/serve_policy.py 启动 WebSocket 服务。
#
# 用法示例（在工程根目录）：
#   1）默认使用 step=49999，端口 8000：
#        bash serve_pi0_tabero_force.sh
#   2）指定 step（例如 39999）：
#        bash serve_pi0_tabero_force.sh 39999
#   3）指定端口（例如 9000）：
#        PORT=9000 bash serve_pi0_tabero_force.sh 49999
###############################################################################

########################
# 可按需修改的变量
########################

# 工程根目录（自动取当前脚本所在目录）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "[INFO] 当前工程根目录: ${ROOT_DIR}"

# Hugging Face 模型仓库（你的 pi0_tabero_force）
HF_REPO_URL="https://huggingface.co/NathanWu7/pi0_tabero_force"

# 本地 clone 位置（可以按需修改）
HF_REPO_DIR="${HOME}/hf/pi0_tabero_force"

# 训练 config / 实验名
CONFIG_NAME="pi0_libero_force_low_mem_finetune"
EXP_NAME="force_test2"

# 想要使用的 checkpoint step，默认 49999。
# 可以通过第一个命令行参数覆盖：
#   bash serve_pi0_tabero_force.sh 39999
CKPT_STEP="${1:-49999}"

# 服务端口，可通过环境变量 PORT 覆盖，默认 8000
PORT="${PORT:-8000}"

########################
# 基本检查
########################

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] 没找到 git，请先安装 git。" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] 没找到 uv（uvx），请先安装 uv，并确保在 PATH 中。" >&2
  echo "        参考：https://github.com/astral-sh/uv" >&2
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[WARN] 未检测到 git-lfs，可能导致从 HF 拉取大文件失败。" >&2
  echo "       建议安装 git-lfs 后重新运行本脚本：" >&2
  echo "         sudo apt install git-lfs && git lfs install" >&2
fi

########################
# 准备 HF 本地仓库
########################

mkdir -p "$(dirname "${HF_REPO_DIR}")"

if [[ ! -d "${HF_REPO_DIR}/.git" ]]; then
  echo "[INFO] 本地不存在 ${HF_REPO_DIR}，开始从 HF clone"
  git lfs install --skip-repo || true
  git clone "${HF_REPO_URL}" "${HF_REPO_DIR}"
else
  echo "[INFO] 已存在本地仓库，执行 git pull 同步到最新"
  (
    cd "${HF_REPO_DIR}"
    git pull --ff-only || true
  )
fi

########################
# 拼出 checkpoint 路径并检查
########################

CKPT_DIR="${HF_REPO_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/${CKPT_STEP}"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "[ERROR] 找不到 checkpoint 目录: ${CKPT_DIR}" >&2
  echo "        请检查 step 是否存在（例如 49999 / 39999 等）。" >&2
  echo "        你可以用以下命令查看有哪些 step：" >&2
  echo "          ls \"${HF_REPO_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}\"" >&2
  exit 1
fi

echo "[INFO] 使用 checkpoint:"
echo "       ${CKPT_DIR}"
echo "[INFO] 将在端口 ${PORT} 上启动 WebSocket policy server"

########################
# 启动服务
########################

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config="${CONFIG_NAME}" \
  --policy.dir="${CKPT_DIR}" \
  --port="${PORT}"


