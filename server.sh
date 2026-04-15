#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 一键启动远程推理 WebSocket policy server 的脚本。
#
# 该脚本会：
# - 根据训练 config 名（通常也作为 exp_name）推导 Hugging Face 仓库与本地 clone 位置
# - 从 Hugging Face 拉取 checkpoints
# - 用 scripts/serve_policy.py 启动 WebSocket 服务
#
# 用法（在工程根目录）：
#   1）默认 step=49999，端口 8000：
#        bash server.sh pi0_lora_tacimg_tabero
#   2）指定 step：
#        bash server.sh pi0_lora_tacimg_tabero 39999
#   3）指定端口：
#        PORT=9000 bash server.sh pi0_lora_tacimg_tabero 49999
#
# 可选环境变量：
#   - HF_OWNER: HF 仓库 owner（默认 NathanWu7）
#   - HF_BASE_DIR: 本地 clone 基目录（默认 "${HOME}/hf"）
#   - EXP_NAME: 覆盖实验名（默认与 config 相同）
###############################################################################

usage() {
  local default_owner default_base
  default_owner="${HF_OWNER:-NathanWu7}"
  default_base="${HF_BASE_DIR:-$HOME/hf}"

  cat <<EOF
用法：
  bash server.sh <config_name> [ckpt_step]

示例：
  bash server.sh pi0_lora_tacimg_tabero 49999
  PORT=9000 bash server.sh pi0_lora_tacforce_tabero

常用 config（可在 src/openpi/training/config.py 里查看更多）：
  - pi0_lora_tacimg_tabero
  - pi0_lora_tacforce_tabero
  - pi0_lora_tacfield_tabero
  - pi0_lora_tacall_tabero

说明：
  - HF 仓库默认推导为：https://huggingface.co/${default_owner}/<config_name>
  - 本地 clone 默认目录：${default_base}/<config_name>
  - checkpoint 目录默认：<repo_dir>/checkpoints/<config_name>/<exp_name>/<ckpt_step>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${#}" -lt 1 ]]; then
  usage
  exit 0
fi

########################
# 参数 / 默认值
########################

CONFIG_NAME="${1}"
CKPT_STEP="${2:-49999}"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}}"
PORT="${PORT:-8000}"

HF_OWNER="${HF_OWNER:-NathanWu7}"
HF_BASE_DIR="${HF_BASE_DIR:-${HOME}/hf}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

HF_REPO_URL="https://huggingface.co/${HF_OWNER}/${CONFIG_NAME}"
HF_REPO_DIR="${HF_BASE_DIR}/${CONFIG_NAME}"

echo "[INFO] 当前工程根目录: ${ROOT_DIR}"
echo "[INFO] config: ${CONFIG_NAME}"
echo "[INFO] exp   : ${EXP_NAME}"
echo "[INFO] step  : ${CKPT_STEP}"
echo "[INFO] port  : ${PORT}"
echo "[INFO] HF repo: ${HF_REPO_URL}"
echo "[INFO] local : ${HF_REPO_DIR}"

########################
# 基本检查
########################

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] 没找到 git，请先安装 git。" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] 没找到 uv（uvx），请先安装 uv，并确保在 PATH 中。" >&2
  echo "        参考：\`https://github.com/astral-sh/uv\`" >&2
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
  git lfs install --skip-repo >/dev/null 2>&1 || true
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
  echo "        可能原因：" >&2
  echo "        - step 不存在（例如 49999 / 39999 等）" >&2
  echo "        - EXP_NAME 不匹配（可用环境变量 EXP_NAME 覆盖）" >&2
  echo "        你可以用以下命令查看有哪些 step：" >&2
  echo "          ls \"${HF_REPO_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}\"" >&2
  exit 1
fi

echo "[INFO] 使用 checkpoint:"
echo "       ${CKPT_DIR}"
echo "[INFO] 启动 WebSocket policy server..."

########################
# 启动服务
########################

# Workaround for `uv` installing deps (e.g. `lerobot`) from GitHub with git-lfs:
# some upstream LFS objects may be missing and break checkout.
# These artifacts are not needed for serving, so we skip LFS smudge by default.
: "${GIT_LFS_SKIP_SMUDGE:=1}"
export GIT_LFS_SKIP_SMUDGE

uv run scripts/serve_policy.py \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config="${CONFIG_NAME}" \
  --policy.dir="${CKPT_DIR}"


