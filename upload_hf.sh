#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 把指定训练实验的 checkpoint（仅保留指定 step）和 norm_stats
# 一次性打包并上传到同一个 Hugging Face 仓库。
#
# 使用前准备：
#   1）已经跑完训练：checkpoints/<CONFIG_NAME>/<EXP_NAME>/...
#   2）已经跑完 norm_stats 统计：
#        uv run scripts/compute_norm_stats.py --config-name <CONFIG_NAME>
#      会生成：
#        assets/<CONFIG_NAME>/<DATA_REPO_ID>/...
#   3）已经登录 HF：
#        huggingface-cli login
#      说明：huggingface-cli 登录不一定能让 git push 自动拿到凭据。
#      最简方案：在运行脚本前导出 HF_TOKEN（Hugging Face 的访问 token），脚本会用它配置 git remote：
#        export HF_TOKEN=xxxxxxxx
#
# 运行方式（在工程根目录）：
#   bash upload_pi0_tabero_force_to_hf.sh
###############################################################################

########################
# 可按需修改的变量
########################

# HF 仓库 id（模型 + norm_stats 都放这里）
HF_REPO_ID="NathanWu7/pi0_lora_notac_bin_tabero"
HF_REPO_TYPE="model"   # 你也可以改成 "dataset"

# 训练 config 名 + 实验名
CONFIG_NAME="pi0_lora_notac_bin_tabero"
EXP_NAME="pi0_lora_notac_bin_tabero"

# 想要上传/同步到 HF 的 checkpoint step（子目录名）。
# 例：UPLOAD_STEPS=("49999")
# 为空则导出整个实验目录（不推荐）。
UPLOAD_STEPS=("30000" "49999")

# 训练 / 统计时用到的 repo_id（HF 数据集）
DATA_REPO_ID="NathanWu7/tabero_binary"

# 本地导出目录（脚本会自动创建/覆盖）
EXPORT_DIR="export/pi0_lora_notac_bin_tabero"

# 上传成功后，是否清理本地 CKPT_SRC 下除 LOCAL_KEEP_STEPS 以外的 step（不可逆！）
PRUNE_LOCAL_AFTER_UPLOAD="true"

# 本地想保留的 step（可以包含你不想上传的 step，比如 30000）
# 为空则不进行本地按 step 清理（即使 PRUNE_LOCAL_AFTER_UPLOAD=true）
LOCAL_KEEP_STEPS=("30000" "49999")

########################
# 脚本开始
########################

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "[INFO] 当前工程根目录: ${ROOT_DIR}"

if ! uv run huggingface-cli env >/dev/null 2>&1; then
  echo "[ERROR] 没找到 huggingface-cli，请先安装：uv pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] 没找到 git，请先安装 git。" >&2
  exit 1
fi

CKPT_SRC="${ROOT_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
NORM_SRC="${ROOT_DIR}/assets/${CONFIG_NAME}/${DATA_REPO_ID}"

if [[ ! -d "${CKPT_SRC}" ]]; then
  echo "[ERROR] 找不到 checkpoint 实验目录: ${CKPT_SRC}" >&2
  echo "        请先确认训练已完成，路径和 CONFIG_NAME/EXP_NAME 是否一致。" >&2
  exit 1
fi

if [[ ! -d "${NORM_SRC}" ]]; then
  echo "[ERROR] 找不到 norm_stats 目录: ${NORM_SRC}" >&2
  echo "        请先运行：uv run scripts/compute_norm_stats.py ${CONFIG_NAME}" >&2
  exit 1
fi

if (( ${#UPLOAD_STEPS[@]} > 0 )); then
  echo "[INFO] 将仅导出并上传以下 checkpoints step："
  for s in "${UPLOAD_STEPS[@]}"; do
    if [[ ! -d "${CKPT_SRC}/${s}" ]]; then
      echo "[ERROR] 找不到指定 step 的 checkpoint 目录: ${CKPT_SRC}/${s}" >&2
      echo "        请检查 UPLOAD_STEPS 配置，或用 ls ${CKPT_SRC} 查看有哪些 step。" >&2
      exit 1
    fi
    echo "       - ${s}"
  done
else
  echo "[INFO] 将导出整个实验目录下的所有 checkpoint："
  echo "       Checkpoints: ${CKPT_SRC}"
fi
echo "       Norm stats : ${NORM_SRC}"

########################################
# 创建 / 更新 HF 仓库并同步到本地
########################################

echo "[INFO] 确保 Hugging Face 仓库存在: ${HF_REPO_ID}"
uv run huggingface-cli repo create "${HF_REPO_ID}" --type "${HF_REPO_TYPE}" --yes || true

echo "[INFO] 清理并从 Hugging Face 仓库 clone 到本地导出目录: ${EXPORT_DIR}"
rm -rf "${EXPORT_DIR}"
git clone "https://huggingface.co/${HF_REPO_ID}" "${EXPORT_DIR}"

cd "${EXPORT_DIR}"

# 如果提供了 HF_TOKEN，则将 remote url 设置为带 token 的 https，避免 git push 401/askpass 交互失败。
if [[ -n "${HF_TOKEN:-}" ]]; then
  git remote set-url origin "https://user:${HF_TOKEN}@huggingface.co/${HF_REPO_ID}"
fi

#----------------------------------------
# 首次运行时自动为 checkpoints / norm_stats 启用 Git LFS
#----------------------------------------
if command -v git-lfs >/dev/null 2>&1; then
  git lfs install >/dev/null 2>&1 || true

  # 为大文件目录添加 LFS 规则（幂等，多次运行不会重复追加）
  if ! grep -q "checkpoints/" .gitattributes 2>/dev/null; then
    git lfs track "checkpoints/**"
  fi
  if ! grep -q "norm_stats/" .gitattributes 2>/dev/null; then
    git lfs track "norm_stats/**"
  fi

  # 确保 .gitattributes 被纳入版本控制
  git add .gitattributes || true
else
  echo "[WARN] 未找到 git-lfs，checkpoint 和 norm_stats 将不会通过 LFS 管理，" \
       "后续 git push 可能会非常慢或失败（建议安装 git-lfs）。" >&2
fi

# 组织成清晰的目录层级，方便以后在代码里引用
CKPT_DST="checkpoints/${CONFIG_NAME}"
NORM_DST="norm_stats/${CONFIG_NAME}"

mkdir -p "${CKPT_DST}" "${NORM_DST}"

echo "[INFO] 拷贝 checkpoint -> ${CKPT_DST}"

EXP_DST="${CKPT_DST}/${EXP_NAME}"
echo "[INFO] 为确保 HF 仓库中只保留期望的 step，将重建目录: ${EXP_DST}"
rm -rf "${EXP_DST}"
mkdir -p "${EXP_DST}"

if (( ${#UPLOAD_STEPS[@]} > 0 )); then
  for s in "${UPLOAD_STEPS[@]}"; do
    echo "[INFO] 复制 step=${s}"
    cp -r "${CKPT_SRC}/${s}" "${EXP_DST}/"
  done
  # 如果存在 wandb_id.txt，也一并复制，方便恢复 run
  if [[ -f "${CKPT_SRC}/wandb_id.txt" ]]; then
    cp "${CKPT_SRC}/wandb_id.txt" "${EXP_DST}/"
  fi
else
  # 复制整个实验目录（包括所有 step 和 wandb_id.txt）
  cp -r "${CKPT_SRC}/"* "${EXP_DST}/"
fi

echo "[INFO] 拷贝 norm_stats -> ${NORM_DST}"
cp -r "${NORM_SRC}" "${NORM_DST}/"

# 生成一个简单的 README，方便在 HF 页面查看信息
README_PATH="README.md"
cat > "${README_PATH}" <<EOF
# ${CONFIG_NAME} / ${EXP_NAME}

本仓库包含：

- 模型 checkpoint：
  - 路径：\`checkpoints/${CONFIG_NAME}/${EXP_NAME}/...\`
- 归一化统计（norm_stats）：
  - 路径：\`norm_stats/${CONFIG_NAME}/${DATA_REPO_ID}/...\`

训练配置基于 \`${CONFIG_NAME}\`，数据集为 Hugging Face 上的
\`${DATA_REPO_ID}\`（LeRobot 格式）。

推理时的典型用法示例（伪代码）：

- checkpoint 加载路径示例：
  - \`/path/to/clone/pi0_tabero_force/checkpoints/${CONFIG_NAME}/${EXP_NAME}/<step>/params\`
- norm_stats 加载路径示例（AssetsConfig）：
  - \`assets_dir="/path/to/clone/pi0_tabero_force/norm_stats/${CONFIG_NAME}"\`
  - \`asset_id="${DATA_REPO_ID}"\`
EOF

echo "[INFO] 已生成导出目录结构："
find "." -maxdepth 3 -type d | sed "s#^#  - #"

echo "[INFO] 提交并推送到 Hugging Face"
git add .
if git commit -m "Update checkpoints and norm_stats" 2>/dev/null; then
  echo "[INFO] 本地 commit 完成"
else
  echo "[INFO] 没有新的改动需要提交（可能和上次内容一致）"
fi

git push -u origin main

echo "[SUCCESS] 已将 checkpoint 和 norm_stats 上传到: https://huggingface.co/${HF_REPO_ID}"

########################################
# （可选）清理本地多余 checkpoints step
########################################
if [[ "${PRUNE_LOCAL_AFTER_UPLOAD}" == "true" ]] && (( ${#LOCAL_KEEP_STEPS[@]} > 0 )); then
  echo "[INFO] 开始清理本地多余 step（保留：${LOCAL_KEEP_STEPS[*]}）: ${CKPT_SRC}"
  shopt -s nullglob
  for d in "${CKPT_SRC}/"*; do
    bn="$(basename "${d}")"
    # 跳过非目录和 wandb 文件
    [[ -d "${d}" ]] || continue
    keep="false"
    for s in "${LOCAL_KEEP_STEPS[@]}"; do
      if [[ "${bn}" == "${s}" ]]; then
        keep="true"
        break
      fi
    done
    if [[ "${keep}" == "false" ]]; then
      echo "[INFO] 删除本地 step: ${d}"
      rm -rf "${d}"
    fi
  done
  shopt -u nullglob
  echo "[SUCCESS] 本地清理完成：${CKPT_SRC}"
fi


