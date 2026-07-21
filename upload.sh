#!/bin/bash
# =============================================================
#  将项目上传到训练服务器
#  用法: bash upload.sh <服务器IP> [用户名] [SSH端口]
#  示例: bash upload.sh 192.168.1.100
#        bash upload.sh 192.168.1.100 root 22
# =============================================================

SERVER_IP="${1:?❌ 请提供服务器IP，例: bash upload.sh 192.168.1.100}"
USER="${2:-root}"
PORT="${3:-22}"
REMOTE_DIR="~/minimind"   # 服务器上的目标目录，~ 会自动解析为用户家目录

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  上传项目到服务器"
echo "  目标: ${USER}@${SERVER_IP}:${REMOTE_DIR}"
echo "=========================================="

# ---------- 1. 创建远程目录 ----------
echo ""
echo "📁 创建远程目录..."
ssh -p "$PORT" "${USER}@${SERVER_IP}" "mkdir -p ${REMOTE_DIR}"

# ---------- 2. 上传代码（排除数据集和缓存）----------
echo ""
echo "📦 上传代码文件（排除数据集）..."
rsync -avz --progress \
    -e "ssh -p ${PORT}" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='.vscode' \
    --exclude='out/' \
    --exclude='*.output' \
    --exclude='dataset/*.jsonl' \
    "${LOCAL_DIR}/" "${USER}@${SERVER_IP}:${REMOTE_DIR}/"

# ---------- 3. 上传数据集 ----------
echo ""
echo "📊 上传数据集（约 21GB，可能需要较长时间）..."
echo "   使用 rsync 支持断点续传，中断后重新运行即可继续"
echo ""

ssh -p "$PORT" "${USER}@${SERVER_IP}" "mkdir -p ${REMOTE_DIR}/dataset"

rsync -avz --progress \
    -e "ssh -p ${PORT}" \
    "${LOCAL_DIR}/dataset/" "${USER}@${SERVER_IP}:${REMOTE_DIR}/dataset/"

echo ""
echo "=========================================="
echo "  ✅ 上传完成！"
echo ""
echo "  后续步骤："
echo "  1. SSH 登录服务器:"
echo "     ssh -p ${PORT} ${USER}@${SERVER_IP}"
echo ""
echo "  2. 检查环境:"
echo "     cd ${REMOTE_DIR} && bash check_env.sh"
echo ""
echo "  3. 安装 Docker（如需要）:"
echo "     sudo bash setup_docker.sh"
echo ""
echo "  4. 构建镜像并启动训练:"
echo "     bash run_train.sh"
echo "=========================================="
