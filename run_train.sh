#!/bin/bash
# =============================================================
#  训练启动脚本（适配 RTX 4090 / A100，支持 bf16）
#  用法: bash run_train.sh
# =============================================================

IMAGE_NAME="minimind:latest"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 检查镜像是否存在
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "📦 镜像 $IMAGE_NAME 不存在，正在构建..."
    docker build -t "$IMAGE_NAME" "$PROJECT_DIR"
fi

echo "🚀 启动训练（单卡）..."

docker run --rm \
    --gpus all \
    --shm-size=16g \
    --ipc=host \
    -v "$PROJECT_DIR":/workspace/minimind \
    -w /workspace/minimind \
    "$IMAGE_NAME" \
    python3 trainer/train_pretrain.py \
        --batch_size 64 \
        --epochs 2 \
        --dtype bfloat16 \
        --max_seq_len 512 \
        --learning_rate 5e-4 \
        --grad_clip 1.0 \
        --accumulation_steps 1 \
        --num_workers 8 \
        --data_path dataset/pretrain_t2t.jsonl \
        --save_dir out \
        --tokenizer_path dataset \
        --hidden_size 768 \
        --num_hidden_layers 8 \
        --use_compile 0

echo "✅ 训练结束"
