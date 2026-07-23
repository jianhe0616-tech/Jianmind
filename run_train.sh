#!/bin/bash
# =============================================================
#  八卡 DDP 训练启动脚本
#  用法: bash run_train.sh
# =============================================================

IMAGE_NAME="minimind:latest"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 检查镜像是否存在
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "📦 镜像 $IMAGE_NAME 不存在，正在构建..."
    docker build -t "$IMAGE_NAME" "$PROJECT_DIR"
fi

echo "🚀 启动八卡 DDP 训练..."

docker run --rm \
    --gpus all \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$PROJECT_DIR":/workspace/minimind \
    -w /workspace/minimind \
    "$IMAGE_NAME" \
    torchrun --nproc_per_node=8 \
        --master_port=29500 \
        trainer/train_pretrain.py \
        --batch_size 40 \
        --epochs 2 \
        --dtype float32 \
        --max_seq_len 512 \
        --learning_rate 3e-4 \
        --grad_clip 0.5 \
        --accumulation_steps 1 \
        --num_workers 4 \
        --data_path dataset/pretrain_t2t.jsonl \
        --save_dir out \
        --tokenizer_path dataset \
        --hidden_size 768 \
        --num_hidden_layers 8 \
        --use_compile 0

echo "✅ 训练结束"
