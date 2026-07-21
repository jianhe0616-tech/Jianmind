# =============================================================
#  JianMind / MiniMind 训练镜像
#  目标: 8× RTX 2080 Ti (22GB) | CUDA 13.0
# =============================================================

FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---------- 系统依赖 + Python 3.11（来自 Ubuntu 官方 PPA，无需 Conda）----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        wget curl git ca-certificates \
        build-essential \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ---------- pip（使用清华镜像加速）----------
RUN wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python3 /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# ---------- PyTorch (CUDA 13.0) ----------
# torchaudio 在 cu130 上最高 2.11.0，torch 对齐到 2.11.0
RUN pip install --no-cache-dir \
    torch==2.11.0+cu130 torchaudio==2.11.0+cu130 torchvision \
    --index-url https://download.pytorch.org/whl/cu130

# ---------- 项目依赖 ----------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---------- 工作目录 ----------
WORKDIR /workspace/minimind
COPY . /workspace/minimind/

# ---------- 验证 PyTorch 安装 ----------
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__} installed, CUDA build: {torch.version.cuda}')"

CMD ["bash"]
