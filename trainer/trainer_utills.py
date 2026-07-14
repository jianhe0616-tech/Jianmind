"""
训练工具函数集合
train_pretrain.py
    ├── init_distributed_mode()      ← 分布式初始化
    ├── setup_seed()                 ← 固定随机种子
    ├── lm_checkpoint() (加载)       ← 断点续训
    ├── init_model()                 ← 创建模型
    │   └── get_model_params()       ← 打印参数量
    ├── train_epoch()
    │   ├── get_lr()                 ← 更新学习率
    │   └── lm_checkpoint() (保存)   ← 定期保存
    └── SkipBatchSampler()           ← 数据采样
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model import JianMindForCausalLM

#分布式初始化
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

#学习率
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

#打印报告，只有主进程打印
def is_main_process():
    return not dist.i
def Logger(content):
    if is_main_process():
        print(content)

#设置种子
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

