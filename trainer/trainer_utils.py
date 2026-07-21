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
from transformers import AutoTokenizer
from model.model import JianMindForCausalLM
from torch.utils.data import Sampler

#打印报告，只有主进程打印
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
def Logger(content):
    if is_main_process():
        print(content)

#分布式初始化
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

#设置种子
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#断点续训
def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None,
                  epoch=0, step=0, wandb=None, 
                  save_dir='../checkpoints', **kwargs):
    
    #创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    #moe下 多加后缀_moe
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth' #权重文件，只存储模型参数（用于推理或继续训练）。
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth' #恢复文件，存储模型、优化器、epoch、step 等完整状态（用于断点续训）

    ###保存模式分支：
    if model is not None:
        #在 DDP 多卡训练时，模型被 DistributedDataParallel 包装。直接取 state_dict 会带 module. 前缀，破坏加载逻辑。这里通过 .module 剥掉这层包装。
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        #如果启用了 torch.compile，PyTorch 会在模型上挂一个 _orig_mod 属性指向原始模型。这行代码确保无论如何包装，都能拿到最底层的原始模型对象。
        raw_model = getattr(raw_model, '_orig_mod', raw_model)

        #提取权重参数
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        #原子保存文件，防止因磁盘不够保存失败还损失上一次权重文件
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        #提取wandb——id
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        #构建完整的恢复数据结构
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            #记录当前训练的 GPU 数量。这是为了应对“保存时用 8 卡，续训时只用 4 卡”的情况，后续加载时用来修正 step 进度（后面会讲到）
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        #处理额外参数 (**kwargs)
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):  #如果有state_dict，说明是模型参数，三层剥离
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        #原子写入恢复文件+内存清理
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()

    ###加载模式分支：   ckp:checkpoint   ws:world_size  the num of gpu
    else:  # 加载模式 
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu', weights_only=False)
        #动态适配 GPU 数量变化（智能 Step 转换） step:若有八卡，之前一次数据8*bs 现在四卡，4*bs
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data #返回恢复文件，包括权重梯度等等进行续训
        Logger(f"⚠️ 未找到检查点文件，从头训练")
        return None #

#创建模型
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../dataset', save_dir='../out', device='cuda'):
    #创建模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = JianMindForCausalLM(lm_config)
    #加载预训练权重
    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        missing, unexpected = model.load_state_dict(weights, strict=False) #允许权重文件中的键与模型不完全匹配。
        if missing:
            Logger(f'⚠️  加载权重时缺失的 key ({len(missing)}个): {missing[:5]}{"..." if len(missing)>5 else ""}')
        if unexpected:
            Logger(f'⚠️  加载权重时多余的 key ({len(unexpected)}个): {unexpected[:5]}{"..." if len(unexpected)>5 else ""}')

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer

#打印模型参数函数
def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6

    #moe
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')



#学习率（带 warmup）
def get_lr(current_step, total_steps, lr, warmup_steps=50):
    # Warmup 阶段：线性增长到 lr
    if current_step < warmup_steps:
        return lr * (current_step + 1) / warmup_steps
    # Cosine decay 阶段
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


#断点续训采样，续训时跳过已经训练过的样本
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler          # 原始采样器（或索引列表）
        self.batch_size = batch_size    # 每个批次的大小
        self.skip_batches = skip_batches # 要跳过的批次数

    def __iter__(self):
        batch = []          # 当前正在累积的批次
        skipped = 0         # 已经跳过了多少个批次
        for idx in self.sampler:   # 遍历原始采样器中的每个索引
            batch.append(idx)      # 把索引加入当前批次
            if len(batch) == self.batch_size:   # 攒满了一个批次
                if skipped < self.skip_batches:
                    skipped += 1       # 跳过计数 +1
                    batch = []         # 清空当前批次（丢弃）
                    continue           # 继续取下一个批次
                yield batch            # 跳过阶段结束，正常返回批次
                batch = []             # 清空，准备下一个批次
        # 处理最后剩余不足一个批次的数据
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self): #返回总批次数
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
    
#这个 yield 的作用： 节省内存，return是把所有数据处理好一次发送，占内存
# 当 SkipBatchSampler 被用于 DataLoader 时，DataLoader 会不断调用 __iter__ 生成器来获取批次。
# 每次遇到 yield batch，就把当前积攒好的一个批次抛给 DataLoader，然后函数暂停。
# DataLoader 拿到一个批次后，交给模型训练。
# 训练完这个批次，DataLoader 继续请求下一个批次 → 生成器从上次 yield 的地方恢复，继续执行。
# 如此循环，直到所有数据遍历完毕。