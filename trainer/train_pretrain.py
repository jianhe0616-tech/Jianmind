import time
import os
import sys
import warnings

# ========== 1. 声明当前脚本所属的包 ==========
# 作用：让 Python 知道这个脚本是 "trainer" 包的一部分
# 为什么需要：直接运行脚本时（如 python train_pretrain.py），
# 如果不声明，Python 可能无法正确识别相对导入（如 from ..xxx import yyy），
# 导致 ModuleNotFoundError
__package__ = "trainer"

# ========== 2. 将项目根目录添加到 Python 模块搜索路径 ==========
# 作用：让 Python 能导入项目根目录下的其他模块（如 model/、dataset/）
#   - __file__: 当前脚本的完整路径，如 /path/to/minimind/trainer/train_pretrain.py
#   - os.path.dirname(__file__): 获取脚本所在目录，如 /path/to/minimind/trainer/
#   - os.path.join(..., '..'): 拼接上一级目录，得到 /path/to/minimind/trainer/.. => /path/to/minimind/
#   - os.path.abspath(...): 规范化为绝对路径，得到 /path/to/minimind
#   - sys.path.append(...): 将项目根目录添加到 Python 搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.distributed as dist #分布式通信库，用于多卡训练时的同步
from contextlib import nullcontext
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler #DataLoader 用来加载数据批次；DistributedSampler 在多卡时用于数据分片。
from torch.nn.parallel import DistributedDataParallel# 多卡训练的模型包装器。单卡时不会用到。

from train_config import get_train_config
from dataset.lm_dataset import PretrainDataset
from model.model import JianMindConfig
from model.model import JianMindForCausalLM
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
args = get_train_config()

def train_epoch(
    epoch: int,
    loader: DataLoader,
    iterations: int,
    optimizer: Optimizer,  # ← 显式声明为 Optimizer 类型
    start_step: int = 0,
    wandb=None
) :
    # ---- 1. 初始化计时器 ----
    start_time = time.time()                    # 记录epoch开始的时间戳
    last_step = start_step                      # 记录最近执行的步数（用于计算ETA）
    # ---- 2. 遍历数据加载器 ----
    # enumerate(loader, start=start_step+1) 让step从断点处继续编号
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # ---- 3. 将数据移到指定设备（GPU/CPU） ----
        input_ids = input_ids.to(args.device)   # 输入token ID → GPU
        labels = labels.to(args.device)         # 标签token ID → GPU
        last_step = step                        # 更新当前步数
        grad_norm = 0.0                         # 梯度范数
        # ---- 4. 动态更新学习率（warmup + 余弦退火） ----
        # 前 50 步 warmup，防止初始梯度爆炸污染 AdamW 动量
        lr = get_lr(
            epoch * iterations + step,          # 当前全局步数
            args.epochs * iterations,           # 训练总步数
            args.learning_rate,                 # 初始学习率
            warmup_steps=50                     # warmup 步数
        )
        # 将计算出的学习率赋给优化器的所有参数组
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # ---- 5. 前向传播 ----
        with autocast_ctx:
            # ✅ 温度缩放：logits 除以 temperature，防止过大值导致 NaN
            # 初始 temperature=1.0，随着 logits_max 增长可以增加到 2.0
            temperature = 1.0 + 0.5 * (step / iterations)  # 从 1.0 线性增长到 1.5
            res = model(input_ids, labels=labels, temperature=temperature)
            loss = res.loss
        # 梯度累积：loss 除以累积步数
        if args.accumulation_steps > 1:
            loss = loss / args.accumulation_steps

        # ---- 6. 反向传播 ----
        loss.backward()

        # === 诊断：前 10 步 + NaN 检测 ===
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                break
        if (step <= 10 or has_nan_grad) and is_main_process():
            loss_val = loss.item() * args.accumulation_steps
            logits_max = res.logits.abs().max().item()
            if has_nan_grad:
                nan_count = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
                Logger(f'[step={step}] ⚠️ loss={loss_val:.4f} logits_max={logits_max:.2f} NaN梯度={nan_count}个 跳过更新')
            else:
                raw_grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
                Logger(f'[step={step}] loss={loss_val:.4f} logits_max={logits_max:.2f} raw_grad={raw_grad_norm:.2f} lr={lr:.6f}')

        # ---- 7. 梯度累积：达到累积步数时更新参数 ----
        if step % args.accumulation_steps == 0:
            if has_nan_grad:
                # NaN 梯度 → 跳过参数更新，保护 AdamW 内部状态不被污染
                if is_main_process():
                    Logger(f'[step={step}] ⚠️ 检测到 NaN 梯度，跳过更新')
                optimizer.zero_grad(set_to_none=True)
            else:
                # 7a. 梯度裁剪（防止梯度爆炸），同时记录梯度范数
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip
                ).item()
                # 7b. 更新模型参数
                optimizer.step()
                # 7c. 清零梯度
                optimizer.zero_grad(set_to_none=True)
        # ---- 8. 日志记录（每log_interval步打印一次） ----
        if step % args.log_interval == 0 or step == iterations:
            # 8a. 计算已消耗时间
            spend_time = time.time() - start_time
            # 8b. 恢复真实损失（乘以累积步数）
            current_loss = loss.item() * args.accumulation_steps  
            # 8c. 提取辅助损失（MoE专用，非MoE为0）
            current_aux_loss = (
                0.0
                # res.aux_loss.item() 
                # if res.aux_loss is not None 
                # else 0.0
            )
            # 8d. 主损失 = 总损失 - 辅助损失
            current_logits_loss = current_loss - current_aux_loss
            # 8e. 获取当前学习率
            current_lr = optimizer.param_groups[-1]['lr']
            # 8f. 估算剩余时间（分钟）
            # 公式：平均每步耗时 × 剩余步数 / 60
            eta_min = (
                spend_time / max(step - start_step, 1)  # 平均每步耗时
                * (iterations - step)                   # 剩余步数
                // 60                                   # 转为分钟
            )
            # 8g. 打印训练指标
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]'
                f'({step}/{iterations}), '
                f'loss: {current_loss:.4f}, '
                f'logits_loss: {current_logits_loss:.4f}, '
                f'lr: {current_lr:.8f}, '
                f'grad_norm: {grad_norm:.4f}, '
                f'epoch_time: {eta_min:.1f}min'
            )
            
            # 8h. 如果启用了wandb，记录训练曲线
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                  #  "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # ---- 9. 保存模型检查点（每save_interval步保存一次） ----
        if (step % args.save_interval == 0 or step == iterations) and is_main_process():
            # 9a. 切换到评估模式（保存时禁用Dropout等）
            model.eval()
            # 9b. 构造文件名（MoE模型加_moe后缀）
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = (
                f'{args.save_dir}/'
                f'{args.save_weight}_'
                f'{lm_config.hidden_size}'
                f'{moe_suffix}.pth'
            )
            # 9c. 剥掉DDP和torch.compile的包装层，得到纯净模型
            raw_model = (
                model.module 
                if isinstance(model, DistributedDataParallel) 
                else model
            )
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 9d. 提取权重，转为float16并移到CPU（减小文件体积）
            state_dict = raw_model.state_dict()
            state_dict = {
                k: v.half().cpu() 
                for k, v in state_dict.items()
            }
            # 9e. 保存权重文件
            torch.save(state_dict, ckp)
            # 9f. 保存完整恢复文件（含optimizer、scaler、epoch、step等）
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=args.save_dir
            )

            # 9g. 切回训练模式
            model.train()
            # 9h. 手动释放内存
            del state_dict
        # ---- 10. 清理当前step的变量（释放显存/内存） ----
        del input_ids, labels, res, loss
    # ---- 11. 处理最后不足一个累积步数的残余梯度 ----
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
        # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config =JianMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.save_dir) if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    if args.dtype == "float32":
        # fp32 全精度：不需要 autocast，所有计算在 fp32 下进行
        autocast_ctx = nullcontext()
        Logger('使用 float32 全精度训练（推荐 2080Ti 等 Turing 卡）')
    elif args.dtype == "bfloat16":
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
        Logger('使用 bfloat16 混合精度（需要 Ampere+ GPU）')
    else:
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        Logger('使用 float16 混合精度')
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    Logger(f'batch_size={args.batch_size} dtype={args.dtype} '
           f'accumulation={args.accumulation_steps} grad_clip={args.grad_clip}')
    
    # ========== 8. 开始训练 ==========
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # DDP 下 start_step 是全局步数，需转为每卡的批次数
        skip = (start_step // world_size) if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{skip}个batch（全局step {start_step}），从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, optimizer, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), optimizer, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        
        


