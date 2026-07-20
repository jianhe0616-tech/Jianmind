# 训练超参数 training hyperparameters
import argparse
import os
import torch


def get_train_config():
    """获取训练配置参数"""
    parser = argparse.ArgumentParser(description="JianMind Pretraining")
    
    # ====== 训练控制参数 ======
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=160, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    
    # ====== 模型结构参数（可覆盖config默认值） ======
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    
    # ====== 数据参数 ======
    parser.add_argument('--max_seq_len', default=512, type=int, help="最大序列长度")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t.jsonl", help="数据路径")
    
    # ====== 硬件参数 ======
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="float16", help="混合精度类型(bf16需Ampere及以上,2080ti等Turing卡请用float16)")
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 4), help="数据加载线程数")
    
    # ====== 日志与保存 ======
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    
    # ====== 恢复与续训 ======
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    
    # ====== 其他 ======
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    
    return parser.parse_args()