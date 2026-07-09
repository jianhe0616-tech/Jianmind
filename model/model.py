import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from config import JianMindConfig
from layers import RMSNorm, precompute_freqs_cis, apply_rotary_pos_emb

def kv_repeat(x, num_group):
    """
    重复 KV 张量以匹配 Q 的头数。
    假设输入张量 x 的形状为 [batch_size, seq_len, num_kv_heads, head_dim]，
    输出张量的形状为 [batch_size, seq_len, num_attention_heads, head_dim]，
    其中 num_attention_heads = num_kv_heads * num_group。
    """
    if num_group == 1:
        return x  # 如果不需要重复，直接返回原始张量
    # 先在第2维度上重复 num_group 次，然后在第2维度上重新排列,
    # repeat_interleave 会将每个元素重复 num_group 次，最终形状为 [batch_size, seq_len, num_kv_heads * num_group, head_dim]
    return x.repeat_interleave(num_group, dim=2)


class MultiHeadAttention(nn.Module):
    def __init__ (self, config: JianMindConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.hidden_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.hidden_act = config.hidden_act
        self.dropout = config.dropout
        self.flash_attention = hasattr(config, "flash_attention") and config.flash_attention

        assert self.hidden_dim % self.heads == 0, "hidden_dim must be divisible by num_attention_heads"
        self.head_dim = self.hidden_dim // self.heads  #每个头的维度
        self.num_group = self.heads // self.kv_heads  #每组4个q 共享一组kv
        
        self.wq = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wk = nn.Linear(self.hidden_dim, self.kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_dim, self.kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor, 
                pos_embedding: tuple[torch.Tensor, torch.Tensor], 
                past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        q = self.wq(x).view(batch_size, self.heads, self.head_dim, seq_len).transpose(2, 3)  # [batch_size, num_attention_heads, seq_len, head_dim]
        k = self.wk(x).view(batch_size, self.kv_heads * self.num_group, self.head_dim, seq_len).transpose(2, 3)
        v = self.wv(x).view(batch_size, self.kv_heads * self.num_group, seq_len, self.head_dim)
        