import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from config import JianMindConfig
from layers import RMSNorm, precompute_freqs_cis, apply_rotary_pos_emb

# 激活函数映射
ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}

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
        self.hidden_dim = config.hidden_size
        self.hidden_layers = config.num_hidden_layers
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dropout = config.dropout
        self.rope_base = config.rope_base
        self.rope_scaling = config.rope_scaling
        self.flash_attention = hasattr(config, "flash_attention") and config.flash_attention

        assert self.hidden_dim % self.heads == 0, "hidden_dim must be divisible by num_attention_heads"
        self.head_dim = self.hidden_dim // self.heads  #每个头的维度
        self.num_kv_group = self.heads // self.kv_heads  #每组4个q 共享一组kv
        
        #注意这里维度的变化，q保持不变，为了更丰富的表达能力，kv变成了kv_heads * head_dim, 也就是每组4个q共享一组kv，减少了参数量
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
        q = self.wq(x).view(batch_size, seq_len, self.heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.kv_heads, self.head_dim)

        #添加位置编码
        freqs_cos, freqs_sin = pos_embedding
        q_rotated, k_rotated = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        #如果使用缓存，则将当前的 k 和 v 与过去的 k 和 v 拼接
        if past_kv is not None:
            past_k, past_v = past_kv
            k_rotated = torch.cat([past_k, k_rotated], dim=1)
            v = torch.cat([past_v, v], dim=1)  #这里 [cache_k, k] 创建了一个包含两个张量的列表
        if use_cache:
            past_kv = (k_rotated, v)  #更新缓存

        #重复 KV 张量以匹配 Q 的头数
        k_rotated = kv_repeat(k_rotated, self.num_kv_group)
        v = kv_repeat(v, self.num_kv_group)

        #计算注意力分数
        #交换一下头维度和序列维度，方便后续矩阵乘法
        q_rotated = q_rotated.transpose(1, 2)  # [batch_size, heads, seq_len, head_dim]
        k_rotated = k_rotated.transpose(1, 2)  # [batch_size, heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, heads, seq_len, head_dim]
        attn_scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1))/(math.sqrt(self.head_dim))

        #加入注意力掩码
        #采用加法，不需要掩盖的下三角为0，掩盖的上三角为一个很大的负数-inf，这样在softmax之后就会接近0
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask  #注意力掩码通常是一个很大的负数，用于屏蔽不需要关注的部分

        #softmax+dropout
        attn_scores = F.softmax(attn_scores,dim=-1)
        attn_scores = self.dropout_layer(attn_scores)

        #计算注意力输出
        attn_output = torch.matmul(attn_scores, v)
        #交换回头维度和序列维度，并将多头的输出拼接起来
        #contiguous() 确保内存连续，view() 重新调整形状为 [batch_size, seq_len, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        #输出线性变换
        output = self.wo(attn_output)
        return output, past_kv if use_cache else None
        
class FeedForward(nn.Module):
    def __init__(self, config: JianMindConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        # intermediate_size 如果没设，按 LLaMA 惯例取 8/3 倍
        if config.intermediate_size is None:
            self.intermediate_dim = int(2 * config.hidden_size * 4 / 3)
        else:
            self.intermediate_dim = config.intermediate_size
        self.dropout = nn.Dropout(config.dropout)
        #swiGLU激活函数曾
        self.gate_proj = nn.Linear(self.hidden_dim,self.intermediate_dim,bias=False)
        self.up_proj=nn.Linear(self.hidden_dim,self.intermediate_dim,bias=False)
        self.down_proj=nn.Linear(self.intermediate_dim,self.hidden_dim,bias=False)
        self.act = ACT2FN[config.hidden_act]   # "silu" → F.silu
    
    def forward(self,x:torch.Tensor):
        return self.dropout(self.down_proj(self.up_proj(x) * self.act((self.gate_proj(x)))))
    
class Transformer_block(nn.Module):
    def __init__ (self,config:JianMindConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.rms_norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)   
        self.rms_norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)   


    def forward(self, x: torch.Tensor, 
                pos_embedding: tuple[torch.Tensor, torch.Tensor], 
                past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None)->torch.Tensor:
        x1 = self.rms_norm1(x)
        h1 , cur_kv_cache = self.attention(x1,pos_embedding,past_kv,use_cache,attention_mask)
        x = x+h1
        x2 = self.rms_norm2(x)
        h2 = self.ffn(x2)
        x = x+h2
        return x , cur_kv_cache

class JianMind(nn.Module):
    def __init__ (self,config:JianMindConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.dim = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_base = config.rope_base
        self.rope_scaling = config.rope_scaling
        self.num_layer_trans = nn.ModuleList([
            Transformer_block(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size,config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size)
        self.lm_head.weight = self.tok_embeddings.weight   # ✅ lm_head 复用 embedding 的权重 自己训练词表对应的权重

    def forward(self,input_id:torch.Tensor,labels:torch.Tensor = None):
        batch_size , seq_len = input_id.shape
        x = self.tok_embeddings(input_id)
        x = self.dropout(x)
        pos_emb = precompute_freqs_cis(self.head_dim, seq_len, self.rope_base, self.rope_scaling)
        attention_mask = torch.triu(torch.ones(seq_len,seq_len,device=x.device) * float('-inf'),diagonal=1)
        for layer in self.num_layer_trans:
            x, _ = layer(x, pos_emb, None, False, attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return logits, loss
