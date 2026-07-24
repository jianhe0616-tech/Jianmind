import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional , Union
from .config import JianMindConfig
from .layers import RMSNorm, precompute_freqs_cis, apply_rotary_pos_emb
from transformers import PreTrainedModel ,GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

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
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.attention_dropout = config.dropout

        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.head_dim = self.hidden_size // self.num_heads  #每个头的维度
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  #每组2个q 共享一组kv

        #注意这里维度的变化，q保持不变，为了更丰富的表达能力，kv变成了num_key_value_heads * head_dim, 也就是每组4个q共享一组kv，减少了参数量
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, rms_norm_eps=config.rms_norm_eps)      # QK 归一化
        self.k_norm = RMSNorm(self.head_dim, rms_norm_eps=config.rms_norm_eps)      # QK 归一化
        self.attention_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor,
                pos_embedding: tuple[torch.Tensor, torch.Tensor],
                past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # QK 归一化（LLaMA3 风格：RoPE 之前对 Q 和 K 做 RMSNorm）
        q = self.q_norm(q)
        k = self.k_norm(k)

        #添加位置编码
        freqs_cos, freqs_sin = pos_embedding
        q_rotated, k_rotated = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        #如果使用缓存，则将当前的 k 和 v 与过去的 k 和 v 拼接
        #可以改进，新的dynamic cache框架可以原地拼接优化峰值
        if past_kv is not None:
            past_k, past_v = past_kv
            k_rotated = torch.cat([past_k, k_rotated], dim=1)
            v = torch.cat([past_v, v], dim=1)  #这里 [cache_k, k] 创建了一个包含两个张量的列表
        if use_cache:
            past_kv = (k_rotated, v)  #更新缓存

        #重复 KV 张量以匹配 Q 的头数
        k_rotated = kv_repeat(k_rotated, self.num_key_value_groups)
        v = kv_repeat(v, self.num_key_value_groups)

        #计算注意力分数
        #交换一下头维度和序列维度，方便后续矩阵乘法
        q_rotated = q_rotated.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k_rotated = k_rotated.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        attn_scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1))/(math.sqrt(self.head_dim))

        #加入注意力掩码
        #采用加法，不需要掩盖的下三角为0，掩盖的上三角为一个很大的负数-inf，这样在softmax之后就会接近0
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask  #注意力掩码通常是一个很大的负数，用于屏蔽不需要关注的部分

        #softmax+dropout（数值稳定版：减去最大值防止 exp 溢出）
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.attention_dropout(attn_scores)

        #计算注意力输出
        attn_output = torch.matmul(attn_scores, v)
        #交换回头维度和序列维度，并将多头的输出拼接起来
        #contiguous() 确保内存连续，view() 重新调整形状为 [batch_size, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        #输出线性变换
        output = self.o_proj(attn_output)
        return output, past_kv if use_cache else None
        
class FeedForward(nn.Module):
    def __init__(self, config: JianMindConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        # intermediate_size 如果没设，按 LLaMA 惯例取 8/3 倍
        if config.intermediate_size is None:
            self.intermediate_size = int(2 * config.hidden_size * 4 / 3)
        else:
            self.intermediate_size = config.intermediate_size
        self.dropout = nn.Dropout(config.dropout)
        #swiGLU激活函数
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]   # "silu" → F.silu

    def forward(self, x: torch.Tensor):
        return self.dropout(self.down_proj(self.up_proj(x) * self.act((self.gate_proj(x)))))
    
class DecoderLayer(nn.Module):
    def __init__ (self,config:JianMindConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.mlp = FeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)


    def forward(self, x: torch.Tensor,
                pos_embedding: tuple[torch.Tensor, torch.Tensor],
                past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None)->torch.Tensor:
        x1 = self.input_layernorm(x)
        h1 , cur_kv_cache = self.self_attn(x1,pos_embedding,past_kv,use_cache,attention_mask)
        x = x+h1
        x2 = self.post_attention_layernorm(x)
        h2 = self.mlp(x2)
        x = x+h2
        return x , cur_kv_cache

class JianMindModel(nn.Module):
    def __init__ (self,config:JianMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.head_dim = config.hidden_size // config.num_attention_heads
        # 预计算最大长度 RoPE 之后要取用直接按seq切片
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rope_scaling
        )
        # register_buffer 会把张量注册到模型上，自动跟随 model.to(device) 和 model.eval()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.layers = nn.ModuleList([
                DecoderLayer(config) for _ in range(config.num_hidden_layers)
            ])
        self.norm = RMSNorm(config.hidden_size,config.rms_norm_eps)

    def forward(self,input_ids:torch.Tensor,
                past_key_values: Optional[list] = None,
                use_cache: bool = False):
        batch_size , seq_len = input_ids.shape

        # 1. 计算已缓存的序列长度（用于位置编码偏移）
        # 健壮检查：确保 past_key_values 有效
        if (past_key_values
            and isinstance(past_key_values, list)
            and len(past_key_values) > 0
            and past_key_values[0] is not None
            and isinstance(past_key_values[0], tuple)
            and len(past_key_values[0]) > 0
            and past_key_values[0][0] is not None
            and isinstance(past_key_values[0][0], torch.Tensor)):
            past_len = past_key_values[0][0].shape[1]  # 第一层 past_k 的 seq_len
        else:
            past_len = 0

        x = self.embed_tokens(input_ids)
        x = self.dropout(x)

        # 2. 位置编码：从 past_len 处开始切片，保证新 token 拿到正确的位置
        pos_emb = (
            self.freqs_cos[past_len:past_len + seq_len],
            self.freqs_sin[past_len:past_len + seq_len]
        )

        # 3. causal mask：当新输入有多个 token 时需要掩码（防止看到未来信息）
        #    单 token decode（seq_len==1）不需要 mask
        if seq_len > 1:
            total_len = past_len + seq_len
            # 创建 total_len × total_len 的掩码，只掩盖新 token 之间的未来位置
            mask = torch.zeros(seq_len, total_len, device=x.device, dtype=x.dtype)
            mask[:, past_len:] = torch.triu(
                torch.full((seq_len, seq_len), -1e4, device=x.device, dtype=x.dtype),
                diagonal=1
            )
            attention_mask = mask
        else:
            attention_mask = None

        # 4. 逐层传递，每层独立管理自己的 KV cache
        new_past_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, cur_kv = layer(x, pos_emb, past_kv, use_cache, attention_mask)
            if use_cache:
                new_past_key_values.append(cur_kv)

        x = self.norm(x)
        hidden_state = x
        return hidden_state, new_past_key_values

class JianMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    遵循 HuggingFace CausalLM 标准接口的生成模型
    支持 model.generate() 的所有功能：采样、贪心、beam search 等
    """
    config_class = JianMindConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = JianMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 先初始化权重，再绑定 lm_head 与 embed_tokens
        self.post_init()
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module):
        """由 post_init 调用"""
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        temperature: float = None,
        **kwargs
    ):
        # 前向传播
        hidden_states, new_past_key_values = self.model(
            input_ids, past_key_values, use_cache or False
        )

        # 计算 logits
        logits = self.lm_head(hidden_states)

        # 计算 loss（如果有 labels）
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 动态温度缩放：防止 logits 过大导致 NaN
            if temperature is None:
                logits_max = shift_logits.abs().max().item()
                temperature = logits_max / 15.0 if logits_max > 30 else 1.0

            if temperature != 1.0:
                shift_logits = shift_logits / temperature

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=0.1,  # 标签平滑：防止 logits 过大，提高泛化
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past_key_values,
            hidden_states=hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """
        HuggingFace GenerationMixin 每步调用的标准接口
        负责：1) 处理 KV cache 2) 裁剪 input_ids 3) 传递必要参数
        """
        # 有有效的 KV cache 时，只传入最后一个 token
        if self._has_valid_cache(past_key_values):
            input_ids = input_ids[:, -1:]
            past_key_values = past_key_values
        else:
            past_key_values = None

        # 如果提供了 inputs_embeds 且没有 cache，用 embeds 代替 input_ids
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        })
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        """beam search 时重排 KV cache（GenerationMixin 要求的标准方法）"""
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past.append(
                tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                      for past_state in layer_past)
            )
        return reordered_past

    @staticmethod
    def _has_valid_cache(past_key_values):
        """检查 KV cache 是否有效"""
        if past_key_values is None:
            return False
        if isinstance(past_key_values, (list, tuple)):
            if len(past_key_values) == 0:
                return False
            first = past_key_values[0]
            if first is None:
                return False
            if isinstance(first, tuple) and len(first) > 0:
                return first[0] is not None and isinstance(first[0], torch.Tensor)
        return False