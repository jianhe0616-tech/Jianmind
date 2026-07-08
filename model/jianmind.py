from transformers import PretrainedConfig


class JianMindConfig(PretrainedConfig):
    model_type = "jianmind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import math
from typing import Optional

###RMSNorm 模块实现
class RMSNorm(nn.Module):
## 首先是init初始化
    def __init__(self,dim:int,eps:float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
## 然后是norm的计算公式
## rsqrt:开方之后取倒数
## mean(纬度：-1,keepdim=True):对最后一个纬度求均值，keepdim=True表示保持维度不变
    def norm(self,x:torch.tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
    
## 然后前向传播
    def forward(self,x:torch.tensor):
        output = self.norm(x.float()) * self.weight
        return output
###

### YaRN参数计算方法  dim:纬度   end：训练最大序列长度   base：频率计算基底  rope_scaling：缩放公式各参数
def precompute_freqs_cis(dim:int,end:int,rope_base:int,rope_scaling:Optional[dict]=None):
    #初始化频率，和softmax缩放系数  语法：python允许在任何对象后面直接加.或者[]来调用方法
    #如下直接调用了[]截断（防止奇数）和.float方法
    i=torch.arange(0, dim, 2)[:dim//2].float()  
    fre = 1/(rope_base**(2*i/dim))
    attn_factor = 1.0
    if rope_scaling is not None:
        max_pre_context = rope_scaling["original_max_position_embeddings"]
        beta_fast = rope_scaling["beta_fast"]
        beta_slow = rope_scaling["beta_slow"]
        factor_s = rope_scaling["factor"]

        #判断推理长度和训练文本长度
        #当b=1和32时，复用lambda函数可以求出对应的维度边界,即波长b到i的映射
        if end>max_pre_context:
            inv_dim = lambda b: (dim * math.log(max_pre_context/(b*2*math.pi)))/(2*math.log(rope_base))
        
        #划分高低维度
        dim_low = max(0, int(math.floor(inv_dim(beta_slow))))
        dim_high = min(dim // 2, int(math.ceil(inv_dim(beta_fast))))

        

        


