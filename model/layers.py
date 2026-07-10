##这里放无需要更新权重的函数
import torch
import torch.nn as nn
import math
from typing import Optional
from config import JianMindConfig

###RMSNorm 模块实现
class RMSNorm(nn.Module):
## 首先是init初始化
    def __init__(self,dim:int,rms_norm_eps:float = 1e-5):
        super().__init__()
        self.dim = dim
        self.rms_norm_eps = rms_norm_eps
        self.weight = nn.Parameter(torch.ones(dim))
## 然后是norm的计算公式
## rsqrt:开方之后取倒数
## mean(纬度：-1,keepdim=True):对最后一个纬度求均值，keepdim=True表示保持维度不变,后面加eps防止除零
    def norm(self,x:torch.tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.rms_norm_eps)
    
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
    attn_factor = 1.0  #随着缩放因子S（扩展倍数）动态变化,这里取1.0，后续可以根据需要调整
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
            dim_low = max(0, int(math.floor(inv_dim(beta_slow))))  #不需要缩放的低维度
            dim_high = min(dim // 2, int(math.ceil(inv_dim(beta_fast))))   #需要缩放的高维度
            
            #计算缩放因子
            indices = torch.arange(0, dim//2,device=fre.device).float() #计算索引
            ramp= torch.clamp((indices-dim_low)/(dim_high-dim_low),0,1)  #计算缩放系数,clamp函数用于限制范围在0到1之间
            #计算缩放后的频率
            fre = fre * (1.0 - ramp + ramp / factor_s)
    #计算旋转角度   torch.cat用于拼接张量，dim=0表示按行拼接，dim=1表示按列拼接
    #首先生成位置索引
    m = torch.arange(end, device=fre.device).float()
    #计算旋转角度
    freqs = torch.outer(m, fre).float() #先外积，得到每个位置旋转角度m theta
    #计算正弦余弦值
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    #同时注入softmax缩放系数atte_factor，防止后面遗漏
    # ✅ 修复：改成单行，确保返回的是张量而不是元组
    freqs_cos = freqs_cos * attn_factor
    freqs_sin = freqs_sin * attn_factor
    return freqs_cos, freqs_sin
##
##RoPE旋转编码函数
def apply_rotary_pos_emb(q,k,freqs_cos, freqs_sin):

    #纬度修正：
    #确保freqs_cos和freqs_sin的形状与q和k的最后一个维度匹配
    freqs_cos = freqs_cos.unsqueeze(1)
    freqs_sin = freqs_sin.unsqueeze(1)

    #将q和k的最后一个维度拆分为两部分，分别表示偶数和奇数位置的向量
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    #计算旋转后的q和k
    q_rotated = torch.cat([q1 * freqs_cos - q2 * freqs_sin, q1 * freqs_sin + q2 * freqs_cos], dim=-1)
    k_rotated = torch.cat([k1 * freqs_cos - k2 * freqs_sin, k1 * freqs_sin + k2 * freqs_cos], dim=-1)
    return q_rotated, k_rotated
