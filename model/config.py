##这里放各参数设置
from transformers import PretrainedConfig
class JianMindConfig(PretrainedConfig):
    model_type = "jianmind"

    def __init__(
        self,
        # ==================== 基础架构参数 ====================
        dropout: float = 0.0,                     # Dropout 比率
        bos_token_id: int = 1,                    # 句子开始 token ID
        eos_token_id: int = 2,                    # 句子结束 token ID
        hidden_act: str = "silu",                 # 激活函数 (silu/swish)
        hidden_size: int = 512,                   # 隐藏层维度（模型宽度）
        intermediate_size: int = None,            # FFN 中间维度，None 则自动计算
        max_position_embeddings: int = 32768,     # 最大上下文长度 (32k)
        num_attention_heads: int = 8,             # Query 头数
        num_hidden_layers: int = 8,               # Transformer 层数
        num_key_value_heads: int = 2,             # KV 头数（GQA 分组）
        vocab_size: int = 6400,                   # 词表大小
        rms_norm_eps: float = 1e-05,              # RMSNorm 防除零系数
        rope_base: int = 1000000,                # RoPE 基数（用于位置编码）
        inference_rope_scaling: bool = False,     # 是否启用 YaRN 上下文扩展
        flash_attention: bool = True,             # 是否使用 Flash Attention
        # ==================== MoE 混合专家参数 ====================
        use_moe: bool = False,                    # 是否启用 MoE
        num_experts_per_tok: int = 2,             # 每个 token 激活的专家数
        n_routed_experts: int = 4,                # 参与路由的专家总数
        n_shared_experts: int = 1,                # 共享专家数
        scoring_func: str = "softmax",            # 路由打分函数
        aux_loss_alpha: float = 0.01,             # 辅助损失权重（负载均衡）
        seq_aux: bool = True,                     # 是否在序列级别计算辅助损失
        norm_topk_prob: bool = True,              # 是否对 top-k 概率归一化
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
        self.rope_base = rope_base
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

        # YaRN 扩展配置（当 inference_rope_scaling 为 True 时启用）
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