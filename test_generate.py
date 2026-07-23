#!/usr/bin/env python3
"""
测试预训练模型的文本生成能力
"""

import torch
from transformers import AutoTokenizer
from model.model import JianMindForCausalLM, JianMindConfig

# ========== 配置 ==========
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 测试 prompts
TEST_PROMPTS = [
    "今天天气很好，",
    "The future of artificial intelligence is",
    "def fibonacci(n):\n    \"\"\"计算斐波那契数列\"\"\"",
    "深度学习是一种",
    "Once upon a time,",
]


# ========== 加载模型 ==========
print("🔄 加载模型...")
config = JianMindConfig(hidden_size=768, num_hidden_layers=8)
model = JianMindForCausalLM(config)

checkpoint_path = "out/pretrain_768.pth"
state_dict = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

param_count = sum(p.numel() for p in model.parameters()) / 1e6
print(f"✅ 模型加载完成 ({DEVICE}, {param_count:.2f}M 参数)\n")

# ========== 加载 Tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained("dataset")
print(f"✅ Tokenizer 加载完成 (vocab_size={tokenizer.vocab_size})")
print(f"   bos_token_id: {tokenizer.bos_token_id}")
print(f"   eos_token_id: {tokenizer.eos_token_id}")
print(f"   pad_token_id: {tokenizer.pad_token_id}\n")

# 调试：检查 tokenizer 的特殊 token
print("🔍 检查 tokenizer 配置:")
print(f"   bos_token: {repr(tokenizer.bos_token)}")
print(f"   eos_token: {repr(tokenizer.eos_token)}")
print(f"   pad_token: {repr(tokenizer.pad_token)}")
print()


# ========== 测试生成 ==========
print("=" * 80)
print("🎯 开始测试生成能力")
print("=" * 80)

for i, prompt in enumerate(TEST_PROMPTS, 1):
    print(f"\n{'='*80}")
    print(f"【测试 {i}/{len(TEST_PROMPTS)}】")
    print(f"Prompt: {prompt}")
    print("-" * 80)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    print(f"输入 tokens: {input_ids[0].tolist()}")
    print(f"输入长度: {input_ids.shape[1]}")

    # 方法1: 贪心解码（确定性输出）
    print("\n[方法1] 贪心解码 (greedy):")
    with torch.no_grad():
        output_ids_greedy = model.generate(
            input_ids,
            max_new_tokens=100,
            min_new_tokens=20,  # 强制生成至少20个token
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    result_greedy = tokenizer.decode(output_ids_greedy[0], skip_special_tokens=True)
    new_tokens_greedy = output_ids_greedy.shape[1] - input_ids.shape[1]
    print(f"生成: {result_greedy}")
    print(f"新生成 token 数: {new_tokens_greedy}")
    print(f"生成的 token IDs: {output_ids_greedy[0, input_ids.shape[1]:].tolist()}")

    # 方法2: 采样（随机性输出）
    print("\n[方法2] 采样 (sampling):")
    with torch.no_grad():
        output_ids_sample = model.generate(
            input_ids,
            max_new_tokens=100,
            min_new_tokens=20,  # 强制生成至少20个token
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    result_sample = tokenizer.decode(output_ids_sample[0], skip_special_tokens=True)
    new_tokens_sample = output_ids_sample.shape[1] - input_ids.shape[1]
    print(f"生成: {result_sample}")
    print(f"新生成 token 数: {new_tokens_sample}")
    print(f"生成的 token IDs: {output_ids_sample[0, input_ids.shape[1]:].tolist()}")

print("\n" + "=" * 80)
print("✅ 测试完成")
print("=" * 80)
print("\n📊 总结:")
print("- 贪心解码：输出确定，适合评估模型基础能力")
print("- 采样解码：输出多样，适合评估生成质量")
print("- 如果两种方法都生成很少的内容，可能是模型训练不足或数据问题")
print("- 如果贪心解码效果好但采样差，可能是采样参数需要调整")
