import torch
import json
from torch.utils.data import Dataset
import os
import random
from datasets import load_dataset, Features, Sequence, Value

# 防止 tokenizer 的多进程与 DataLoader 的 num_workers 冲突导致死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    """
    预训练数据集类
    将原始文本 tokenize 并填充到固定长度 max_length
    返回 input_ids 和 labels（labels 中 pad 位置设为 -100 忽略）
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        data_path:  jsonl 文件路径，每行 {"text": "..."}
        tokenizer:  HuggingFace tokenizer 实例
        max_length: 序列最大长度（包含 bos 和 eos）
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 惰性加载 JSON 文件，不一次性读入内存
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 1. 取出一条原始文本
        sample = self.samples[index]

        # 2. tokenize：预留 2 个位置给 bos/eos，超长截断
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        ).input_ids

        # 3. 拼接：[bos] + tokens + [eos]
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # 4. 右侧填充到 max_length，不足部分用 pad_token_id 补齐
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 5. labels = input_ids 的复制，但 pad 位置设为 -100
        #    CrossEntropyLoss(ignore_index=-100) 会自动跳过这些位置
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels
    

     
