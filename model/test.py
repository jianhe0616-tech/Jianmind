import torch
from config import JianMindConfig
from model import JianMind

config = JianMindConfig()
model = JianMind(config)

total = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total:,}")

input_ids = torch.randint(0, config.vocab_size, (2, 16))
labels = torch.randint(0, config.vocab_size, (2, 16))

model.eval()
with torch.no_grad():
    logits, loss = model(input_ids, labels)

print(f"logits: {logits.shape}")
print(f"loss:   {loss.item():.4f}")
