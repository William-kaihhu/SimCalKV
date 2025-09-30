import torch
import math
import torch.nn as nn
# batch=1, head=2, seq_len=4, head_dim=3
k1 = torch.randn(1, 2, 4, 3)
v1 = torch.randn(1, 2, 4, 3)
k2 = torch.randn(1, 2, 4, 3)
v2 = torch.randn(1, 2, 4, 3)

q = torch.randn(1, 2, 4, 3)
attention_score = q @ k1.transpose(2, 3) / math.sqrt(3)
print(attention_score)
attention_weight = torch.softmax(attention_score, dim=-1)
print(attention_weight)
output = attention_weight @ v1
print(output)

# 假设 output 形状为 (batch, head, seq_len, head_dim)
output = output.permute(0, 2, 1, 3).reshape(1, 4, -1)  # (batch, seq_len, head*head_dim)
lm_head = nn.Linear(2 * 3, 5)  # head*head_dim=6, vocab_size=10
print(lm_head.weight)
print(lm_head.bias)
logits = lm_head(output)  # (batch, seq_len, vocab_size)
probs = torch.softmax(logits, dim=-1)  # 概率分布

# 取最后一个 token 的概率分布
next_token_probs = probs[:, -1, :]  # (batch, vocab_size)
print(next_token_probs)