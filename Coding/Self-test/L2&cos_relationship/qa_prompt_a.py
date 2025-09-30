"""
Compares key similarity and pairwise L2 differences (as heatmap) before merge
using a QA question
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# -------------------- 加载模型 --------------------
model_name = "/Applications/All/py_code/pythonProject/LLM-Inference-self/TinyLLama"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
device = torch.device("mps")
model.to(device)

# -------------------- Prompt --------------------
prompt = "<|system|> \
You are a helpful assistant.\
<|user|>\
The Apollo program was the third United States human spaceflight program carried out by NASA, \
which accomplished landing the first humans on the Moon from 1969 to 1972. \
First conceived during Dwight D. Eisenhower's administration as a three-person spacecraft \
to follow the one-person Project Mercury, which put the first Americans in space, \
Apollo was later dedicated to President John F. Kennedy's national goal of \
""landing a man on the Moon and returning him safely to the Earth"" by the end of the 1960s, \
which he proposed in an address to Congress in May 1961.\
**Question:** Which U.S. president is credited with the national goal of landing a man on the Moon?\
<|assistant|>"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# -------------------- 生成 token --------------------
def generate_token_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        next_token_id = logits[0, -1, :].argmax()
        return next_token_id, outputs.past_key_values

next_token_id, past_key_values = generate_token_with_past(inputs)
# 生成 15 个 token
generated_tokens = []
next_inputs = inputs
for _ in range(15):
    next_token_id, past_key_values = generate_token_with_past(next_inputs)
    next_inputs = {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat([next_inputs["attention_mask"], torch.tensor([[1]], device=device)], dim=1),
        "past_key_values": past_key_values,
    }
    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)

print("Generated tokens(before merge):", generated_tokens)

# -------------------- 相似度和 L2 差 --------------------
def calc_full_key_similarity(keys):
    k1 = keys.unsqueeze(0)
    k2 = keys.unsqueeze(1)
    return F.cosine_similarity(k1, k2, dim=-1)

def calc_pairwise_key_l2_diff(keys):
    """
    keys: [seq_len, head_dim]
    返回矩阵 [seq_len, seq_len]，第 (i,j) 个元素为 ||key_i - key_j||_2
    """
    k1 = keys.unsqueeze(0)  # [1, seq_len, head_dim]
    k2 = keys.unsqueeze(1)  # [seq_len, 1, head_dim]
    diff = k1 - k2
    l2_matrix = torch.norm(diff, dim=-1)
    return l2_matrix

# -------------------- 取某层某 head --------------------
layer_idx = 4
head_idx = 2
keys = past_key_values[layer_idx][0][0, head_idx]  # [seq_len, head_dim]

sim_matrix = calc_full_key_similarity(keys).cpu().numpy()
l2_diff_matrix = calc_pairwise_key_l2_diff(keys).cpu().numpy()

# -------------------- 可视化 --------------------
fig, axes = plt.subplots(1, 2, figsize=(14,5))
mask1 = np.triu(np.ones_like(l2_diff_matrix, dtype=bool), k=1)

# Heatmap: Key Similarity

sns.heatmap(sim_matrix, cmap="coolwarm", mask=mask1, vmin=0, vmax=1, ax=axes[0])
axes[0].set_facecolor("grey")
for spine in axes[0].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)
    spine.set_color('black')
axes[0].set_title(f"Key Similarity (Layer {layer_idx}, Head {head_idx})")
axes[0].set_xlabel("Token Index")
axes[0].set_ylabel("Token Index")

# Heatmap: Pairwise L2 Differences
sns.heatmap(l2_diff_matrix, cmap="viridis", mask=mask1, ax=axes[1])
axes[1].set_facecolor("grey")
for spine in axes[1].spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)
    spine.set_color('black')
axes[1].set_title(f"Pairwise Key L2 Differences (Layer {layer_idx}, Head {head_idx})")
axes[1].set_xlabel("Token Index")
axes[1].set_ylabel("Token Index")

plt.tight_layout()
plt.show()
