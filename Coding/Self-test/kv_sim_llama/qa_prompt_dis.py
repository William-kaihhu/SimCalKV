"""
Compares the key and value similarity before and after merge, 
using a QA question defined by myself
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- 加载模型 --------------------
model_name = "/Applications/All/py_code/pythonProject/LLM-Inference-self/TinyLLama"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
device = torch.device("mps")
model.to(device)

# -------------------- Prompt --------------------
prompt = "<|system|> You are a helpful assistant. <|user|> The Apollo program was the third United States human spaceflight program carried out by NASA, which accomplished landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as a three-person spacecraft to follow the one-person Project Mercury, which put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's national goal of \"landing a man on the Moon and returning him safely to the Earth\" by the end of the 1960s, which he proposed in an address to Congress in May 1961. **Question:** Which U.S. president is credited with the national goal of landing a man on the Moon? <|assistant|>"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# -------------------- 合并 KV 函数 --------------------
def merge_kv_pairs(past_key_values, start_idx: int, end_idx: int):
    merged_past = []
    for k, v in past_key_values:
        k_before = k[:, :, :start_idx, :]
        v_before = v[:, :, :start_idx, :]
        k_range = k[:, :, start_idx:end_idx, :]
        v_range = v[:, :, start_idx:end_idx, :]

        even_len = k_range.size(2) // 2 * 2
        k_pairs = (k_range[:, :, :even_len:2, :] + k_range[:, :, 1:even_len:2, :]) / 2
        v_pairs = (v_range[:, :, :even_len:2, :] + v_range[:, :, 1:even_len:2, :]) / 2

        if k_range.size(2) % 2 == 1:
            k_pairs = torch.cat([k_pairs, k_range[:, :, -1:, :]], dim=2)
            v_pairs = torch.cat([v_pairs, v_range[:, :, -1:, :]], dim=2)

        k_after = k[:, :, end_idx:, :]
        v_after = v[:, :, end_idx:, :]
        new_k = torch.cat([k_before, k_pairs, k_after], dim=2)
        new_v = torch.cat([v_before, v_pairs, v_after], dim=2)
        merged_past.append((new_k, new_v))
    return merged_past

# -------------------- 生成 token 函数 --------------------
def generate_token_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        next_token_id = logits[0, -1, :].argmax()
        return next_token_id, outputs.past_key_values

def generate_token_with_past_merged(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        next_token_id = logits[0, -1, :].argmax()
        merged_kv = merge_kv_pairs(outputs.past_key_values, start_idx=0, end_idx=outputs.past_key_values[0][0].size(2))
        return next_token_id, merged_kv

# -------------------- 生成一些 token --------------------
generated_tokens = []
generated_tokens_merge = []
next_inputs = inputs

next_token_id, past_key_values = generate_token_with_past(next_inputs)
next_token = tokenizer.decode(next_token_id)
generated_tokens.append(next_token)

next_inputs_merge = next_inputs.copy()
next_token_id_merge, past_key_values_merge = generate_token_with_past_merged(next_inputs_merge)
generated_tokens_merge.append(tokenizer.decode(next_token_id_merge))
next_inputs_merge = {
    "input_ids": next_token_id_merge.reshape((1,1)),
    "attention_mask": torch.cat([next_inputs_merge["attention_mask"], torch.tensor([[1]], device=device)], dim=1),
    "past_key_values": past_key_values_merge,
}


# -------------------- 可视化热力图(L2距离) --------------------
import seaborn as sns

# 计算 L2 距离矩阵
def calc_full_key_distance(keys):
    """
    keys: [seq_len, head_dim]
    返回每对 token 之间的 L2 距离矩阵 [seq_len, seq_len]
    """
    k1 = keys.unsqueeze(0)  # [1, seq_len, head_dim]
    k2 = keys.unsqueeze(1)  # [seq_len, 1, head_dim]
    diff = k1 - k2          # [seq_len, seq_len, head_dim]
    dist_matrix = torch.norm(diff, dim=-1)  # [seq_len, seq_len]
    return dist_matrix
# -------------------- 取某层某 head --------------------
layer_idx = 3
head_idx = 2
keys_before = past_key_values[layer_idx][0][0, head_idx]  # [seq_len, head_dim]
keys_after  = past_key_values_merge[layer_idx][0][0, head_idx]
dist_before = calc_full_key_distance(keys_before).cpu().numpy()
dist_after  = calc_full_key_distance(keys_after).cpu().numpy()

# 上三角 mask
mask1 = np.triu(np.ones_like(dist_before, dtype=bool), k=1)
mask2 = np.triu(np.ones_like(dist_after, dtype=bool), k=1)

plt.figure(figsize=(12,5))

ax1 = plt.subplot(1, 2, 1)
sns.heatmap(dist_before, cmap="coolwarm", mask=mask1)
ax1.set_facecolor("grey")
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)
    spine.set_color('black')
plt.title(f"Key L2 Distance Before Merge (Layer {layer_idx}, Head {head_idx})")
plt.xlabel("Token Index")
plt.ylabel("Token Index")

ax2 = plt.subplot(1, 2, 2)
sns.heatmap(dist_after, cmap="coolwarm", mask=mask2)
ax2.set_facecolor("grey")
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)
    spine.set_color('black')
plt.title(f"Key L2 Distance After Merge (Layer {layer_idx}, Head {head_idx})")
plt.xlabel("Token Index")
plt.ylabel("Token Index")

plt.tight_layout()
plt.show()
