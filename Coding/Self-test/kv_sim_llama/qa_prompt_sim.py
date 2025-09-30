import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

# -------------------- 前向传播拿 past_key_values --------------------
with torch.no_grad():
    outputs = model(**inputs)
    past_key_values = outputs.past_key_values

# -------------------- 相邻 key 相似度函数 --------------------
def calc_full_key_similarity(keys):
    k1 = keys.unsqueeze(0)  # [1, seq_len, head_dim]
    k2 = keys.unsqueeze(1)  # [seq_len, 1, head_dim]
    sim_matrix = F.cosine_similarity(k1, k2, dim=-1)  # [seq_len, seq_len]
    return sim_matrix

# -------------------- 取某层某 head --------------------
layer_idx = 3
head_idx = 2
keys_before = past_key_values[layer_idx][0][0, head_idx]  # [seq_len, head_dim]

sim_before = calc_full_key_similarity(keys_before)

# -------------------- 可视化热力图(Key Similarity before merge) --------------------
mask = np.triu(np.ones_like(sim_before.cpu().numpy(), dtype=bool), k=1)

plt.figure(figsize=(8,6))
sns.heatmap(sim_before.cpu().numpy(), cmap="coolwarm", mask=mask, vmin=0, vmax=1)

plt.title(f"Adjacent Key Similarity Before Merge (Layer {layer_idx}, Head {head_idx})", fontsize=16)
plt.xlabel("Token Index", fontsize=14)
plt.ylabel("Token Index", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig(f"key_similarity_before_merge_layer{layer_idx}_head{head_idx}.png", dpi=600)  # 高分辨率保存
plt.show()
