"""
Visualize Key L2 Norms and Attention Scores for comparison
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- 加载模型 --------------------
model_name = "/Applications/All/py_code/pythonProject/LLM-Inference-self/TinyLLama"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
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

# -------------------- 前向传播 --------------------
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    attentions = outputs.attentions      # 注意力 (list of [batch, heads, seq, seq])
    past_key_values = outputs.past_key_values  # (key, value) per layer

print(f"Number of layers: {len(attentions)}")
print(f"Shape of one attention map: {attentions[0].shape}")  # (batch, num_heads, seq_len, seq_len)

# -------------------- 选择层和头 --------------------
layer = 7
head = 0

attn_map = attentions[layer][0, head].cpu().numpy()   # [seq, seq]
keys = past_key_values[layer][0][0, head]             # [seq, head_dim]
l2_norms = torch.norm(keys, dim=-1).cpu().numpy()     # 每个 token 的 L2 范数

# -------------------- 可视化对比 --------------------
seq_len = keys.shape[0]
x = np.arange(seq_len)

fig, ax1 = plt.subplots(figsize=(14,5))

# Key L2 norm 柱状图
color1 = 'purple'
ax1.set_xlabel('Token Index')
ax1.set_ylabel('Key L2 Norm', color=color1)
ax1.bar(x, l2_norms, color=color1, alpha=0.6, label='Key L2 Norm')
ax1.tick_params(axis='y', labelcolor=color1)

# Attention score = 最后一个 query 对所有 key 的分布
attn_scores = attn_map[-1]   # shape = (seq_len, )
ax2 = ax1.twinx()
color2 = 'green'
ax2.set_ylabel('Attention Score (last query)', color=color2)
ax2.plot(x, attn_scores, color=color2, marker='o', label='Attention Score')
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.title(f"L2 Norm vs Attention Score (Layer {layer}, Head {head})")
plt.show()
