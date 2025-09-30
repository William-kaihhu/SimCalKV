import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- 加载模型 --------------------
model_name = "/Applications/All/py_code/pythonProject/LLM-Inference-self/TinyLLama"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
model.eval()
device = torch.device("mps")
model.to(device)

# -------------------- Prompt --------------------
prompt = "<|system|> You are a helpful assistant. <|user|> \
The Apollo program was the third United States human spaceflight program carried out by NASA, \
which accomplished landing the first humans on the Moon from 1969 to 1972. \
First conceived during Dwight D. Eisenhower's administration as a three-person spacecraft \
to follow the one-person Project Mercury, which put the first Americans in space, \
Apollo was later dedicated to President John F. Kennedy's national goal of \
\"landing a man on the Moon and returning him safely to the Earth\" by the end of the 1960s, \
which he proposed in an address to Congress in May 1961. \
**Question:** Which U.S. president is credited with the national goal of landing a man on the Moon? \
<|assistant|>"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# -------------------- 前向传播 --------------------
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    attentions = outputs.attentions          # list of [batch, heads, seq_len, seq_len]
    past_key_values = outputs.past_key_values

# -------------------- 可视化函数 --------------------
def plot_l2_vs_attention(attentions, past_key_values, layer_idx, head_idx):
    """
    对比 Key L2 Norm 和 模型真实 Attention (total attention)
    """
    # 取真实 attention
    attn_map = attentions[layer_idx][0, head_idx]      # [seq_len, seq_len]
    total_att = attn_map.sum(dim=0).cpu().numpy()      # 每个 key 被所有 query 关注的总和

    # 取 keys 并计算 L2 Norm
    keys = past_key_values[layer_idx][0][0, head_idx]  # [seq_len, head_dim]
    l2_norms = torch.norm(keys, dim=-1).cpu().numpy()

    # 绘图
    x = np.arange(len(l2_norms))
    fig, ax1 = plt.subplots(figsize=(14,5))

    color1 = "purple"
    ax1.set_xlabel("Token Index")
    ax1.set_ylabel("Key L2 Norm", color=color1)
    ax1.bar(x, l2_norms, color=color1, alpha=0.6, label="Key L2 Norm")
    ax1.tick_params(axis='y', labelcolor=color1)

    color2 = "green"
    ax2 = ax1.twinx()
    ax2.set_ylabel("Total Attention (from model)", color=color2)
    ax2.plot(x, total_att, color=color2, marker="o", label="Total Attention")
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f"L2 Norm vs Total Attention (Layer {layer_idx}, Head {head_idx})")
    fig.tight_layout()
    plt.show()

# -------------------- 使用函数 --------------------
layer = 12
head = 2
plot_l2_vs_attention(attentions, past_key_values, layer, head)
