import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os

# -------------------- 加载模型 --------------------
model_name = "/Applications/All/py_code/pythonProject/LLM-Inference-self/TinyLlama"
if not os.path.exists(model_name):
    raise FileNotFoundError(f"模型路径 {model_name} 不存在")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

# -------------------- 相似度计算函数 --------------------
def calc_full_key_similarity(keys):
    k1 = keys.unsqueeze(0)  # [1, seq_len, head_dim]
    k2 = keys.unsqueeze(1)  # [seq_len, 1, head_dim]
    return F.cosine_similarity(k1, k2, dim=-1)

# -------------------- 从 OpenBookQA 取一个样本 --------------------
dataset = load_dataset("allenai/openbookqa", "main", split="train[:1]")
sample = dataset[0]

question = sample["question_stem"]
choices = sample["choices"]["text"]
labels = sample["choices"]["label"]

text = "<|system|> You are a helpful assistant. <|user|>\n"
text += f"Question: {question}\n"
for lbl, ch in zip(labels, choices):
    text += f"{lbl}: {ch}\n"
text += "<|assistant|>"

print("Sample Question:\n", text)

inputs = tokenizer(text, return_tensors="pt").to(device)

# -------------------- 前向传播获取 past_key_values --------------------
with torch.no_grad():
    outputs = model(**inputs)
    past_key_values = outputs.past_key_values

# -------------------- 取某层某 head --------------------
layer_idx = 17
head_idx = 1
keys_before = past_key_values[layer_idx][0][0, head_idx]  # [seq_len, head_dim]

sim_before = calc_full_key_similarity(keys_before).cpu().numpy()

# -------------------- 可视化热力图 --------------------
mask = np.triu(np.ones_like(sim_before, dtype=bool), k=1)  # 上三角掩码
plt.rcParams.update({'font.size': 16}) 
plt.figure(figsize=(8,6))
ax = sns.heatmap(sim_before, cmap="coolwarm", mask=mask, vmin=-1, vmax=1)
ax.set_facecolor("grey")  # 灰色背景
for spine in ax.spines.values():  # 黑色边框
    spine.set_visible(True)
    spine.set_linewidth(0.5)
    spine.set_color('black')

plt.title(f"Layer {layer_idx}, Head {head_idx}", fontsize=18)
plt.xlabel("")
plt.ylabel("")

plt.tight_layout()
plt.savefig(f"key_similarity_layer{layer_idx}_head{head_idx}.pdf", bbox_inches="tight")
plt.show()
