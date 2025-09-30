import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# -------------------- 加载模型 --------------------
model_name = "/Applications/All/py_code/pythonProject/LLM-Inference-self/TinyLLama"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
model.eval()
device = torch.device("mps")
model.to(device)

# -------------------- Prompt --------------------
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

inputs = tokenizer(text, return_tensors="pt").to(device)

# -------------------- 前向传播拿注意力 --------------------
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions

# -------------------- 选择层和头 --------------------
layer = 18   # 自己选择
head = 3    # 自己选择

attention_map = attentions[layer][0, head].cpu().numpy()  # shape (seq_len, seq_len)

# -------------------- 统计注意力分布 --------------------
all_attention = attention_map.flatten()

num_bins = 100
bin_edges = np.linspace(0, 1, num_bins + 1)

# 用比例（方便比较不同层/头）
hist, _ = np.histogram(all_attention, bins=bin_edges)
hist = hist / hist.sum()

# -------------------- 绘图 --------------------
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(10, 5))

plt.bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]),
        color="#69b3a2", edgecolor='black', alpha=0.7, align='edge')

plt.xlabel("Attention score")
plt.ylabel("Proportion")
plt.title(f"Attention Distribution - Layer {layer}, Head {head}")
plt.xlim(0, 1)

# ✅ 方法二：y 轴对数坐标
plt.yscale("log")

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"attention_distribution_layer{layer}_head{head}_log.pdf", bbox_inches="tight")
plt.show()
