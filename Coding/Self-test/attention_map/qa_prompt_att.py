import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
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

# -------------------- 前向传播并拿到注意力 --------------------
with torch.no_grad():
    outputs = model(**inputs, outputs_attentions=True)
    logits = outputs.logits
    attentions = outputs.attentions  # list, 每层的注意力

print(f"Number of layers: {len(attentions)}")
print(f"Shape of one attention map: {attentions[0].shape}")
# (batch, num_heads, seq_len, seq_len)

# -------------------- 可视化某一层某一头 --------------------
layer = 10   # 第一层
head = 2    # 第一个头
attention_map = attentions[layer][0, head].cpu().numpy()  # shape (seq_len, seq_len)

plt.figure(figsize=(8, 6))
sns.heatmap(attention_map, cmap="coolwarm")
plt.title(f"Attention Map - Layer {layer}, Head {head}")
plt.xlabel("Key Positions")
plt.ylabel("Query Positions")
plt.show()
