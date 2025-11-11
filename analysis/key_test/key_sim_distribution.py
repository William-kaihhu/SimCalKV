import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ========== Model Loading ==========
model_name = "meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# ========== Similarity Function ==========
def calc_full_key_similarity(keys):
    """keys: [seq_len, head_dim] → [seq_len, seq_len] Cosine similarity matrix"""
    k1 = keys.unsqueeze(0).float()
    k2 = keys.unsqueeze(1).float()
    sim = F.cosine_similarity(k1, k2, dim=-1)
    return torch.clamp(sim, -1.0, 1.0)

# ========== Load Example (OpenBookQA) ==========
dataset = load_dataset("openbookqa", "main", split="train[:1]")
sample = dataset[0]
question = sample["question_stem"]
choices = sample["choices"]["text"]
labels = sample["choices"]["label"]

text = "<|system|> You are a helpful assistant. <|user|>\n"
text += f"Question: {question}\n"
for lbl, ch in zip(labels, choices):
    text += f"{lbl}: {ch}\n"
text += "<|assistant|>"

inputs = tokenizer(text, return_tensors="pt").to(model.device)

# ========== Prefill Stage ==========
with torch.no_grad():
    outputs = model(**inputs)
    past_key_values = outputs.past_key_values

layer_idx = 3
head_idx = 2

prefill_len = inputs['input_ids'].shape[1]
keys_prefill = past_key_values[layer_idx][0][0, head_idx][:prefill_len]

# ========== Decode Stage ==========
num_decode_steps = 20
next_inputs = inputs
decode_key_list = []

with torch.no_grad():
    past_key_values = None
    for step in range(num_decode_steps):
        outputs = model(**next_inputs)
        past_key_values = outputs.past_key_values
        key_step = past_key_values[layer_idx][0][0, head_idx][-1]
        decode_key_list.append(key_step.unsqueeze(0))

        # Generate the next token
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        next_inputs = {"input_ids": next_token, "past_key_values": past_key_values}

keys_decode = torch.cat(decode_key_list, dim=0)
keys_all = torch.cat([keys_prefill, keys_decode], dim=0)

# ========== Compute Similarity ==========
sim_all = calc_full_key_similarity(keys_all)
sim_vals = sim_all.flatten().cpu().numpy()

# Count similarities greater than 0.9
count_high = (sim_vals > 0.9).sum()
ratio_high = count_high / len(sim_vals)
print(f"Number of similarities > 0.9: {count_high}")
print(f"Ratio: {ratio_high * 100:.2f}%")

# ========== Plot Histogram ==========
plt.figure(figsize=(7, 5))
plt.hist(sim_vals, bins=40, range=(-1, 1), color='steelblue', edgecolor='black', alpha=0.8)
plt.title(f"Cosine Similarity Distribution (Layer {layer_idx}, Head {head_idx})", fontsize=14)
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.grid(alpha=0.3)
plt.tight_layout()

# Save as vector PDF
plt.savefig("./similarity_hist.pdf", format="pdf")
plt.show()

print("Vector plot saved as: similarity_hist.pdf")
