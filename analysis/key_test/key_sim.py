import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ========== Model Loading ==========
model_name = "meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        # Automatically allocate to available devices
    torch_dtype=torch.float16 # Use half precision to save memory
)
model.eval()

# ========== Cosine Similarity & L2 Distance Computation ==========
def calc_full_key_similarity(keys):
    """keys: [seq_len, head_dim] 
       Returns: [seq_len, seq_len] cosine similarity matrix"""
    k1 = keys.unsqueeze(0)
    k2 = keys.unsqueeze(1)
    return F.cosine_similarity(k1, k2, dim=-1)

def calc_full_key_l2(keys):
    """keys: [seq_len, head_dim] 
       Returns: [seq_len, seq_len] L2 distance matrix"""
    diff = keys.unsqueeze(0) - keys.unsqueeze(1)
    return torch.norm(diff, dim=-1)

# ========== Load an OpenBookQA Sample ==========
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
print("Sample Question:\n", text)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

# ========== Keys in the Prefill Stage ==========
with torch.no_grad():
    outputs = model(**inputs)
    past_key_values = outputs.past_key_values

layer_idx = 3
head_idx = 2

prefill_len = inputs['input_ids'].shape[1]
keys_prefill = past_key_values[layer_idx][0][0, head_idx][:prefill_len]  # [actual_tokens, head_dim]

# ========== Keys in the Decode Stage ==========
num_decode_steps = 20
next_inputs = inputs
decode_key_list = []

with torch.no_grad():
    past_key_values = None
    for step in range(num_decode_steps):
        outputs = model(**next_inputs)
        past_key_values = outputs.past_key_values
        key_step = past_key_values[layer_idx][0][0, head_idx][-1]  # Take the last token
        decode_key_list.append(key_step.unsqueeze(0))  # [1, head_dim]

        # Generate the next token
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        next_inputs = {"input_ids": next_token, "past_key_values": past_key_values}

keys_decode = torch.cat(decode_key_list, dim=0)  # [decode_len, head_dim]

# ========== Concatenate Prefill + Decode Keys ==========
keys_all = torch.cat([keys_prefill, keys_decode], dim=0)
split_idx = keys_prefill.shape[0]

# ========== Plot Cosine Similarity ==========
sim_all = calc_full_key_similarity(keys_all).cpu().numpy()
mask = np.triu(np.ones_like(sim_all, dtype=bool), k=1)

plt.figure(figsize=(8, 6))
ax = sns.heatmap(sim_all, cmap="coolwarm", mask=mask, vmin=-1, vmax=1)
ax.set_facecolor("grey")
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)
    spine.set_color('black')

plt.title(f"Key Cosine Similarity - Layer {layer_idx}, Head {head_idx}", fontsize=16)
plt.axhline(split_idx, color='black', linestyle='--', linewidth=1)
plt.axvline(split_idx, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig(f"openbookqa_key_cosine_layer{layer_idx}_head{head_idx}_prefill_decode.pdf", bbox_inches="tight")
plt.show()

# ========== Plot L2 Distance ==========
l2_all = calc_full_key_l2(keys_all).cpu().numpy()
mask = np.triu(np.ones_like(l2_all, dtype=bool), k=1)

plt.figure(figsize=(8, 6))
ax = sns.heatmap(l2_all, cmap="viridis", mask=mask, vmin=0, vmax=np.max(l2_all))
ax.set_facecolor("grey")
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)
    spine.set_color('black')

plt.title(f"Key L2 Distance - Layer {layer_idx}, Head {head_idx}", fontsize=16)
plt.axhline(split_idx, color='black', linestyle='--', linewidth=1)
plt.axvline(split_idx, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig(f"openbookqa_key_l2_layer{layer_idx}_head{head_idx}_prefill_decode.pdf", bbox_inches="tight")
plt.show()
