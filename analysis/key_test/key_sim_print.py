import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn.functional as F
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
    """keys: [seq_len, head_dim] → returns [seq_len, seq_len] cosine similarity matrix"""
    k1 = keys.unsqueeze(0)
    k2 = keys.unsqueeze(1)
    return F.cosine_similarity(k1, k2, dim=-1)

# ========== Load Sample (OpenBookQA) ==========
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

# ========== Print Top-k Similarities ==========
sim_all = calc_full_key_similarity(keys_all)  # [seq_len, seq_len]
seq_len = sim_all.size(0)
topk = 5

print(f"\nTop-{topk} Cosine similarity values (highest similarities for each token):")
for i in range(seq_len):
    sim_row = sim_all[i]
    # Exclude self-similarity
    sim_row_no_self = sim_row.clone()
    sim_row_no_self[i] = -float("inf")
    top_vals, top_idx = torch.topk(sim_row_no_self, k=topk)
    print(f"Token {i:>2d}: top-{topk} indices = {top_idx.tolist()}, values = {top_vals.tolist()}")
