import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


model_name = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
)
model.eval()


local_file = "Longbench/data/gov-report.jsonl"
if not os.path.exists(local_file):
    raise ValueError(f"Local file {local_file} not found")

ds = load_dataset("json", data_files=local_file, split="test[:1]") 
samples = []

for ex in ds:
    text = ex.get("input", "") or ex.get("context", "")
    ref = ex["answers"][0] if isinstance(ex["answers"], list) else ex["answers"]
    if not text.strip():
        continue  

    prompt = (
        "Summarize the following government report into a concise abstract.\n\n"
        f"Report:\n{text}\n\nSummary:"
    )
    samples.append({
        "dataset": "govreport",
        "prompt": prompt,
        "reference": ref
    })


text = "<|system|> You are a helpful assistant. <|user|>\n"
text += samples[0]["prompt"] + "\n<|assistant|>"

inputs = tokenizer(text, return_tensors="pt").to("cuda:0")


layer_idx = 14
head_idx = 1
num_decode_steps = 60
past_key_values = None
next_inputs = inputs
decode_q_list = []

with torch.no_grad():
    for step in range(num_decode_steps):
        outputs = model(**next_inputs, use_cache=True, output_hidden_states=True)
        past_key_values = outputs.past_key_values
        hidden_states = outputs.hidden_states

        h = hidden_states[layer_idx][:, -1:, :]
        attn = model.model.layers[layer_idx].self_attn
        q = attn.q_proj(h)
        num_heads = model.config.num_attention_heads
        head_dim = q.shape[-1] // num_heads
        q = q.view(1, 1, num_heads, head_dim)

        decode_q_list.append(q[0, 0, head_idx].detach().cpu())

        
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        next_inputs = {"input_ids": next_token, "past_key_values": past_key_values}


Q_decode = torch.stack(decode_q_list)
element_sums = Q_decode.sum(dim=-1)  # shape: [num_decode_steps]
sum_q = 0
for i, s in enumerate(element_sums):
    print(f"Step {i+1}: sum of elements in q = {s.item()}")
    sum_q += s.item()

print(f"all_sum: {sum_q}")
