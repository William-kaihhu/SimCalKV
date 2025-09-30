import torch
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from PyramidInfer import Decode
from utils import generate_token_with_past, generate_text
import matplotlib.pyplot as plt
import time
# -------------------- 模型加载 --------------------
model_name = "/Applications/All/py_code/pythonProject/LLM-Inference-self/TinyLLama"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
device = torch.device("mps")
model.to(device)

def measure_kv_memory(past_key_values):
    """计算 past_key_values 占用内存"""
    total_bytes = 0
    for k, v in past_key_values:
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024**2)  # MB

# -------------------- 输入 --------------------

prompt = "<|system|> \
You are a helpful assistant.\
<|user|>\
The Apollo program was the third United States human spaceflight program carried out by NASA, \
which accomplished landing the first humans on the Moon from 1969 to 1972. \
First conceived during Dwight D. Eisenhower's administration as a three-person spacecraft \
to follow the one-person Project Mercury, which put the first Americans in space, \
Apollo was later dedicated to President John F. Kennedy's national goal of \
\"landing a man on the Moon and returning him safely to the Earth\" by the end of the 1960s, \
which he proposed in an address to Congress in May 1961.\
**Question:** Which U.S. president is credited with the national goal of landing a man on the Moon?\
<|assistant|>"

print("Number of tokens:", len(tokenizer.encode(prompt)))
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    attention = outputs.attentions
    hidden_states = outputs.hidden_states # (batch, seq_len, hidden_dim)
    past_key_values = outputs.past_key_values

next_token_id, past_key_values = generate_token_with_past(model, inputs)
next_token = tokenizer.decode(next_token_id)

next_inputs = inputs
next_inputs = {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]], device=device)],
                dim=1
        ),
        "past_key_values": past_key_values,
    }
start_time_before = time.time() # 总时长-------------------------------------------------

print("--------before merge--------")
generated_tokens, durations_cached_s = generate_text(model, tokenizer, 30, next_inputs)
generated_tokens.insert(0, next_token)
print(generated_tokens)
print(durations_cached_s, "s")

end_time_before = time.time()
total_before = end_time_before - start_time_before
print("Total time before merge:", total_before, "s")

memory1 = measure_kv_memory(past_key_values)
print("Memory before merge:", memory1, "MB")

start_time_after = time.time() # 总时长----------------------------------------------------

past_key_values_merged = Decode(model, past_key_values, attention, hidden_states, 0.2)
next_inputs_merged = inputs
next_inputs_merged = {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat(
            [next_inputs_merged["attention_mask"], torch.tensor([[1]], device=device)],
                dim=1
        ),
        "past_key_values": past_key_values_merged,
    }
print("--------after merge--------")
generated_tokens_merged, durations_cached_s_merged = generate_text(model, tokenizer, 30, next_inputs_merged)
generated_tokens_merged.insert(0, next_token)
print(generated_tokens_merged)
print(durations_cached_s_merged, "s")

end_time_after = time.time()
total_after = end_time_after - start_time_after
print("Total time after merge:", total_after, "s")

memory2 = measure_kv_memory(past_key_values_merged)
print("Memory after merge:", memory2, "MB")