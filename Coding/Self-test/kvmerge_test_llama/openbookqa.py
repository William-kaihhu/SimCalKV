import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import seaborn as sns
from datasets import load_dataset
# --------------------加载模型--------------------
model_name = "/Applications/All/py_code/pythonProject/LLM-Inference-self/TinyLLama"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
device = torch.device("mps")  # 使用 MPS
model.to(device)

dataset = load_dataset("allenai/openbookqa", "main", split="train[:8]")
sample = dataset[2]  # type: ignore

question = sample["question_stem"]
choices = sample["choices"]["text"]   # type: ignore
labels = sample["choices"]["label"]   # type: ignore

# 拼接成 ChatML 格式
text = "<|system|> You are a helpful assistant. <|user|>\n"
text += f"Question: {question}\n"
for lbl, ch in zip(labels, choices):
    text += f"{lbl}: {ch}\n"
text += "<|assistant|>"
inputs = tokenizer(text, return_tensors="pt").to(device)
print("Sample Question:\n", text)
# -------------------- 函数：合并K/V（两行取平均）--------------------
def merge_kv_pairs(past_key_values, start_idx: int, end_idx: int):
    """
    从 start_idx 到 end_idx 两两合并 KV, 其余部分保持不变
    """
    merged_past = []
    for k, v in past_key_values:
        # k, v: [batch, num_heads, seq_len, head_dim]
        k_before = k[:, :, :start_idx, :]
        v_before = v[:, :, :start_idx, :]

        k_range = k[:, :, start_idx:end_idx, :]
        v_range = v[:, :, start_idx:end_idx, :]

        # 两两合并
        even_len = k_range.size(2) // 2 * 2
        k_pairs = (k_range[:, :, :even_len:2, :] + k_range[:, :, 1:even_len:2, :]) / 2
        v_pairs = (v_range[:, :, :even_len:2, :] + v_range[:, :, 1:even_len:2, :]) / 2

        # 如果长度是奇数，最后一个直接保留
        if k_range.size(2) % 2 == 1:
            k_pairs = torch.cat([k_pairs, k_range[:, :, -1:, :]], dim=2)
            v_pairs = torch.cat([v_pairs, v_range[:, :, -1:, :]], dim=2)

        k_after = k[:, :, end_idx:, :]
        v_after = v[:, :, end_idx:, :]

        new_k = torch.cat([k_before, k_pairs, k_after], dim=2)
        new_v = torch.cat([v_before, v_pairs, v_after], dim=2)

        merged_past.append((new_k, new_v))
    return merged_past

# -------------------- 生成多个token（带合并的KV缓存）--------------------
def generate_token_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        last_logits = logits[0, -1, :]
        next_token_id = last_logits.argmax()
    return next_token_id, outputs.past_key_values

def generate_token_with_past_merged(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        last_logits = logits[0, -1, :]
        next_token_id = last_logits.argmax()
        merged_kv = merge_kv_pairs(
            outputs.past_key_values,
            start_idx=12,
            end_idx=70
        )
    return next_token_id, merged_kv

generated_tokens = []
generated_tokens_merge = []
next_inputs = inputs
durations_cached_s = []
durations_cached_s_merge = []


# 正常shape
next_token_id, past_key_values = generate_token_with_past(next_inputs)
next_token = tokenizer.decode(next_token_id)
print(past_key_values[0][0].shape)

# 合并后shape
next_token_id_merge, past_key_values_merge = generate_token_with_past_merged(next_inputs)
next_token_merge = tokenizer.decode(next_token_id_merge)
next_inputs_merge = {
        "input_ids": next_token_id_merge.reshape((1, 1)),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]], device=device)],
            dim=1,
        ),
        "past_key_values": past_key_values_merge,
    }
print(past_key_values_merge[0][0].shape)

# 算内存
def calc_kv_memory(s):
    kv_bytes = 0
    for k, v in s:
        kv_bytes += k.numel() * k.element_size()
        kv_bytes += v.numel() * v.element_size()
    return kv_bytes / 1024 / 1024  # 转 MB
# KV cache 占用
kv_before_mb = calc_kv_memory(past_key_values)
kv_after_mb  = calc_kv_memory(past_key_values_merge)

print(f"\nKV cache before merge: {kv_before_mb:.2f} MB")
print(f"KV cache after merge:  {kv_after_mb:.2f} MB")

# 测试
for _ in range(4):
    t0 = time.time()
    next_token_id, past_key_values = generate_token_with_past(next_inputs)
    durations_cached_s.append(time.time() - t0)
    next_inputs = {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]], device=device)],
            dim=1
        ),
        "past_key_values": past_key_values,
    }
    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)
print("Generated tokens(before merge):", generated_tokens)

for _ in range(4):
    t0 = time.time()
    next_token_id, past_key_values = generate_token_with_past(next_inputs_merge)
    durations_cached_s_merge.append(time.time() - t0)
    next_inputs_merge = {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]], device=device)],
            dim=1
        ),
        "past_key_values": past_key_values,
    }
    next_token = tokenizer.decode(next_token_id)
    generated_tokens_merge.append(next_token)
print("Generated tokens(after merge):", generated_tokens_merge)


# --------------------绘图--------------------
plt.figure(figsize=(8, 5))
plt.plot(durations_cached_s, label="With KV Cache (No Merge)", marker='o')
plt.plot(durations_cached_s_merge, label="With KV Cache + Merge", marker='x')
plt.xlabel("Token Index")
plt.ylabel("Generation Time (s)")
plt.title("Token Generation Time Comparison")
plt.legend()
plt.grid(True)
plt.show()

