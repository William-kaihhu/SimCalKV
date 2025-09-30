import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
# --------------------第一阶段：逐步生成token（使用GPT2）--------------------
model_name = "/Applications/All/py_code/pythonProject/LLM-Inference-self/TinyLlama"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
 
# 打印模型信息
print("Model Information:")
print(model)
 
prompt = "The quick brown fox jumped over the"
inputs = tokenizer(prompt, return_tensors="pt")
 
# `inputs` 是一个字典：{'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}
# input_ids 的形状为 (batch_size, sequence_length)，其中 batch_size=1，sequence_length=7（token 数）
print()
print("Inputs:")
print(inputs)
 
with torch.no_grad():
    outputs = model(**inputs)
 
# logits 的形状是 (batch_size, sequence_length, vocab_size)，
# vocab_size 是 GPT-2 的词汇表大小（约 50257）
logits = outputs.logits
print()
print(logits.shape)
 
# 提取第一个样本（batch=0）最后一个位置（-1）对应的 logits 向量
last_logits = logits[0, -1, :]
next_token_id = last_logits.argmax()  # 概率最大的下一个 token 的 ID
print()
print(next_token_id)
 
next_word = tokenizer.decode(next_token_id)  # 解码 token ID 为文本
print()
print(next_word)
 
# `torch.topk(last_logits, k=10)` 返回一个元组 (values, indices)，
# 其中 values 是 last_logits 中得分最高的前 10 个值，indices 是对应的 token ID
top_k = torch.topk(last_logits, k=10)
tokens = [tokenizer.decode(tk) for tk in top_k.indices]
related_values = [values for values in top_k.values]
 
# tokens 列表存储最可能的 10 个下一个 token（或 token 片段）文本表示
print()
print(tokens)
print(related_values)
 
# --------------------第二阶段：生成多个token，并显示用时--------------------
 
# 定义 token 生成函数
def generate_token(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()
    return next_token_id
 
generated_tokens = []
next_inputs = inputs
durations_s = []  # 每个 token 生成所花费的时间（秒）
 
for _ in range(10):
    t0 = time.time()
    next_token_id = generate_token(next_inputs)
    durations_s += [time.time() - t0]
    # 将新生成的 token 添加到已有输入序列中
    next_inputs = {
        "input_ids": torch.cat(
            [next_inputs["input_ids"], next_token_id.reshape((1, 1))],
            dim=1),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]])],
            dim=1),
    }
    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)
 
print()
print(f"{sum(durations_s)} s")  # 总共生成 token 所花费的时间
print(generated_tokens)
 
# --------------------第三阶段：使用 KV 缓存加速生成--------------------
 
def generate_token_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()
    return next_token_id, outputs.past_key_values
 
generated_tokens = []
next_inputs = inputs
durations_cached_s = []
 
for _ in range(10):
    t0 = time.time()
    next_token_id, past_key_values = generate_token_with_past(next_inputs)
    durations_cached_s += [time.time() - t0]
    next_inputs = {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]])],
            dim=1),
        "past_key_values": past_key_values,
    }
    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)
 
print(f"{sum(durations_cached_s)} s")
print(generated_tokens)
 
# 可视化每个 token 的生成时间
# 可以观察到随着序列增长，生成每个 token 的时间逐渐增加；
# 第一个 token 所花时间通常也较长，可能是因为缓存预热。
# 明显看到使用 KV 缓存能显著减少生成时间。
plt.plot(durations_s, label="Without KV Cache")
plt.plot(durations_cached_s, label="With KV Cache")
plt.xlabel("Token Index")
plt.ylabel("Generation Time (s)")
plt.legend()
plt.title("Token Generation Time Comparison")
plt.show()
