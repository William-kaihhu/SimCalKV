import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# -------------------- 加载 Mistral (MPS) --------------------
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("mps")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=None
).to(device)
model.eval()

# -------------------- 加载 COPA --------------------
copa = load_dataset("super_glue", "copa", split="validation[:5]")  # 前5个样本

# -------------------- KV 合并函数 --------------------
def merge_kv_pairs(past_key_values, keep_ratio: float = 0.2):
    merged_past = []
    for k, v in past_key_values:
        seq_len = k.size(2)
        if seq_len < 20:  # 太短就不合并
            merged_past.append((k, v))
            continue

        start_idx = int(keep_ratio * seq_len)
        end_idx   = int((1 - keep_ratio) * seq_len)

        k_before, v_before = k[:, :, :start_idx, :], v[:, :, :start_idx, :]
        k_range, v_range   = k[:, :, start_idx:end_idx, :], v[:, :, start_idx:end_idx, :]

        even_len = k_range.size(2) // 2 * 2
        k_pairs = (k_range[:, :, :even_len:2, :] + k_range[:, :, 1:even_len:2, :]) / 2
        v_pairs = (v_range[:, :, :even_len:2, :] + v_range[:, :, 1:even_len:2, :]) / 2

        if k_range.size(2) % 2 == 1:
            k_pairs = torch.cat([k_pairs, k_range[:, :, -1:, :]], dim=2)
            v_pairs = torch.cat([v_pairs, v_range[:, :, -1:, :]], dim=2)

        k_after, v_after = k[:, :, end_idx:, :], v[:, :, end_idx:, :]
        new_k = torch.cat([k_before, k_pairs, k_after], dim=2)
        new_v = torch.cat([v_before, v_pairs, v_after], dim=2)

        merged_past.append((new_k, new_v))
    return merged_past

# -------------------- KV 内存计算 --------------------
def calc_kv_memory(past_key_values):
    kv_bytes = 0
    for k, v in past_key_values:
        kv_bytes += k.numel() * k.element_size()
        kv_bytes += v.numel() * v.element_size()
    return kv_bytes / 1024 / 1024  # 转成 MB

# -------------------- 正常推理 --------------------
def generate_choice(prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    kv_mem = calc_kv_memory(outputs.past_key_values)

    generated = []
    next_token_id = outputs.logits[:, -1, :].argmax().unsqueeze(0).unsqueeze(0)
    past = outputs.past_key_values

    for _ in range(max_new_tokens):
        next_inputs = {
            "input_ids": next_token_id,
            "attention_mask": torch.cat(
                [inputs["attention_mask"], torch.tensor([[1]], device=device)], dim=1
            ),
            "past_key_values": past,
        }
        with torch.no_grad():
            outputs = model(**next_inputs)
        next_token_id = outputs.logits[:, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        past = outputs.past_key_values
        generated.append(tokenizer.decode(next_token_id[0]))

    return " ".join(generated).strip(), kv_mem

# -------------------- 合并 KV 的推理 --------------------
def generate_choice_with_merge(prompt, max_new_tokens=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 第一步 forward
    with torch.no_grad():
        outputs = model(**inputs)
    past = merge_kv_pairs(outputs.past_key_values, keep_ratio=0.2)
    kv_mem = calc_kv_memory(past)

    generated = []
    next_token_id = outputs.logits[:, -1, :].argmax().unsqueeze(0).unsqueeze(0)

    for _ in range(max_new_tokens):
        next_inputs = {
            "input_ids": next_token_id,
            "attention_mask": torch.cat(
                [inputs["attention_mask"], torch.tensor([[1]], device=device)], dim=1
            ),
            "past_key_values": past,
        }
        with torch.no_grad():
            outputs = model(**next_inputs)
        next_token_id = outputs.logits[:, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        past = outputs.past_key_values
        generated.append(tokenizer.decode(next_token_id[0]))

    return "".join(generated).strip(), kv_mem

# -------------------- 跑实验 --------------------
acc_before, acc_after = [], []
for i, ex in enumerate(copa):
    premise, q_type = ex["premise"], ex["question"]
    c1, c2, label = ex["choice1"], ex["choice2"], ex["label"]

    prompt = f"<s>[INST] Premise: {premise}\n" \
             f"Question type: {q_type}\n" \
             f"Choice1: {c1}\n" \
             f"Choice2: {c2}\n" \
             f"Which choice is more plausible, 1 or 2? " \
             f"Answer ONLY with '1' or '2'. [/INST]"

    ans_before, kv_before = generate_choice(prompt)
    ans_after, kv_after = generate_choice_with_merge(prompt)

    pred_before = 1 if "1" in ans_before else (2 if "2" in ans_before else -1)
    pred_after  = 1 if "1" in ans_after  else (2 if "2" in ans_after else -1)

    acc_before.append(int(pred_before == label))
    acc_after.append(int(pred_after == label))

    print(f"\n=== Example {i} ===")
    print("Premise:", premise)
    print("Q type:", q_type)
    print("Choice1:", c1)
    print("Choice2:", c2)
    print("Label:", label)
    print(f"Answer (before merge): {ans_before} => {pred_before} | KV {kv_before:.2f} MB")
    print(f"Answer (after  merge): {ans_after} => {pred_after} | KV {kv_after:.2f} MB")

print("\n=== Accuracy ===")
print("Before merge:", np.mean(acc_before))
print("After  merge:", np.mean(acc_after))
