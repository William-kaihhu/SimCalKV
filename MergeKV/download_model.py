from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# -------------------- 配置路径 --------------------
save_path = os.path.expanduser("~/Desktop/opt")
os.makedirs(save_path, exist_ok=True)

# -------------------- 选择设备 --------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

# -------------------- 加载模型和分词器 --------------------
model_name = "facebook/opt-1.3b"
print("正在下载模型和分词器...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ 推荐做法：自动分配设备，避免显存爆掉
# 如果显存不足，可以改成 load_in_4bit=True
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        # 自动分配到 MPS + CPU
    # load_in_4bit=True,      # ← 如果显存不够，取消注释启用4bit量化
)

# -------------------- 测试一下推理 --------------------
print("测试模型推理...")
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
print("模型输出:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# -------------------- 保存模型 --------------------
print(f"正在保存模型到 {save_path} ...")
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("完成！模型已保存到桌面。")
