from datasets import load_dataset

# 加载 openbookqa 数据集
dataset = load_dataset("openbookqa", "main")

# 查看包含哪些子集
print(dataset)

# 获取训练集前 5 条数据
train_data = dataset["train"]
for i in range(5):
    print(f"Example {i+1}:")
    print("Question:", train_data[i]["question_stem"])
    print("Choices:", train_data[i]["choices"]["text"])
    print("Answer:", train_data[i]["answerKey"])
    print("-" * 50)
