from datasets import load_dataset

dataset = load_dataset("ai2-adapt-dev/synth-wikipedia-questions-sample", split="train[:1]")

def synthwiki_to_prompt(sample):
    question = sample["prompt"]
    # answer = sample["response"]  # 如果想也包含答案可以加
    return f"Question: {question}\nAnswer:"

prompts = [synthwiki_to_prompt(sample) for sample in dataset]

print(prompts[0])
