import torch 
import time
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate   # 用 HuggingFace 的 evaluate 库来算 ROUGE

from merge_keepkv import Repair
from utils_lzz import generate_token_with_past, generate_text

# -------------------- KVCache 内存计算 --------------------
def measure_kv_memory(past_key_values):
    """计算 past_key_values 占用内存 (MB)"""
    total_bytes = 0
    for k, v in past_key_values:
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024**2)


# -------------------- 单个样本评估 --------------------
def evaluate_example(model, tokenizer, prompt, reference, compress_ratio=0.2, device="mps"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        past_key_values = outputs.past_key_values
        attention = outputs.attentions
        hidden_states = outputs.hidden_states

    # -------- baseline --------
    start_time_before = time.time()
    next_token_id, past_key_values = generate_token_with_past(model, inputs)
    generated_tokens, _ = generate_text(model, 60, {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat(
            [inputs["attention_mask"], torch.tensor([[1]], device=device)], dim=1
        ),
        "past_key_values": past_key_values,
    })
    total_before = time.time() - start_time_before
    memory_before = measure_kv_memory(past_key_values)
    n_tokens_before = len(generated_tokens)
    throughput_before = n_tokens_before / max(total_before, 1e-5)

    # -------- compressed --------
    start_time_after = time.time()
    past_key_values_merged = Repair(model, past_key_values, attention, hidden_states, compress_ratio)
    generated_tokens_merged, _ = generate_text(model, 60, {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat(
            [inputs["attention_mask"], torch.tensor([[1]], device=device)], dim=1
        ),
        "past_key_values": past_key_values_merged,
    })
    total_after = time.time() - start_time_after
    memory_after = measure_kv_memory(past_key_values_merged)
    n_tokens_after = len(generated_tokens_merged)
    throughput_after = n_tokens_after / max(total_after, 1e-5)

    # print(type(generated_tokens[0]), generated_tokens[:10])

    return {
        "memory_before": memory_before,
        "memory_after": memory_after,
        "time_before": total_before,
        "time_after": total_after,
        "throughput_before": throughput_before,
        "throughput_after": throughput_after,
        "answer_before": tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True),
        "answer_after": tokenizer.decode(generated_tokens_merged, skip_special_tokens=True, clean_up_tokenization_spaces=True),
        "reference": reference,
    }

def build_prompt(dataset_name: str, text: str) -> str:
    """
    根据数据集名称构造合适的摘要 prompt。
    
    Args:
        dataset_name (str): "xsum" 或 "cnn_dailymail"
        text (str): 文章正文（xsum 用 document，cnn_dailymail 用 article）

    Returns:
        str: 格式化后的 prompt
    """
    if dataset_name.lower() == "xsum":
        prompt = (
            "Please summarize the following article in one short sentence.\n\n"
            f"Article:\n{text}\n\n"
            "Summary:"
        )
    elif dataset_name.lower() == "cnn_dailymail":
        prompt = (
            "Please summarize the following news article into 3-5 concise sentences.\n\n"
            f"Article:\n{text}\n\n"
            "Summary:"
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return prompt

# -------------------- 主流程 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/Applications/All/py_code/pythonProject/LLM-Inference-self/TinyLLama")
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--split", type=str, default="test[:1]", help="子集, 例如 test[:5] 表示前5条")
    parser.add_argument("--compress_ratio", type=float, default=0.1)
    parser.add_argument("--output_csv", type=str, default="results_keepkv_cnnDailymail.csv")
    args = parser.parse_args()

    device = "mps" 

    # 模型加载
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    # 数据集加载
    dataset = load_dataset(args.dataset, split=args.split)

    # ROUGE metric
    rouge = evaluate.load("rouge")

    results = []
    preds_before, preds_after, refs = [], [], []

    for i, example in enumerate(dataset):
        if "cnn_dailymail" in args.dataset:
            prompt = build_prompt("cnn_dailymail", example["article"])
            reference = example["highlights"]
        elif "xsum" in args.dataset:
            prompt = build_prompt("xsum", example["document"])
            reference = example["summary"]
        else:
            raise ValueError(f"Unsupported dataset")

        metrics = evaluate_example(model, tokenizer, prompt, reference, args.compress_ratio, device)
        metrics.update({
            "id": i,
            "model": args.model_name,
            "dataset": args.dataset,
        })
        results.append(metrics)

        preds_before.append(metrics["answer_before"])
        preds_after.append(metrics["answer_after"])
        refs.append(reference)

        # print(f"[{i}] {metrics}")

    # 计算ROUGE
    rouge_before = rouge.compute(predictions=preds_before, references=refs)
    rouge_after = rouge.compute(predictions=preds_after, references=refs)

    # print("ROUGE (before compression):", rouge_before)
    # print("ROUGE (after compression):", rouge_after)

    # 保存到 CSV
    df = pd.DataFrame(results)
    # 把整体ROUGE也存进去
    df.attrs["rouge_before"] = rouge_before
    df.attrs["rouge_after"] = rouge_after
    df.to_csv(args.output_csv, index=False)
    print(f"结果已保存到 {args.output_csv}")


if __name__ == "__main__":
    main()
