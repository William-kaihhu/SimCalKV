import torch 
import time
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate   # 用 HuggingFace 的 evaluate 库来算 ROUGE
import json
import os
from merge_ours import Repair
from utils_lzz import generate_token_with_past, generate_text

# -------------------- KVCache 内存计算 --------------------
def measure_kv_memory(past_key_values):
    """计算 past_key_values 占用内存 (MB)"""
    total_bytes = 0
    for k, v in past_key_values:
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024**2)

# -------------------- KV 方法选择 --------------------
def apply_kv_method(method, model, past_key_values, attention, hidden_states, compress_ratio):
    if method == "ours":
        return Repair(model, past_key_values, attention, hidden_states, compress_ratio)
    elif method == "KeepKV":
        from merge_keepkv import KeepKV
        return KeepKV(model, past_key_values, attention, hidden_states, compress_ratio)
    elif method == "CaM":
        from cam import CaM
        return CaM(model, past_key_values, compress_ratio)
    elif method == "PyramidInfer":
        from eviction_pyramid import Decode
        return Decode(model, past_key_values, attention, hidden_states, compress_ratio)
    else:
        raise ValueError(f"Unknown KV method: {method}")

# -------------------- 单个样本评估 --------------------
def evaluate_example(model, tokenizer, prompt, reference, method="ours", compress_ratio=0.1, device="mps"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)

    # 存储五次推理结果的列表
    memory_before_list, time_before_list, throughput_before_list = [], [], []
    memory_after_list, time_after_list, throughput_after_list = [], [], []
    answer_before, answer_after = "", ""

    for i in range(1):  # 进行五次推理
        print(f"  > Performing inference {i + 1}/1...")

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
            past_key_values = outputs.past_key_values
            attention = outputs.attentions
            hidden_states = outputs.hidden_states

        # -------- baseline --------
        next_token_id, past_key_values = generate_token_with_past(model, inputs)
        start_time_before = time.time()
        generated_tokens, durations_cached_s = generate_text(model, tokenizer, 500, {
            "input_ids": next_token_id.reshape((1, 1)),
            "attention_mask": torch.cat(
                [inputs["attention_mask"], torch.tensor([[1]], device=device)], dim=1
            ),
            "past_key_values": past_key_values,
        })
        total_before = time.time() - start_time_before
        memory_before = measure_kv_memory(past_key_values)
        n_tokens_before = len(generated_tokens)
        total_generation_time_before = sum(durations_cached_s)
        throughput_before = n_tokens_before / total_generation_time_before

        memory_before_list.append(memory_before)
        time_before_list.append(total_before)
        throughput_before_list.append(throughput_before)
        if i == 0:  # 只保存第一次生成的文本，因为每次生成的文本内容应该是一致的
            answer_before = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # -------- compressed --------
        # past_key_values_merged = Repair(model, past_key_values, attention, hidden_states, compress_ratio)
        start_time_after = time.time()
        past_key_values_merged = apply_kv_method(method, model, past_key_values, attention, hidden_states, compress_ratio)
        generated_tokens_merged, durations_cached_s_merged = generate_text(model, tokenizer, 500, {
            "input_ids": next_token_id.reshape((1, 1)),
            "attention_mask": torch.cat(
                [inputs["attention_mask"], torch.tensor([[1]], device=device)], dim=1
            ),
            "past_key_values": past_key_values_merged,
        })
        total_after = time.time() - start_time_after
        memory_after = measure_kv_memory(past_key_values_merged)
        n_tokens_after = len(generated_tokens_merged)
        total_generation_time_after = sum(durations_cached_s_merged)
        throughput_after = n_tokens_after / total_generation_time_after

        memory_after_list.append(memory_after)
        time_after_list.append(total_after)
        throughput_after_list.append(throughput_after)
        if i == 0:  # 只保存第一次生成的文本
            answer_after = tokenizer.decode(generated_tokens_merged, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # 计算平均值
    avg_memory_before = sum(memory_before_list) / len(memory_before_list)
    avg_time_before = sum(time_before_list) / len(time_before_list)
    avg_throughput_before = sum(throughput_before_list) / len(throughput_before_list)

    avg_memory_after = sum(memory_after_list) / len(memory_after_list)
    avg_time_after = sum(time_after_list) / len(time_after_list)
    avg_throughput_after = sum(throughput_after_list) / len(throughput_after_list)
    
    return {
        "memory_before": avg_memory_before,
        "memory_after": avg_memory_after,
        "time_before": avg_time_before,
        "time_after": avg_time_after,
        "throughput_before": avg_throughput_before,
        "throughput_after": avg_throughput_after,
        "answer_before": answer_before,
        "answer_after": answer_after,
        "reference": reference,
        "compression_ratio": compress_ratio
    }

def load_task_dataset(name, split="validation[:10]"):
    try:
        if name == "cnn_dailymail":
            ds = load_dataset(name, "3.0.0", split=split, trust_remote_code=True)
        elif "THUDM/LongBench" in name:
            subtask = name.split("/")[-1]
            ds = load_dataset("THUDM/LongBench", subtask, split=split, trust_remote_code=True)
        elif name =='piqa':
            ds = load_dataset('piqa', split="validation[:1000]", trust_remote_code=True)
        elif name == "winogrande":
            ds = load_dataset("winogrande", "winogrande_debiased", split=split, trust_remote_code=True)
        else:
            ds = load_dataset(name, split=split, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"加载数据集 {name} 失败: {str(e)}")
    samples = []

    # 摘要类
    if name == "xsum":
        for ex in ds:
            prompt = (
                "Please summarize the following article in one short sentence.\n\n"
                f"Article:\n{ex['document']}\n\nSummary:"
            )
            samples.append({"dataset": "xsum", "prompt": prompt, "reference": ex["summary"]})

    elif name == "cnn_dailymail":
        for ex in ds:
            prompt = (
                "Please summarize the following news article into 3-5 concise sentences.\n\n"
                f"Article:\n{ex['article']}\n\nSummary:"
            )
            samples.append({"dataset": "cnn_dailymail", "prompt": prompt, "reference": ex["highlights"]})

    elif name == "THUDM/LongBench/gov_report":
        for ex in ds:
            prompt = (
                "Summarize the following government report into a concise abstract.\n\n"
                f"Report:\n{ex['input']}\n\nSummary:"
            )
            samples.append({"dataset": "govreport", "prompt": prompt, "reference": ex["answers"]})

    # QA / 多选
    elif name == "piqa":
        for ex in ds:
            prompt = f"Question:\n{ex['goal']}\n\nAnswer with 1 or 2:"
            reference = ex[f"sol{ex['label'] + 1}"]
            samples.append({"dataset": "piqa", "prompt": prompt, "reference": reference})


    elif name == "openbookqa":
        for ex in ds:
            choices_text = "\n".join(
                [f"{label}. {text}" for label, text in zip(ex["choices"]["label"], ex["choices"]["text"])]
            )
            prompt = (
                f"Question:\n{ex['question_stem']}\n\n"
                f"Choices:\n{choices_text}\n\n"
                "Answer with only the letter (A, B, C, D):"
            )
            samples.append({
                "dataset": "openbookqa",
                "prompt": prompt,
                "reference": ex['answerKey']
            })

    elif name == "winogrande":
        for ex in ds:
            prompt = (
                f"Sentence with blank:\n{ex['sentence']}\n\n"
                f"Fill in the blank with the correct option.\n\nOption1:{ex['option1']}\nOption2:{ex['option2']}\nAnswer:"
            )
            samples.append({"dataset": "winogrande", "prompt": prompt, "reference": int(ex["answer"].lower()) - int('a')})

    elif name == "THUDM/LongBench/2wikimqa":
        for ex in ds:
            prompt = (
                f"Read the following passages and answer the multi-hop question.\n\n"
                f"Passages:\n{ex['context']}\n\nQuestion:\n{ex['input']}\n\nAnswer:"
            )
            samples.append({"dataset": "2wikimqa", "prompt": prompt, "reference": ex["answers"][0]})

    elif name == "THUDM/LongBench/narrativeqa":
        for ex in ds:
            prompt = (
                f"Read the following story and answer the question.\n\n"
                f"Story:\n{ex['context']}\n\nQuestion:\n{ex['input']}\n\nAnswer:"
            )
            samples.append({"dataset": "narrativeqa", "prompt": prompt, "reference": ex["answers"][0]})

    elif name == "THUDM/LongBench/hotpotqa":
        for ex in ds:
            prompt = (
                f"Read the following passages and answer the question.\n\n"
                f"Passages:\n{ex['context']}\n\nQuestion:\n{ex['input']}\n\nAnswer:"
            )
            samples.append({"dataset": "hotpotqa", "prompt": prompt, "reference": ex["answers"][0]})

    elif name == "THUDM/LongBench/multifieldqa_zh":
        for ex in ds:
            prompt = (
                f"阅读以下文档并回答问题。\n\n"
                f"文档：\n{ex['context']}\n\n问题：\n{ex['input']}\n\n回答："
            )
            samples.append({"dataset": "multifieldqa_zh", "prompt": prompt, "reference": ex["answers"][0]})

    # 代码生成
    elif name == "THUDM/LongBench/lcc":
        for ex in ds:
            prompt = f"Passages:\n{ex['context']}\n\nWrite a {ex['language']} function to solve the following problem:\n\n{ex['input']}\n\nAnswer:"
            samples.append({"dataset": "lcc", "prompt": prompt, "reference": ex["answers"]})

    elif name == "THUDM/LongBench/repobench-p":
        for ex in ds:
            prompt = f"Passages:\n{ex['context']}\n\nWrite a {ex['language']} function to solve the following problem:\n\n{ex['input']}\n\nAnswer:"
            samples.append({"dataset": "repobench-p", "prompt": prompt, "reference": ex["answers"]})

    # 分类
    elif name == "THUDM/LongBench/trec":
        for ex in ds:
            prompt = (
                f"Quesion:\n\n{ex['input']}\n\n"
                f"Choices:\n{ex['context']}\n\nAnswer:"
            )
            samples.append({"dataset": "trec", "prompt": prompt, "reference": ex["answers"][0]})

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return samples

# -------------------- 主流程 --------------------
import argparse
import os
import json
import pandas as pd
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设你已有的函数
# from your_module import load_task_dataset, evaluate_example  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/Users/chengyukai/Desktop/qwen2.5-1.5B-Instruct")
    parser.add_argument("--split", type=str, default="test[:5]")
    parser.add_argument("--compress_ratio", type=float, default=0.2)
    parser.add_argument("--kv_method", type=str, default="ours", choices=["ours", "KeepKV", "CaM", "PyramidInfer"])
    parser.add_argument("--output_dir", type=str, default="results")  # 输出目录
    args = parser.parse_args()

    # 自动创建目录
    os.makedirs(args.output_dir, exist_ok=True)
    device = "mps"
    print(f"Results will be saved to: {args.output_dir}")

    # 定义要运行的数据集列表
    datasets = [
        #"xsum",
        #"cnn_dailymail",
        #"THUDM/LongBench/gov_report", # Summarization
        #"THUDM/LongBench/lcc",        # Code Python/C#/Java
        "THUDM/LongBench/repobench-p" # Code Python/Java
    ]

    DATASET_METRICS = {
        "xsum": ["rouge"],
        "cnn_dailymail": ["rouge"],
        "piqa": ["accuracy"],
        "openbookqa": ["accuracy"],
        "winogrande_debiased": ["accuracy"],
        "2wikimqa": ["f1"],
        "narrativeqa": ["f1"],
        "gov_report": ["rouge"],
        "hotpotqa": ["f1"],
        "trec": ["accuracy"],
        "multifieldqa_zh": ["f1"],
        "repobench-p": ["bleu"],
        "lcc": ["bleu"],
    }
    
    # 模型加载
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device,
        torch_dtype="auto",
        attn_implementation="eager"
    )
    model.eval()
    print(">>> 模型加载完成")

    for dataset in datasets:
        print(f"\n>>> 开始处理数据集: {dataset}")
        try:
            # 数据集加载
            samples = load_task_dataset(dataset, args.split)

            # 加载指标
            dataset_name = dataset.split("/")[-1] if "/" in dataset else dataset
            metrics_to_use = DATASET_METRICS.get(dataset_name, [])
            metric_evaluators = {m: evaluate.load(m) for m in metrics_to_use}

            results = []
            preds_before, preds_after, refs = [], [], []

            for i, example in enumerate(samples):
                metrics = evaluate_example(
                    model, tokenizer,
                    example["prompt"], example["reference"],
                    args.kv_method, args.compress_ratio, device
                )
                metrics.update({
                    "id": i,
                    "model": args.model_name,
                    "dataset": dataset,
                    "kv_method": args.kv_method
                })
                results.append(metrics)
                preds_before.append(metrics["answer_before"])
                preds_after.append(metrics["answer_after"])
                refs.append(example["reference"])

            # 计算指标
            score_before, score_after = {}, {}
            for metric_name, evaluator in metric_evaluators.items():
                if metric_name == "rouge":
                    score_before[metric_name] = evaluator.compute(predictions=preds_before, references=refs)
                    score_after[metric_name] = evaluator.compute(predictions=preds_after, references=refs)
                elif metric_name in ["accuracy", "f1", "bleu"]:
                    score_before[metric_name] = evaluator.compute(predictions=preds_before, references=refs)[metric_name]
                    score_after[metric_name] = evaluator.compute(predictions=preds_after, references=refs)[metric_name]

            # 保存精简的 CSV
            dataset_name = dataset.split("/")[-1] if "/" in dataset else dataset
            output_csv = f"{args.output_dir}/results_{dataset_name}_{args.kv_method}.csv"
            df = pd.DataFrame([{
                "id": r["id"],
                "dataset": r["dataset"],
                "kv_method": r["kv_method"],
                "memory_before_MB": r["memory_before"],
                "memory_after_MB": r["memory_after"],
                "time_before_s": r["time_before"],
                "time_after_s": r["time_after"],
                "throughput_before_tokens_s": r["throughput_before"],
                "throughput_after_tokens_s": r["throughput_after"],
                "answer_before": r["answer_before"],
                "answer_after": r["answer_after"]
            } for r in results])
            df.to_csv(output_csv, index=False)
            print(f"结果已保存到 {output_csv}")

            # 保存指标到 JSON
            summary_json = f"{args.output_dir}/qwen_summary_{dataset_name}_{args.kv_method}_{args.compress_ratio}.json"
            summary = {
                "dataset": dataset,
                "kv_method": args.kv_method,
                "metrics_before": score_before,
                "metrics_after": score_after,
                "avg_memory_saving_MB": df["memory_before_MB"].mean() - df["memory_after_MB"].mean(),
                "time_before": df["time_before_s"].mean(),
                "time_after":  df["time_after_s"].mean(),
                "avg_time_saving_s": df["time_before_s"].mean() - df["time_after_s"].mean(),
                "throughput_before": df["throughput_before_tokens_s"].mean(),
                "throughput_after": df["throughput_after_tokens_s"].mean(),
                "avg_throughput_improvement_tokens_s": df["throughput_after_tokens_s"].mean() - df["throughput_before_tokens_s"].mean()
            }
            with open(summary_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"指标汇总已保存到 {summary_json}")
        
        except Exception as e:
            print(f"处理数据集 {dataset} 时出错: {str(e)}")
            continue


if __name__ == "__main__":
    main()
