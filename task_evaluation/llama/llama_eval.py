import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch 
import time
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate   
import json
import os
from merge_ours import Repair
from merge_lerp import Repair_lerp
from utils import generate_token_with_past, generate_text


from merge_cam import CaMPress

# -------------------- KV Cache Memory Calculation --------------------
def measure_kv_memory(past_key_values):
    total_bytes = 0
    for k, v in past_key_values:
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024**2)

# -------------------- Compression methods --------------------

def Repair_CaM(model, past_key_values, attention, hidden_states, compress_ratio):
    """
    A wrapper function to iterate through all layers and apply CaMPress.
    CaM_press.py is designed to operate layer by layer.
    """
    new_past_key_values = []
    cam_press_instance = CaMPress(compression_ratio=compress_ratio)
    num_layers = len(past_key_values)
    
    for i in range(num_layers):
        # Extract the current layer's keys, values, and attention
        layer_keys = past_key_values[i][0]
        layer_values = past_key_values[i][1]
        # 'attention' is a tuple containing attention tensors for all layers
        layer_attentions = attention[i]
        
        # Call the .compress() method in CaM_press.py
        # As seen in CaM_press.py, if 'attentions' is provided,
        # 'module', 'hidden_states', and 'kwargs' are not required
        # since _compute_full_attention will be skipped.
        new_k, new_v = cam_press_instance.compress(
            module=None,          # Not needed, since attentions are provided directly
            hidden_states=None,   # Not needed
            keys=layer_keys,
            values=layer_values,
            attentions=layer_attentions,
            kwargs=None           # Not needed
        )
        
        new_past_key_values.append((new_k, new_v))
    
    # Return a tuple containing the new K/V pairs for all layers
    return tuple(new_past_key_values)


# -------------------- Compression methods --------------------
def apply_kv_method(method, model, past_key_values, attention, hidden_states, compress_ratio):
    if method == "SimCalKV":
        return Repair(model, past_key_values, attention, hidden_states, compress_ratio, 0)
    elif method == "KeepKV":
        from merge_keepkv import KeepKV
        return KeepKV(model, past_key_values, attention, hidden_states, compress_ratio, 0)
    elif method == "PyramidInfer":
        from eviction_pyramid import Decode
        return Decode(model, past_key_values, attention, hidden_states, compress_ratio)
    elif method == "CaM":
        return Repair_CaM(model, past_key_values, attention, hidden_states, compress_ratio)
    else:
        raise ValueError(f"Unknown KV method: {method}")

# -------------------- Evaluation --------------------
def evaluate_example(model, tokenizer, prompt, reference, method="SimCalKV", compress_ratio=0.1, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    # save the results
    memory_before_list, time_before_list, throughput_before_list = [], [], []
    memory_after_list, time_after_list, throughput_after_list = [], [], []
    answer_before, answer_after = "", ""

    for i in range(1):  # Inference
        print(f"  > Performing inference {i + 1}/1...")
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
            past_key_values = outputs.past_key_values
            attention = outputs.attentions
            hidden_states = outputs.hidden_states

        # -------- baseline --------
        start_time_before = time.time()
        next_token_id, past_key_values = generate_token_with_past(model, inputs)
        total_kv, _, generated_tokens, durations_cached_s, _, _ = generate_text(model, tokenizer, 512, {
            "input_ids": next_token_id.reshape((1, 1)),
            "attention_mask": torch.cat(
                [inputs["attention_mask"], torch.tensor([[1]], device=device)], dim=1
            ),
            "past_key_values": past_key_values,
        })
        total_before = time.time() - start_time_before
        memory_before = measure_kv_memory(total_kv)
        n_tokens_before = len(generated_tokens)
        total_generation_time_before = sum(durations_cached_s)
        throughput_before = n_tokens_before / total_generation_time_before

        memory_before_list.append(memory_before)
        time_before_list.append(total_before)
        throughput_before_list.append(throughput_before)
        if i == 0:  
            answer_before = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # -------- compressed --------
        # past_key_values_merged = Repair(model, past_key_values, attention, hidden_states, compress_ratio)
        
        kv_decode_merged = None
        total_kv_merged_1 = None
        total_kv_merged = None
        generated_tokens_merged_1 = []
        generated_tokens_merged = []
        durations_cached_s_merged_1 = []
        durations_cached_s_merged = []
        attention_mask1 = torch.tensor([[1]])
        next_inputs = inputs
        #past_key_values_merged = past_key_values
        start_time_after = time.time()
        past_key_values_merged = apply_kv_method(method, model, past_key_values, attention, hidden_states, compress_ratio)
        
        for j in range(2):
            if j == 0:
                next_inputs = {
                    "input_ids": next_token_id.reshape((1, 1)),
                    "attention_mask": torch.cat(
                        [inputs["attention_mask"], torch.tensor([[1]], device=device)], dim=1
                    ),
                    "past_key_values": past_key_values_merged,
                }
                total_kv_merged_1, attention_mask1, generated_tokens_merged_1, durations_cached_s_merged_1, attention_merged, hidden_states_merged = \
                    generate_text(model, tokenizer, 256, next_inputs)
                kv_decode_merged = total_kv_merged_1
                #kv_decode_merged = apply_kv_method(method, model, total_kv_merged_1, attention_merged, hidden_states_merged, compress_ratio)
            else:
                next_inputs = {
                    "input_ids": generated_tokens_merged_1[-1].reshape((1, 1)),
                    "attention_mask": torch.cat(
                        [attention_mask1, torch.tensor([[1]], device=device)], dim=1
                    ),
                    "past_key_values": kv_decode_merged,
                }
                total_kv_merged, _, generated_tokens_merged, durations_cached_s_merged, attention_merged, hidden_states_merged = \
                    generate_text(model, tokenizer, 256, next_inputs)
                #kv_decode_merged = apply_kv_method(method, model, total_kv_merged, attention_merged, hidden_states_merged, compress_ratio)
        total_after = time.time() - start_time_after
        
        memory_after = measure_kv_memory(total_kv_merged)
        n_tokens_after = len(generated_tokens_merged_1) + len(generated_tokens_merged)
        total_generation_time_after = sum(durations_cached_s_merged_1)+ sum(durations_cached_s_merged)
        throughput_after = n_tokens_after / total_generation_time_after

        memory_after_list.append(memory_after)
        time_after_list.append(total_after)
        throughput_after_list.append(throughput_after)
        all_generated_tokens = generated_tokens_merged_1 + generated_tokens_merged
        if i == 0: 
            answer_after = tokenizer.decode(all_generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

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
            ds = load_dataset(name, "3.0.0", split=split)
        elif "LongBench" in name:
            subtask = name.split("/")[-1]
            local_file = f"LongBench/data/{subtask}.jsonl"  # local file
            if not os.path.exists(local_file):
                raise ValueError(f"Local file {local_file} not found")
            ds = load_dataset("json", data_files=local_file, split=split)
        elif name =='piqa':
            ds = load_dataset('piqa', split="validation[:1000]", trust_remote_code=True)
        elif name == "winogrande":
            ds = load_dataset("winogrande", "winogrande_debiased", split=split, trust_remote_code=True)
        else:
            ds = load_dataset(name, split=split, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"Fail to load dataset{name}: {str(e)}")
    samples = []

    # summarization
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

    elif name == "LongBench/gov_report":
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


    # QA
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
                f"Read the document and answer the question。\n\n"
                f"Document:\n{ex['context']}\n\nQuestion:\n{ex['input']}\n\nAnswer:"
            )
            samples.append({"dataset": "multifieldqa_zh", "prompt": prompt, "reference": ex["answers"][0]})

    # code generation
    elif name == "LongBench/lcc":
        for ex in ds:
            prompt = f"Passages:\n{ex['context']}\n\nWrite a {ex['language']} function to solve the following problem:\n\n{ex['input']}\n\nAnswer:"
            samples.append({"dataset": "lcc", "prompt": prompt, "reference": ex["answers"]})

    elif name == "LongBench/repobench-p":
        for ex in ds:
            prompt = f"Passages:\n{ex['context']}\n\nWrite a {ex['language']} function to solve the following problem:\n\n{ex['input']}\n\nAnswer:"
            samples.append({"dataset": "repobench-p", "prompt": prompt, "reference": ex["answers"]})

    # classification
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

# from your_module import load_task_dataset, evaluate_example  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="xsum", help="Dataset names, e.g. xsum piqa openbookqa winogrande")
    parser.add_argument("--split", type=str, default="test[:5]")
    parser.add_argument("--compress_ratio", type=float, default=0.4)
    parser.add_argument("--kv_method", type=str, default="SimCalKV", 
                        choices=["SimCalKV", "KeepKV", "PyramidInfer", "CaM"])
    parser.add_argument("--output_dir", type=str, default="results")  
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"
    print(f"Results will be saved to: {args.output_dir}")

    
    datasets = [args.dataset]
    # list of datasets to be evaluated
    #datasets = [
        #"xsum",
        #"cnn_dailymail",
        #"THUDM/LongBench/gov_report", # Summarization
        #"THUDM/LongBench/lcc",        # Code Python/C#/Java
        #"THUDM/LongBench/repobench-p" # Code Python/Java
    #]

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
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="cuda",
        #device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager"
    )
    model.eval()
    print(">>> Model loading Completed")

    for dataset in datasets:
        print(f"\n>>> Processing dataset: {dataset}")
        try:
            samples = load_task_dataset(dataset, args.split)

            # load the metrics
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

            # computation
            score_before, score_after = {}, {}
            for metric_name, evaluator in metric_evaluators.items():
                if metric_name == "rouge":
                    score_before[metric_name] = evaluator.compute(predictions=preds_before, references=refs)
                    score_after[metric_name] = evaluator.compute(predictions=preds_after, references=refs)
                elif metric_name in ["accuracy", "f1", "bleu"]:
                    score_before[metric_name] = evaluator.compute(predictions=preds_before, references=refs)[metric_name]
                    score_after[metric_name] = evaluator.compute(predictions=preds_after, references=refs)[metric_name]

            # save the results(csv)
            dataset_name = dataset.split("/")[-1] if "/" in dataset else dataset
            output_csv = f"{args.output_dir}/llama_results_{dataset_name}_{args.kv_method}.csv"
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
            print(f"The results are saved in  {output_csv}")

            # save the results(JSON)
            summary_json = f"{args.output_dir}/llama_{dataset_name}_{args.kv_method}_{args.compress_ratio}.json"
            summary = {
                "dataset": dataset,
                "kv_method": args.kv_method,
                "model_name": args.model_name,
                "compress_ratio": args.compress_ratio, 
                "metrics_before": score_before,
                "metrics_after": score_after,
                "avg_memory_saving_MB": df["memory_before_MB"].mean() - df["memory_after_MB"].mean(),
                "time_before": df["time_before_s"].sum(),
                "time_after":  df["time_after_s"].sum(),
                "avg_time_saving_s": df["time_before_s"].sum() - df["time_after_s"].sum(),
                "throughput_before": df["throughput_before_tokens_s"].mean(),
                "throughput_after": df["throughput_after_tokens_s"].mean(),
                "avg_throughput_improvement_tokens_s": df["throughput_after_tokens_s"].mean() - df["throughput_before_tokens_s"].mean()
            }
            with open(summary_json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"The complete results are saved in {summary_json}")
        
        except Exception as e:
            print(f"Fail to process {dataset}: {str(e)}")
            continue


if __name__ == "__main__":
    main()

