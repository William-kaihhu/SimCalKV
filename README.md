# Efficient LLMs Inference via Similarity-Aware KV Cache Merging with Bias Calibration

This repository contains the code for the paper *Efficient LLMs Inference via Similarity-Aware KV Cache Merging with Bias Calibration*.  

<!-- 缩放图片显示 -->
<p align="center">
  <img src="analysis/Illustration.png" alt="Project Illustration" width="400">
</p>

The continuously growing key-value (KV) cache during the inference of large language models (LLMs) can become an obstacle to efficient deployment. Recently, KV cache compression techniques such as evicting and merging KV pairs have been widely studied, which play an important role in reducing memory usage during the decoding phase of LLM inference. However, these methods can introduce output perturbation and computational redundancy.

Consequently, we propose **SimCalKV**, a theoretically grounded merging strategy that identifies the optimal merging under key similarity assumptions. By averaging similar keys with a simple bias calibration and weighting values according to attention scores, SimCalKV reduces perturbation with negligible computational overhead.

## 🔧 Usage

### Quick Start
```bash
python qwen_eval.py \
    --model_name your_model_path \
    --dataset xsum \
    --split "test[:100]" \
    --compress_ratio 0.5 \
    --kv_method SimCalKV \
    --output_dir results
```
* `--dataset`: Dataset name
* `--split`: Dataset split, e.g. `"validation[:10]"` or `"test[:20]"`
* `--compress_ratio`: Compression ratio
* `--kv_method`: Compression method
* `--output_dir`: Directory to save results

## 📊 Example Runs

Summarization (CNN/DailyMail):

```bash
python qwen_eval.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset cnn_dailymail --split "test[:100]" --compress_ratio 0.4 --kv_method SimCalKV --output_dir results
```

LongBench (Gov_Report):

```bash
python llama_eval.py --model_name meta-llama/Llama-2-7b-hf --dataset THUDM/LongBench/gov_report --split "validation[:200]" --compress_ratio 0.8 --kv_method PyramidInfer --output_dir results
```

## 📈 Output

* **CSV**: per-sample results
* **JSON**: summary of memory saving, time, throughput and task metrics

Example JSON output:

```json
{
  "dataset": "cnn_dailymail",
  "kv_method": "SimCalKV",
  "metrics_before": {
    "rouge": {
      "rouge1": 0.10714013333814124,
      "rouge2": 0.02781681275643416,
      "rougeL": 0.08525080988456278,
      "rougeLsum": 0.09594273935649743
    }
  },
  "metrics_after": {
    "rouge": {
      "rouge1": 0.0648173225671974,
      "rouge2": 0.01509357533851916,
      "rougeL": 0.05245519933253466,
      "rougeLsum": 0.05725653116114888
    }
  },
  "avg_memory_saving_MB": 0.81640625,
  "time_before": 33.558887767791745,
  "time_after": 28.53268699645996,
  "avg_time_saving_s": 5.026200771331784,
  "throughput_before": 25.810928309960286,
  "throughput_after": 28.349171832657607,
  "avg_throughput_improvement_tokens_s": 2.5382435226973215
}
```

⚡ **Work in progress – under active development.**  
Updates and improvements will be added continuously.
