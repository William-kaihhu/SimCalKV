#!/bin/bash
set -e  

python llama/cnn_eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset cnn_dailymail \
    --split "test[:100]" \
    --compress_ratio 0.1 \
    --kv_method SimCalKV \
    --output_dir ./llama3.1-8b/results_cnn_token1024+64
python llama/cnn_eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset cnn_dailymail \
    --split "test[:100]" \
    --compress_ratio 0.2 \
    --kv_method SimCalKV \
    --output_dir ./llama3.1-8b/results_cnn_token1024+64
python llama/cnn_eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset cnn_dailymail \
    --split "test[:100]" \
    --compress_ratio 0.3 \
    --kv_method SimCalKV \
    --output_dir ./llama3.1-8b/results_cnn_token1024+64
python llama/cnn_eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset cnn_dailymail \
    --split "test[:100]" \
    --compress_ratio 0.4 \
    --kv_method SimCalKV \
    --output_dir ./llama3.1-8b/results_cnn_token1024+64
python llama/cnn_eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset cnn_dailymail \
    --split "test[:100]" \
    --compress_ratio 0.5 \
    --kv_method SimCalKV \
    --output_dir ./llama3.1-8b/results_cnn_token1024+64
python llama/cnn_eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset cnn_dailymail \
    --split "test[:100]" \
    --compress_ratio 0.6 \
    --kv_method SimCalKV \
    --output_dir ./llama3.1-8b/results_cnn_token1024+64
python llama/cnn_eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset cnn_dailymail \
    --split "test[:100]" \
    --compress_ratio 0.7 \
    --kv_method SimCalKV \
    --output_dir ./llama3.1-8b/results_cnn_token1024+64
python llama/cnn_eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset cnn_dailymail \
    --split "test[:100]" \
    --compress_ratio 0.8 \
    --kv_method SimCalKV \
    --output_dir ./llama3.1-8b/results_cnn_token1024+64
python llama/cnn_eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B" \
    --dataset cnn_dailymail \
    --split "test[:100]" \
    --compress_ratio 0.9 \
    --kv_method SimCalKV \
    --output_dir ./llama3.1-8b/results_cnn_token1024+64