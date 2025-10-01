# Efficient LLMs Inference via Similarity-Aware KV Cache Merging with Bias Calibration

This repository contains the code for the paper *Efficient LLMs Inference via Similarity-Aware KV Cache Merging with Bias Calibration*.  

<!-- 缩放图片显示 -->
<p align="center">
  <img src="analysis/Illustration.png" alt="Project Illustration" width="400">
</p>

The continuously growing key-value (KV) cache during the inference of large language models (LLMs) can become an obstacle to efficient deployment. Recently, KV cache compression techniques such as evicting and merging KV pairs have been widely studied, which play an important role in reducing memory usage during the decoding phase of LLM inference. However, these methods can introduce output perturbation and computational redundancy.

Consequently, we propose **SimCalKV**, a theoretically grounded merging strategy that identifies the optimal merging under key similarity assumptions. By averaging similar keys with a simple bias calibration and weighting values according to attention scores, SimCalKV reduces perturbation with negligible computational overhead.

⚡ **Work in progress – under active development.**  
Updates and improvements will be added continuously.
