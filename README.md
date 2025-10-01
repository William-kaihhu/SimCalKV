# Efficient LLMs Inference via Similarity-Aware KV Cache Merging with Bias Calibration

This repository contains the code for the paper *Efficient LLMs Inference via Similarity-Aware KV Cache Merging with Bias Calibration*.  

The continuously growing key-value (KV) Cache during the inference of large language models (LLMs) becomes an obstacle to efficient deployment. Recently, KV Cache compression techniques such as evicting and merging KV pairs have been widely studied, which play an important role in reducing memory
usage in the decoding phase of LLM inference. However, these methods inevitably cause problems of output perturbation and
computational redundancy. In this paper, we propose SimCalKV, a theoretically grounded merging strategy that identifies the optimal merging under key similarity assumptions. By averaging
similar keys with a simple bias calibration and weighting values according to attention scores, SimCalKV reduces perturbation with negligible computational overhead.

⚡ **Work in progress – under active development.**  
Updates and improvements will be added continuously.
