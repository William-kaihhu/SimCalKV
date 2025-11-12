
# Instruction

This file provides detailed parameter settings and corresponding explanations for several experiments mentioned in the paper.

---

## 1. Input and Output Tokens Length

Different datasets require different input and output lengths. Some suggested settings are as follows:

| Dataset | Input + Output |
|:----------:|:----------:|
| XSUM | 512 + 64 |
| CNN/DailyMail | 1024 + 64 |
| LongBench / GovReport | 4096 + 512 |

The output length can affect the quality of the generation results. For the same dataset, it is sufficient to use a consistent standard. You can adjust the values according to your hardware capacity.

Parameter settings are implemented in `llama_eval.py`.

- **Line 82**
```python
  inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
```

→ Controls the **input length**.

* **Line 99**
```python
  total_kv, _, generated_tokens, durations_cached_s, _, _ = generate_text(model, tokenizer, 512, {
      "input_ids": next_token_id.reshape((1, 1)),
      "attention_mask": torch.cat(
          [inputs["attention_mask"], torch.tensor([[1]], device=device)], dim=1
      ),
      "past_key_values": past_key_values,
  })
```

  and **Lines 143, 155**

```python
  total_kv_merged_1, attention_mask1, generated_tokens_merged_1, durations_cached_s_merged_1, attention_merged, hidden_states_merged = \
      generate_text(model, tokenizer, 256, next_inputs)
  total_kv_merged, _, generated_tokens_merged, durations_cached_s_merged, attention_merged, hidden_states_merged = \
      generate_text(model, tokenizer, 256, next_inputs)
```

  → Control the **output length** before and after compression.

Our compression strategy is to **apply compression once after the prefill stage**.
Compression during the decoding stage is optional. To enable it, simply uncomment the following line:

```python
# kv_decode_merged = apply_kv_method(method, model, total_kv_merged_1, attention_merged, hidden_states_merged, compress_ratio)
```


## 2. Experiment on Similarity Thresholds (Table 4 in the Paper)

In `llama_eval.py`, line 68:

```python
return Repair(model, past_key_values, attention, hidden_states, compress_ratio, 0)
```

The **last parameter** is the similarity threshold.

* `0` means all pairs are merged.
* `0.8` means only key-value pairs with similarity greater than 0.8 are merged.


## 3. Ablation Study on Bias Vector (Table 5 in the Paper)

In `analysis/query_test.py`, two datasets are provided to compute the elements of the bias vector `delta_k`, which is used for comparison with `ln2`.

To modify the bias vector for the ablation experiment, edit **line 16** in `merge_ours.py`:

```python
return 0.5 * (key1 + key2) + math.log(2)
```


## 4. Experiment of Figure 8 

**Figure 8** in the paper shows model performance under finer-grained compression ratios.
You can reproduce this experiment using the following command:
```bash
bash run_cnn.sh
```
This will generate the corresponding experimental results.

