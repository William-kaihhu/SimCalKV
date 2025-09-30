import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("mps")  # 使用 MPS
def generate_token_with_past(model, inputs):
    """生成一个 token，并返回 token_id 和 past_key_values"""
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        next_token_id = logits[0, -1, :].argmax()
        past_key_values = outputs.past_key_values
    return next_token_id, past_key_values

def generate_text(model, tokenizer, max_tokens, inputs):
    """逐步生成文本，返回生成的 tokens 和每步耗时"""
    generated_tokens = []
    durations_cached_s = []
    next_inputs = inputs
    for _ in range(max_tokens):
        t0 = time.time()
        next_token_id, past_key_values = generate_token_with_past(model, next_inputs)
        durations_cached_s.append(time.time() - t0)
        next_inputs = {
            "input_ids": next_token_id.reshape((1, 1)),
            "attention_mask": torch.cat(
                [next_inputs["attention_mask"], torch.tensor([[1]], device=device)],
                dim=1
            ),
            "past_key_values": past_key_values,
        }
        generated_tokens.append(tokenizer.decode(next_token_id, skip_special_tokens=True))
    return generated_tokens, durations_cached_s
