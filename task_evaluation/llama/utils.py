import time
import torch

device = torch.device("mps")  # cuda
def generate_token_with_past(model, inputs):
    # generate one token and return token_id and past_key_values
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        next_token_id = logits[0, -1, :].argmax()
        past_key_values = outputs.past_key_values
    return next_token_id, past_key_values

def generate_token_for_the_next_merge(model, inputs): # prepare for the next merge
    with torch.no_grad():   
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        logits = outputs.logits
        next_token_id = logits[0, -1, :].argmax()
        past_key_values = outputs.past_key_values
        attention = outputs.attentions
        hidden_states = outputs.hidden_states
    return next_token_id, past_key_values, attention, hidden_states

def generate_text(model, tokenizer, max_tokens, inputs, device=torch.device("mps")): # cuda
    # generate the context and return token_ids and time
    generated_tokens_ids = []
    durations_cached_s = []
    next_inputs = inputs
    attention = None
    hidden_states = None

    for i in range(max_tokens):
        t0 = time.time()
        if i < max_tokens - 1:
            next_token_id, past_key_values = generate_token_with_past(model, next_inputs)
        else:
            next_token_id, past_key_values, attention, hidden_states = \
            generate_token_for_the_next_merge(model, next_inputs)
        durations_cached_s.append(time.time() - t0)
        next_inputs = {
            "input_ids": next_token_id.reshape((1, 1)),
            "attention_mask": torch.cat(
                [next_inputs["attention_mask"], torch.tensor([[1]], device=device)],
                dim=1
            ),
            "past_key_values": past_key_values,
        }
        generated_tokens_ids.append(next_token_id)
    total_kv = next_inputs["past_key_values"]
    attention_mask = next_inputs["attention_mask"]
    return total_kv, attention_mask, generated_tokens_ids, durations_cached_s, attention, hidden_states
