import torch
import torch.nn.functional as F
import math
from typing import Callable, Optional, Tuple, List
def get_qt(layer_idx, head_idx, my_model, hidden_states):
    attn_module = my_model.model.layers[layer_idx].self_attn  # works for both LLaMA and Mistral
    Q = attn_module.q_proj(hidden_states[layer_idx])  # (b, s, num_heads * head_dim)
    b, s, _ = Q.shape
    num_heads = attn_module.num_heads
    d_h = Q.shape[-1] // num_heads
    Q = Q.view(b, s, num_heads, d_h).transpose(1, 2)  # (b, num_heads, s, d_h), now we have Q
    q_t = Q[0, head_idx, -1].unsqueeze(0)  # (0, head_idx, -1) → take the first sample, the head_idx-th head, and the last token
    return q_t

def get_w(q, k):
    """
    q: (1, dim)
    k: (1, dim)
    """
    dim = q.shape[1]
    w = torch.exp((q @ k.transpose(0, -1)) / math.sqrt(dim))  # (1, 1)
    return w.item()

def divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio):
    K_matrix = key_value_cache[layer_idx][0]  # Key matrix of layer layer_idx, shape: (b, num_heads, s, head_dim)
    V_matrix = key_value_cache[layer_idx][1]

    K_of_head = K_matrix[:, head_idx, :, :]  # Matrix of head head_idx, shape: (b, s, head_dim)
    # Assume b=1 (this is basically the case in practice)
    K_of_head = K_of_head[0]  # shape (s, head_dim)

    V_of_head = V_matrix[:, head_idx, :, :]
    V_of_head = V_of_head[0]

    s, head_dim = K_of_head.shape

    num_Ke = int(s * ratio)
    num_Kc = s - num_Ke

    """with torch.no_grad():
        outputs = my_model(**my_inputs, attn_implementation="eager", output_attentions=True)
        attention = outputs.attentions  # a tuple with length num_layer"""

    # outputs.attentions[layer_idx].shape = (b, num_heads, s, s)
    attn_matrix = attention[layer_idx][0, head_idx]  # shape (s, s)

    # Sum over each column, each token gets the total attention from other tokens, result is a tensor of length s
    token_scores = attn_matrix.sum(dim=0)

    _, Ke_idx = torch.topk(-token_scores, num_Ke)  # indices of tokens with the lowest attention scores
    Ke_idx = Ke_idx.tolist()
    Kc_idx = [i for i in range(s) if i not in Ke_idx]

    Ke_tokens_keys = K_of_head[Ke_idx]  # shape (num_Ke, head_dim), keys of tokens indexed by Ke_idx
    Kc_tokens_keys = K_of_head[Kc_idx]  # shape (num_Kc, head_dim)

    Ve_tokens_values = V_of_head[Ke_idx]
    Vc_tokens_values = V_of_head[Kc_idx]  # one-to-one correspondence with keys

    Ke_norm = F.normalize(Ke_tokens_keys, p=2, dim=1)  # [num_Ke, head_dim]
    Kc_norm = F.normalize(Kc_tokens_keys, p=2, dim=1)  # [num_Kc, head_dim]

    # [num_Ke, num_Kc], similarity_matrix[i, j] → cosine similarity between i-th Ke token and j-th Kc token
    similarity_matrix = Ke_norm @ Kc_norm.T

    return Ke_tokens_keys, Kc_tokens_keys, Ve_tokens_values, Vc_tokens_values, similarity_matrix


def Decode(model, key_value_cache, attention, hidden_states, ratio):
    nums_layer = len(model.model.layers)
    past_key_values_merged = []

    for i in range(nums_layer):
        num_kv_heads = key_value_cache[i][0].shape[1]

        # ---- build Kc ----
        Kc_layer_list = []
        for j in range(num_kv_heads):
            _, Kc_head, _, _, _ = divide_by_layer(i, j, key_value_cache, attention, ratio)
            Kc_layer_list.append(Kc_head)
        Kc_layer = torch.stack(Kc_layer_list, dim=0).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]

        # ---- build Vc ----
        Vc_layer_list = []
        for k in range(num_kv_heads):
            _, _, _, Vc_head, _ = divide_by_layer(i, k, key_value_cache, attention, ratio)
            Vc_layer_list.append(Vc_head)
        Vc_layer = torch.stack(Vc_layer_list, dim=0).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]

        # ---- put into tuple and append to list ----
        past_key_values_merged.append((Kc_layer, Vc_layer))

        if i % 2 == 0:
            ratio = ratio + 0.01    # ------ PyramidInfer
    return past_key_values_merged
def Repair_qwen(model,
                key_value_cache,
                attention,
                hidden_states: List[torch.Tensor],
                ratio: float = 0.2,
                rope_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
    """
    model: the HF model instance (Qwen)
    key_value_cache: outputs.past_key_values (list of tuples (K, V) per layer)
    attention: list of attention matrices per layer (each of shape (batch, num_heads, seq, seq))
               --> if you have model outputs with attn, you can pass outputs.attentions
    hidden_states: list/tuple of hidden states per layer (as returned when output_hidden_states=True)
    ratio: fraction of tokens to treat as Ke (low-attention tokens)
    rope_fn: optional function applied to q vectors to match key space if needed
    Returns:
      past_key_values_merged: list of (Kc_layer, Vc_layer) for each layer with shapes matching HF expectation
    """
    nums_layer = len(model.model.layers)
    past_key_values_merged = []

    for i in range(nums_layer):
        num_kv_heads = key_value_cache[i][0].shape[1]

        # ---- build Kc ----
        Kc_layer_list = []
        for j in range(num_kv_heads):
            _, Kc_head, _, _, _ = divide_by_layer(i, j, key_value_cache, attention, ratio)
            Kc_layer_list.append(Kc_head)
        Kc_layer = torch.stack(Kc_layer_list, dim=0).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]

        # ---- build Vc ----
        Vc_layer_list = []
        for k in range(num_kv_heads):
            _, _, _, Vc_head, _ = divide_by_layer(i, k, key_value_cache, attention, ratio)
            Vc_layer_list.append(Vc_head)
        Vc_layer = torch.stack(Vc_layer_list, dim=0).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]

        # ---- put into tuple and append to list ----
        past_key_values_merged.append((Kc_layer, Vc_layer))

        if i % 2 == 0:
            ratio = ratio + 0.01    # ------ PyramidInfer
    return past_key_values_merged
