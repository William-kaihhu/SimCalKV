import torch
import torch.nn.functional as F
import math
from typing import List

eps = 1e-8  
# -------------------- helpers --------------------
def merge_by_token(key1: torch.Tensor, key2: torch.Tensor) -> torch.Tensor:
    """
    Merge two token keys using:
        (key1 + key2)/2 + ln(2)

    Args:
        key1, key2: (head_dim,) or (1, head_dim)

    Returns:
        merged key: same shape & device
    """
    ln2 = torch.log(torch.tensor(2., device=key1.device, dtype=key1.dtype))
    return 0.5 * (key1 + key2) + ln2


def get_qt(layer_idx: int, head_idx: int, model, hidden_states: List[torch.Tensor]) -> torch.Tensor:
    """
    Get the query vector of the last token for a specific layer and attention head.
    Returns shape: (1, head_dim)
    """
    attn = model.model.layers[layer_idx].self_attn  # works for both LLaMA and Mistral
    Q = attn.q_proj(hidden_states[layer_idx])       # (b, s, num_heads * head_dim)
    b, s, _ = Q.shape
    if hasattr(model.config, "num_attention_heads") and model.config.num_attention_heads is not None:
        num_heads = model.config.num_attention_heads
    elif hasattr(attn, "num_key_value_groups"):
        num_heads = attn.num_key_value_groups  # fallback
    elif hasattr(attn, "num_heads"):
        num_heads = attn.num_heads
    else:
        raise ValueError("Cannot determine number of attention heads!")
    head_dim = Q.shape[-1] // num_heads
    Q = Q.view(b, s, num_heads, head_dim).transpose(1, 2)  # (b, num_heads, s, head_dim)
    q_t = Q[0, head_idx, -1].unsqueeze(0)  # take the first sample, head_idx-th head, last token
    return q_t


def get_w(q, k):
    dim = q.shape[1]
    w = torch.exp((q @ k.transpose(0, -1)) / math.sqrt(dim))
    w = torch.clamp(w, min=eps)
    return w.item()


# -------------------- divide layer --------------------
def divide_by_layer(layer_idx: int, head_idx: int, key_value_cache, attention, ratio: float):
    """
    Divide the current layer's key/value into two parts based on attention scores:
        - Ke: low-attention tokens (to be merged)
        - Kc: high-attention tokens (to be preserved)
    Compute the cosine similarity matrix between Ke and Kc.

    Returns:
        Ke_tokens_keys, Kc_tokens_keys, Ve_tokens_values, Vc_tokens_values, similarity_matrix
    """
    K = key_value_cache[layer_idx][0][0, head_idx]  # Key matrix for this layer and head (s, head_dim)
    V = key_value_cache[layer_idx][1][0, head_idx]  # Corresponding Value matrix (s, head_dim)
    s, _ = K.shape

    num_Ke = max(1, int(s * ratio))  # number of tokens to be merged
    num_Kc = s - num_Ke               # number of tokens to be kept

    # attention[layer_idx][0, head_idx] has shape (s, s)
    attn_mat = attention[layer_idx][0, head_idx]
    token_scores = attn_mat.sum(dim=0)           # total attention each token receives
    _, Ke_idx = torch.topk(-token_scores, num_Ke)  # indices of tokens with lowest attention
    Ke_idx = Ke_idx.tolist()
    Kc_idx = [i for i in range(s) if i not in Ke_idx]
    
    # select Ke/Kc keys and values by indices
    Ke_keys = K[Ke_idx]
    Kc_keys = K[Kc_idx]
    Ve = V[Ke_idx]
    Vc = V[Kc_idx]

    # compute cosine similarity matrix between Ke and Kc: (num_Ke, num_Kc)
    if Ke_keys.size(0) == 0 or Kc_keys.size(0) == 0:
        sim_matrix = torch.empty((Ke_keys.size(0), Kc_keys.size(0)), device=K.device)
    else:
        sim_matrix = F.normalize(Ke_keys, dim=1) @ F.normalize(Kc_keys, dim=1).T

    return Ke_keys, Kc_keys, Ve, Vc, sim_matrix

def divide_without_cossim(layer_idx: int, head_idx: int, key_value_cache, attention, ratio: float):
    """
    Divide the current layer's key/value into two parts based on attention scores:
        - Ke: low-attention tokens (to be merged)
        - Kc: high-attention tokens (to be preserved)
    Returns:
        Ke_tokens_keys, Kc_tokens_keys, Ve_tokens_values, Vc_tokens_values
    """
    K = key_value_cache[layer_idx][0][0, head_idx]  # Key matrix for this layer and head (s, head_dim)
    V = key_value_cache[layer_idx][1][0, head_idx]  # Corresponding Value matrix (s, head_dim)
    s, _ = K.shape

    num_Ke = max(1, int(s * ratio))  # number of tokens to be merged
    num_Kc = s - num_Ke               # number of tokens to be kept

    # attention[layer_idx][0, head_idx] has shape (s, s)
    attn_mat = attention[layer_idx][0, head_idx]
    token_scores = attn_mat.sum(dim=0)           # total attention each token receives
    _, Ke_idx = torch.topk(-token_scores, num_Ke)  # indices of tokens with lowest attention
    Ke_idx = Ke_idx.tolist()
    Kc_idx = [i for i in range(s) if i not in Ke_idx]
    
    # select Ke/Kc keys and values by indices
    Ke_keys = K[Ke_idx]
    Kc_keys = K[Kc_idx]
    Ve = V[Ke_idx]
    Vc = V[Kc_idx]
    return Ke_keys, Kc_keys, Ve, Vc

def merge_KV_without_cossim(model, layer_idx, head_idx, key_value_cache, attention, hidden_states, ratio):
    Ke, Kc, Ve, Vc = divide_without_cossim(layer_idx, head_idx, key_value_cache, attention, ratio)
    Kc_new = Kc.clone()
    Vc_new = Vc.clone()
    q_t = get_qt(layer_idx, head_idx, model, hidden_states)
    j = 0
    for i in range(Ke.size(0)):
        tgt = j
        Kc_new[tgt] = merge_by_token(Ke[i], Kc_new[tgt])
        w_e = get_w(q_t, Ke[i])
        w_c = get_w(q_t, Kc[tgt])
        denom = w_e + w_c if (w_e + w_c) != 0 else 1e-6
        Vc_new[tgt] = (w_e * Ve[i] + w_c * Vc_new[tgt]) / denom
        j = j + 1
    return Kc_new, Vc_new

def merge_KV(model, layer_idx, head_idx, key_value_cache, attention, hidden_states, ratio):
    Ke, Kc, Ve, Vc, sim_matrix = divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio)
    if sim_matrix.numel() == 0:
        return Kc, Vc

    _, row_idx = torch.max(sim_matrix, dim=1)  # for each Ke, find the index of the most similar Kc
    Kc_new = Kc.clone()
    Vc_new = Vc.clone()
    q_t = get_qt(layer_idx, head_idx, model, hidden_states)
    for i in range(Ke.size(0)):
        # merge K
        tgt = row_idx[i].item()
        Kc_new[tgt] = merge_by_token(Ke[i], Kc_new[tgt])
        # merge V
        w_e = get_w(q_t, Ke[i])
        w_c = get_w(q_t, Kc[tgt])
        denom = w_e + w_c if (w_e + w_c) != 0 else 1e-6
        Vc_new[tgt] = (w_e * Ve[i] + w_c * Vc_new[tgt]) / denom
    return Kc_new, Vc_new

# -------------------- main Repair --------------------
def Repair(model, key_value_cache, attention, hidden_states, ratio: float):
    """
    Iterate over each layer and attention head to perform Ke/Kc merging.
    Output the new compressed KV cache.

    Returns:
        past_key_values_merged: List[(Kc_layer, Vc_layer)]
    """
    past_key_values_merged = []

    for layer_idx in range(len(model.model.layers)):
        num_heads = key_value_cache[layer_idx][0].shape[1]

        Kc_layer_list = []
        Vc_layer_list = []

        # perform key/value merging for each attention head
        for h in range(num_heads):
            #_, Kc_head = merge_Ke_to_Kc(layer_idx, h, key_value_cache, attention, ratio)
            #_, Vc_head = merge_Ve_to_Vc(model, layer_idx, h, key_value_cache, attention, hidden_states, ratio)
            Kc_head, Vc_head = merge_KV(model, layer_idx, h, key_value_cache, attention, hidden_states, ratio)
            Kc_layer_list.append(Kc_head)
            Vc_layer_list.append(Vc_head)

        # reshape to: [1, num_heads, seq_len, head_dim]
        Kc_layer = torch.stack(Kc_layer_list, dim=0).unsqueeze(0)
        Vc_layer = torch.stack(Vc_layer_list, dim=0).unsqueeze(0)
        past_key_values_merged.append((Kc_layer, Vc_layer))

    return past_key_values_merged
