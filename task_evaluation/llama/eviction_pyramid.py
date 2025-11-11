import torch
import torch.nn.functional as F
import math

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

def divide_by_layer(layer_idx: int, head_idx: int, key_value_cache, attention, ratio: float):
    K = key_value_cache[layer_idx][0][0, head_idx]  # (s, d)
    V = key_value_cache[layer_idx][1][0, head_idx]  # (s, d)
    s, _ = K.shape

    num_Ke = max(1, int(s * ratio))
    attn_mat = attention[layer_idx][0, head_idx]  # (s, s)
    token_scores = attn_mat.sum(dim=0)            # (s,)
    _, Ke_idx = torch.topk(-token_scores, num_Ke) 
    mask = torch.ones(s, dtype=torch.bool, device=K.device)
    mask[Ke_idx] = False
    Kc_idx = mask.nonzero(as_tuple=True)[0]

    Ke_keys, Kc_keys = K[Ke_idx], K[Kc_idx]
    Ve, Vc = V[Ke_idx], V[Kc_idx]

    if Ke_keys.numel() == 0 or Kc_keys.numel() == 0:
        sim_matrix = torch.empty((Ke_keys.size(0), Kc_keys.size(0)), device=K.device)
    else:
        sim_matrix = F.normalize(Ke_keys, dim=1) @ F.normalize(Kc_keys, dim=1).T

    return Ke_keys, Kc_keys, Ve, Vc, sim_matrix

def compress_KV(model, layer_idx, head_idx, key_value_cache, attention, hidden_states, ratio, sim_thr=0):
    Ke, Kc, Ve, Vc, sim_matrix = divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio)
    if sim_matrix.numel() == 0:
        return Kc, Vc

    max_sim, row_idx = torch.max(sim_matrix, dim=1)  # 
    mask = max_sim > sim_thr
    if not mask.any():
        return Kc, Vc

    return Kc, Vc

def Decode(model, key_value_cache, attention, hidden_states, ratio_start: float, sim_thr=0):
    num_layers = len(model.model.layers)
    num_heads = key_value_cache[0][0].shape[1]

    def process_layer(layer_idx, ratio_layer):
        q_t_all = get_qt(layer_idx, 0, model, hidden_states).unsqueeze(0)  # (1, d)
        Ks_list = []
        Vs_list = []
        for h in range(num_heads):
            K, V = compress_KV(model, layer_idx, h, key_value_cache, attention, hidden_states, ratio_layer, sim_thr)
            Ks_list.append(K)
            Vs_list.append(V)
        Ks = torch.stack(Ks_list, dim=0).unsqueeze(0)
        Vs = torch.stack(Vs_list, dim=0).unsqueeze(0)
        return Ks, Vs

    past_key_values_merged = []
    for i in range(num_layers):
        ratio_layer = ratio_start + 0.01 * i
        Ks, Vs = process_layer(i, ratio_layer)
        past_key_values_merged.append((Ks, Vs))

    return past_key_values_merged






