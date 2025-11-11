import torch
import torch.nn.functional as F
import math
from typing import List

eps = 1e-8  
# -------------------- helpers --------------------
def merge_by_token(key1: torch.Tensor, key2: torch.Tensor,
                   w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    if key1.numel() == 0:
        return key1.new_empty((0, key1.shape[-1]))
    w1 = w1.unsqueeze(1) if w1.dim() == 1 else w1
    w2 = w2.unsqueeze(1) if w2.dim() == 1 else w2
    numerator = w1 * key1 + w2 * key2
    denominator = w1 * torch.log(w1) + w2 * torch.log(w2)
    key_merge = torch.log((w1 + w2) / 2 + eps) * numerator / denominator
    denom = (w1 + w2).clamp_min(eps)
    return numerator / denom


def get_qt(layer_idx: int, head_idx: int, model, hidden_states: List[torch.Tensor]) -> torch.Tensor:
    attn = model.model.layers[layer_idx].self_attn
    Q = attn.q_proj(hidden_states[layer_idx])   # (b, s, num_heads * head_dim)
    b, s, _ = Q.shape
    num_heads = getattr(model.config, "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(attn, "num_key_value_groups", getattr(attn, "num_heads", None))
    head_dim = Q.shape[-1] // num_heads
    Q = Q.view(b, s, num_heads, head_dim).transpose(1, 2)  # (b, num_heads, s, head_dim)
    q_t = Q[:, :, -1, :]  # (b, num_heads, head_dim)
    return q_t[:, head_idx]  # (1, head_dim)


def get_w(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    dim = q.shape[-1]
    w = torch.exp((q @ k.T) / math.sqrt(dim))
    return torch.clamp(w, min=eps).squeeze(0)


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


def merge_KV(model, layer_idx, head_idx, key_value_cache, attention, hidden_states, ratio, sim_thr=0):
    Ke, Kc, Ve, Vc, sim_matrix = divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio)
    if sim_matrix.numel() == 0:
        return Kc, Vc

    max_sim, row_idx = torch.max(sim_matrix, dim=1)  # (num_Ke,)
    mask = max_sim > sim_thr
    if not mask.any():
        return Kc, Vc

    Ke_sel, Ve_sel, idx_sel = Ke[mask], Ve[mask], row_idx[mask]
    Kc_sel, Vc_sel = Kc[idx_sel], Vc[idx_sel]

    # ---- merge K ----
    q_t = get_qt(layer_idx, head_idx, model, hidden_states)  # (1, d)
    w_e = get_w(q_t, Ke_sel)
    w_c = get_w(q_t, Kc_sel)
    merged_K = merge_by_token(Ke_sel, Kc_sel, w_e, w_c)

    # ---- merge V ----
    denom = (w_e + w_c).clamp_min(1e-6).unsqueeze(1)
    merged_V = (w_e.unsqueeze(1) * Ve_sel + w_c.unsqueeze(1) * Vc_sel) / denom

    Kc_new, Vc_new = Kc.clone(), Vc.clone()
    Kc_new[idx_sel] = merged_K
    Vc_new[idx_sel] = merged_V

    return Kc_new, Vc_new

def merge_KV_2(model, layer_idx, head_idx, key_value_cache, attention, hidden_states, ratio, sim_thr=0):
    Ke, Kc, Ve, Vc, sim_matrix = divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio)
    if sim_matrix.numel() == 0:
        return Kc, Vc

    max_sim, row_idx = torch.max(sim_matrix, dim=1)  
    mask = max_sim > sim_thr
    if not mask.any():
        return Kc, Vc

    Ke_sel, Ve_sel, idx_sel = Ke[mask], Ve[mask], row_idx[mask]

    device = Kc.device
    dtype = Kc.dtype 
    Kc_new, Vc_new = Kc.clone(), Vc.clone()

    num_Ke = Ke_sel.shape[0]
    num_chunks = max(1, (Ke.shape[0] // Kc.shape[0])) 
    chunk_size = math.ceil(num_Ke / num_chunks)
    q_t = get_qt(layer_idx, head_idx, model, hidden_states)
    
    q_t = q_t.to(device=device, dtype=dtype)

    for i in range(num_chunks):
        start, end = i * chunk_size, min((i + 1) * chunk_size, num_Ke)
        Ke_chunk = Ke_sel[start:end].to(device=device, dtype=dtype)
        Ve_chunk = Ve_sel[start:end].to(device=device, dtype=dtype)
        idx_chunk = idx_sel[start:end].to(device=device)  

        num_Kc = Kc.shape[0]
       
        one_hot = F.one_hot(idx_chunk, num_classes=num_Kc).to(device=device, dtype=dtype)  # now same dtype as Ke_chunk

        denom = one_hot.sum(dim=0, keepdim=True).clamp_min(1.0).to(dtype)  # same dtype
        
        Ke_grouped = (one_hot.T @ Ke_chunk) / denom.T
        Ve_grouped = (one_hot.T @ Ve_chunk) / denom.T
        w_e = get_w(q_t, Ke_grouped).to(device=device, dtype=dtype)
        w_c = get_w(q_t, Kc.to(dtype)).to(device=device, dtype=dtype)
        # merged_K / merged_V 
        merged_K = merge_by_token(Ke_grouped, Kc.to(dtype), w_e, w_c)  
        merged_K = merged_K.to(device=device, dtype=dtype)

        
        denom_v = (w_e + w_c).clamp_min(1e-6).unsqueeze(1).to(dtype)

        merged_V = (w_e.unsqueeze(1) * Ve_grouped + w_c.unsqueeze(1) * Vc.to(dtype)) / denom_v

        Kc_new = merged_K
        Vc_new = merged_V

    return Kc_new, Vc_new


# -------------------- main Repair --------------------
def KeepKV(model, key_value_cache, attention, hidden_states, ratio: float, sim_thr=0):
    
    num_layers = len(model.model.layers)
    num_heads = key_value_cache[0][0].shape[1]

    def process_layer(layer_idx):
        q_t_all = get_qt(layer_idx, 0, model, hidden_states).unsqueeze(0) 
        
        if ratio <= 0.5:
            results = [
                merge_KV(model, layer_idx, h, key_value_cache, attention, hidden_states, ratio, sim_thr)
                for h in range(num_heads)
            ]
        else:
            results = [
                merge_KV_2(model, layer_idx, h, key_value_cache, attention, hidden_states, ratio, sim_thr)
                for h in range(num_heads)
            ]
        Ks, Vs = zip(*results)
        return torch.stack(Ks, dim=0).unsqueeze(0), torch.stack(Vs, dim=0).unsqueeze(0)

    past_key_values_merged = [process_layer(i) for i in range(num_layers)]
    return past_key_values_merged
