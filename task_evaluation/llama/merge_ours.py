import torch
import torch.nn.functional as F
import math
from typing import List

eps = 1e-8

# ============================================================
# helpers
# ============================================================

def merge_by_token(key1: torch.Tensor, key2: torch.Tensor) -> torch.Tensor:
    """
    Vectorized token merge:
        (key1 + key2)/2 + ln(2)
    Supports any shape (N, d)
    """
    return 0.5 * (key1 + key2) + math.log(2)

def get_all_qt(layer_idx: int, model, hidden_states: List[torch.Tensor]) -> torch.Tensor:
    """
    Get the query of the last token for ALL heads at the specified layer.
    Compute the Q projection once to avoid repeated computation inside head loops.

    Returns: (num_heads, head_dim)
    """
    attn = model.model.layers[layer_idx].self_attn
    # (b, s, num_heads * head_dim)
    Q = attn.q_proj(hidden_states[layer_idx])
    b, s, _ = Q.shape
    num_heads = getattr(model.config, "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(attn, "num_key_value_groups", getattr(attn, "num_heads", None))
    head_dim = Q.shape[-1] // num_heads

    # (b, num_heads, s, head_dim)
    Q = Q.view(b, s, num_heads, head_dim).transpose(1, 2)
    # (b, num_heads, head_dim)
    q_t = Q[:, :, -1, :]
    
    # Assume batch_size b is always 1
    return q_t.squeeze(0)  # (num_heads, head_dim)

def get_qt(layer_idx: int, head_idx: int, model, hidden_states: List[torch.Tensor]) -> torch.Tensor:
    """
    Get the query of the last token for a specific layer and head.
    Vectorized implementation to avoid iterating over heads.
    """
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
    """
    Vectorized weight computation:
        w = exp(q @ k^T / sqrt(d))
    q: (1, d)
    k: (N, d)
    Returns: (N,)
    """
    dim = q.shape[-1]
    w = torch.exp((q @ k.T) / math.sqrt(dim))
    return torch.clamp(w, min=eps).squeeze(0)


# ============================================================
# divide layer
# ============================================================

def divide_by_layer(layer_idx: int, head_idx: int, key_value_cache, attention, ratio: float):
    """
    Vectorized implementation:
    - Partition Ke/Kc according to attention scores
    - Compute cosine similarity matrix
    """
    K = key_value_cache[layer_idx][0][0, head_idx]  # (s, d)
    V = key_value_cache[layer_idx][1][0, head_idx]  # (s, d)
    s, _ = K.shape

    num_Ke = max(1, int(s * ratio))
    attn_mat = attention[layer_idx][0, head_idx]  # (s, s)
    token_scores = attn_mat.sum(dim=0)            # (s,)
    _, Ke_idx = torch.topk(-token_scores, num_Ke) # smallest attention scores
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


def merge_KV_unified(
    q_t: torch.Tensor, 
    key_value_cache, 
    attention, 
    layer_idx: int, 
    head_idx: int, 
    ratio: float, 
    sim_thr: float
) -> (torch.Tensor, torch.Tensor): # type: ignore
    """
    Correct and efficient "many-to-one" KV merge implementation.
    
    Parameters:
        q_t (torch.Tensor): the q_t for this head (1, d), passed in to avoid recomputing it.
    """
    Ke, Kc, Ve, Vc, sim_matrix = divide_by_layer(
        layer_idx, head_idx, key_value_cache, attention, ratio
    )
    
    # ---- 1. Handle empty / no-match edge cases ----
    if sim_matrix.numel() == 0:
        return Kc, Vc
    
    # Find for each Ke the index of the best matching Kc
    # max_sim: (num_Ke,), row_idx: (num_Ke,)
    max_sim, row_idx = torch.max(sim_matrix, dim=1)  
    mask = max_sim > sim_thr
    
    if not mask.any():
        return Kc, Vc

    # Select Ke/Ve to be merged and their target Kc indices
    Ke_sel, Ve_sel = Ke[mask], Ve[mask]      # (num_sel, d)
    idx_sel = row_idx[mask]                 # (num_sel,) - index of target Kc

    num_Kc, dim = Kc.shape
    device, dtype = Kc.device, Kc.dtype
    
    # ---- 2. K merge (many-to-one) ----
    
    # Goal: compute Ke_grouped = mean(all Ke_sel mapped to Kc[k])
    # 2.a. accumulate sums
    sum_Ke = torch.zeros((num_Kc, dim), device=device, dtype=dtype)
    sum_Ke.scatter_add_(dim=0, index=idx_sel.unsqueeze(1).expand(-1, dim), src=Ke_sel)
    
    # 2.b. counts
    count_Ke = torch.zeros(num_Kc, device=device, dtype=dtype)
    count_Ke.scatter_add_(dim=0, index=idx_sel, src=torch.ones_like(idx_sel, dtype=dtype))

    # 2.c. compute averages Ke_grouped, and find which Kc are actually merged
    # (avoid division by 0)
    target_Kc_mask = count_Ke > 0
    Ke_grouped = torch.zeros_like(sum_Ke)
    Ke_grouped[target_Kc_mask] = sum_Ke[target_Kc_mask] / count_Ke[target_Kc_mask].unsqueeze(1)
    
    # 2.d. apply the merge formula (only for Ks that are merged)
    Kc_new = Kc.clone()
    Kc_new[target_Kc_mask] = merge_by_token(
        Ke_grouped[target_Kc_mask],  # averaged Ke
        Kc[target_Kc_mask]           # original Kc
    )

    # ---- 3. V merge (many-to-one) ----
    
    # Goal: V_new[k] = (sum(w_e * Ve) + w_c * Vc) / (sum(w_e) + w_c)
    
    # 3.a. compute all relevant attention weights
    w_e = get_w(q_t, Ke_sel)  # (num_sel,)
    w_c = get_w(q_t, Kc)      # (num_Kc,)

    # 3.b. compute numerator
    # (sum(w_e * Ve))
    w_e_x_Ve_sel = w_e.unsqueeze(1) * Ve_sel  # (num_sel, d)
    sum_w_e_x_Ve = torch.zeros((num_Kc, dim), device=device, dtype=dtype)
    sum_w_e_x_Ve.scatter_add_(dim=0, index=idx_sel.unsqueeze(1).expand(-1, dim), src=w_e_x_Ve_sel)
    
    # (w_c * Vc)
    w_c_x_Vc = w_c.unsqueeze(1) * Vc          # (num_Kc, d)
    
    Numerator = sum_w_e_x_Ve + w_c_x_Vc

    # 3.c. compute denominator
    # (sum(w_e))
    sum_w_e = torch.zeros(num_Kc, device=device, dtype=dtype)
    sum_w_e.scatter_add_(dim=0, index=idx_sel, src=w_e)
    
    Denominator = (sum_w_e + w_c).clamp_min(1e-6).unsqueeze(1)
    
    # 3.d. compute Vc_new
    # This formula automatically handles tokens that were not merged:
    # If Kc[k] was not merged, sum_w_e[k] = 0 and sum_w_e_x_Ve[k] = 0
    # Vc_new[k] = (0 + w_c[k]*Vc[k]) / (0 + w_c[k]) = Vc[k]
    Vc_new = Numerator / Denominator

    return Kc_new, Vc_new


# Repair main 
def Repair(model, key_value_cache, attention, hidden_states, ratio: float, sim_thr=0):
    """
    Vectorized processing across layers and heads.
    - Uses get_all_qt to fix the Q-projection efficiency issue.
    - Uses merge_KV_unified to fix the many-to-one merge bug.
    """
    num_layers = len(model.model.layers)
    num_heads = key_value_cache[0][0].shape[1]

    def process_layer(layer_idx):
        # 1. Efficiently compute all heads' q_t at once
        all_q_t = get_all_qt(layer_idx, model, hidden_states)  # (num_heads, d)
        
        # 2. Process all heads in parallel
        results = [
            merge_KV_unified(
                q_t=all_q_t[h].unsqueeze(0), # pass this head's q_t (1, d)
                key_value_cache=key_value_cache,
                attention=attention,
                layer_idx=layer_idx,
                head_idx=h,
                ratio=ratio,
                sim_thr=sim_thr
            )
            for h in range(num_heads)
        ]
        
        Ks, Vs = zip(*results)
        # Stack head dimension back
        return torch.stack(Ks, dim=0).unsqueeze(0), torch.stack(Vs, dim=0).unsqueeze(0)

    # Keep the layer-wise structure
    past_key_values_merged = [process_layer(i) for i in range(num_layers)]
    return past_key_values_merged
