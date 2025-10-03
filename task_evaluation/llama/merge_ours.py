import torch
import torch.nn.functional as F
import math
from typing import List

# -------------------- helpers --------------------
def merge_by_token(key1: torch.Tensor, key2: torch.Tensor) -> torch.Tensor:
    """
    Merge two token keys using formula:
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
    Get the last-token query vector for a given layer and head.
    Returns shape: (1, head_dim)
    """
    attn = model.model.layers[layer_idx].self_attn
    Q = attn.q_proj(hidden_states[layer_idx])  # (batch, seq, num_heads*head_dim)
    b, s, _ = Q.shape
    num_heads = getattr(attn, "num_heads", None) or getattr(attn, "n_head", None)
    head_dim = Q.shape[-1] // num_heads
    Q = Q.view(b, s, num_heads, head_dim).transpose(1, 2)  # (batch, num_heads, seq, head_dim)
    return Q[0, head_idx, -1].unsqueeze(0)


def get_w(q: torch.Tensor, k: torch.Tensor) -> float:
    """
    Compute attention weight between q and k using a softmax over two logits.
    Returns a scalar float.
    """
    d = q.shape[1]
    logits = torch.stack([q @ k.transpose(0, -1), torch.zeros(1, device=q.device)]) / math.sqrt(d)
    probs = F.softmax(logits, dim=0)
    return float(probs[0].item())


# -------------------- divide layer --------------------
def divide_by_layer(layer_idx: int, head_idx: int, key_value_cache, attention, ratio: float):
    """
    Split keys into Ke (low-attention) and Kc (to keep) and compute cosine similarity.
    Returns:
        Ke_keys, Kc_keys, Ve_vals, Vc_vals, similarity_matrix
    """
    K = key_value_cache[layer_idx][0][0, head_idx]  # (seq_len, head_dim)
    V = key_value_cache[layer_idx][1][0, head_idx]  # (seq_len, head_dim)
    s, _ = K.shape

    num_Ke = max(1, int(s * ratio))
    num_Kc = s - num_Ke

    attn_mat = attention[layer_idx][0, head_idx]  # (seq_len, seq_len)
    token_scores = attn_mat.sum(dim=0)  # sum of attention received by each token
    _, Ke_idx = torch.topk(-token_scores, num_Ke)  # low-attention tokens
    Ke_idx = Ke_idx.tolist()
    Kc_idx = [i for i in range(s) if i not in Ke_idx]

    Ke_keys = K[Ke_idx]
    Kc_keys = K[Kc_idx]
    Ve = V[Ke_idx]
    Vc = V[Kc_idx]

    # cosine similarity
    if Ke_keys.size(0) == 0 or Kc_keys.size(0) == 0:
        sim_matrix = torch.empty((Ke_keys.size(0), Kc_keys.size(0)), device=K.device)
    else:
        sim_matrix = F.normalize(Ke_keys, dim=1) @ F.normalize(Kc_keys, dim=1).T

    return Ke_keys, Kc_keys, Ve, Vc, sim_matrix


# -------------------- merge functions --------------------
def merge_Ke_to_Kc(layer_idx, head_idx, key_value_cache, attention, ratio):
    Ke, Kc, _, _, sim_matrix = divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio)
    if sim_matrix.numel() == 0:
        return sim_matrix, Kc

    _, row_idx = torch.max(sim_matrix, dim=1)
    Kc_new = Kc.clone()
    for i in range(Ke.size(0)):
        tgt = row_idx[i].item()
        Kc_new[tgt] = merge_by_token(Ke[i], Kc_new[tgt])
    return sim_matrix, Kc_new


def merge_Ve_to_Vc(model, layer_idx, head_idx, key_value_cache, attention, hidden_states, ratio):
    Ke, Kc, Ve, Vc, sim_matrix = divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio)
    if sim_matrix.numel() == 0:
        return sim_matrix, Vc

    _, row_idx = torch.max(sim_matrix, dim=1)
    q_t = get_qt(layer_idx, head_idx, model, hidden_states)

    Vc_new = Vc.clone()
    for i in range(Ve.size(0)):
        tgt = row_idx[i].item()
        w_e = get_w(q_t, Ke[i])
        w_c = get_w(q_t, Kc[tgt])
        denom = w_e + w_c if (w_e + w_c) != 0 else 1e-6
        Vc_new[tgt] = (w_e * Ve[i] + w_c * Vc_new[tgt]) / denom
    return sim_matrix, Vc_new


# -------------------- main Repair --------------------
def Repair(model, key_value_cache, attention, hidden_states, ratio: float):
    """
    Merge low-attention KV pairs into high-attention ones for all layers.
    Returns:
        List of tuples: [(Kc_layer, Vc_layer), ...] per layer
    """
    past_key_values_merged = []

    for layer_idx in range(len(model.model.layers)):
        num_heads = key_value_cache[layer_idx][0].shape[1]

        Kc_layer_list = []
        Vc_layer_list = []

        for h in range(num_heads):
            _, Kc_head = merge_Ke_to_Kc(layer_idx, h, key_value_cache, attention, ratio)
            _, Vc_head = merge_Ve_to_Vc(model, layer_idx, h, key_value_cache, attention, hidden_states, ratio)
            Kc_layer_list.append(Kc_head)
            Vc_layer_list.append(Vc_head)

        Kc_layer = torch.stack(Kc_layer_list, dim=0).unsqueeze(0)
        Vc_layer = torch.stack(Vc_layer_list, dim=0).unsqueeze(0)
        past_key_values_merged.append((Kc_layer, Vc_layer))

    return past_key_values_merged
