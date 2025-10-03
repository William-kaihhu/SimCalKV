import torch
import torch.nn.functional as F
import math
from typing import Callable, Optional, Tuple, List
eps = 1e-8
# ---------------------- util helpers ----------------------

def merge_by_token(key1, key2, w1, w2):
    w1 = max(w1, eps)
    w2 = max(w2, eps)
    numerator = w1 * key1 + w2 * key2
    denominator = w1 * math.log(w1) + w2 * math.log(w2)
    denominator = max(denominator, eps)
    key_merge = math.log((w1 + w2) / 2 + eps) * numerator / denominator
    return numerator / (w1 + w2)

def stable_pair_weight(q: torch.Tensor, k_e: torch.Tensor, k_c: torch.Tensor) -> Tuple[float, float]:
    """
    Compute weights for Ve and Vc in a numerically stable way.
    We compute two logits: l_e = (q @ k_e) / sqrt(d), l_c = (q @ k_c) / sqrt(d)
    Then apply softmax over [l_e, l_c] and return weights (w_e, w_c) as floats.
    q: (1, d) or (d,), k_e/k_c: (d,) or (1,d)
    """
    if q.dim() == 2: qv = q.squeeze(0)
    else: qv = q
    d = qv.shape[-1]
    l_e = torch.dot(qv, k_e.view(-1)) / math.sqrt(d)
    l_c = torch.dot(qv, k_c.view(-1)) / math.sqrt(d)
    logits = torch.stack([l_e, l_c], dim=0)  # (2,)
    # clamp logits for numerical safety
    logits = torch.clamp(logits, min=-50.0, max=50.0)
    probs = F.softmax(logits, dim=0)  # (2,)
    return float(probs[0].item()), float(probs[1].item())

# ---------------------- q extraction ----------------------
def get_q_proj_from_attn(attn_module, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    For Qwen: try qkv_proj first; fallback to q_proj.
    hidden_states: assumed shape (batch, seq, dim)
    returns Q: (batch, seq, num_heads * head_dim)
    """
    # Qwen typically uses qkv_proj, but handle both
    if hasattr(attn_module, "qkv_proj"):
        qkv = attn_module.qkv_proj(hidden_states)  # (b, s, 3*hidden)
        # split the triple - q occupies first third
        total = qkv.size(-1)
        third = total // 3
        Q = qkv[..., :third]
        return Q
    elif hasattr(attn_module, "q_proj"):
        return attn_module.q_proj(hidden_states)
    else:
        raise RuntimeError("Unknown attention module: no qkv_proj or q_proj found")

def get_qt_qwen(layer_idx: int,
                head_idx: int,
                model,
                hidden_states: List[torch.Tensor],
                rope_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
    """
    Get last-token query vector for given layer & head.
    rope_fn: optional function to apply the same RoPE/pos embedding transform that keys in cache use.
    Returns: (1, head_dim) tensor
    """
    attn_module = model.model.layers[layer_idx].self_attn
    Q = get_q_proj_from_attn(attn_module, hidden_states[layer_idx])  # (b,s,num_heads*head_dim)
    b, s, _ = Q.shape
    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is None:
        # try common attribute name
        num_heads = getattr(attn_module, "n_head", None)
    if num_heads is None:
        raise RuntimeError("Cannot determine number of heads from attn module.")
    head_dim = Q.shape[-1] // num_heads
    Q = Q.view(b, s, num_heads, head_dim).transpose(1, 2)  # (b, num_heads, s, head_dim)
    q_t = Q[0, head_idx, -1].unsqueeze(0)  # (1, head_dim)
    if rope_fn is not None:
        # ensure rope_fn returns same shape
        q_t = rope_fn(q_t)
    return q_t

# ---------------------- divide and merge (Qwen style) ----------------------
def divide_by_layer_qwen(layer_idx: int,
                         head_idx: int,
                         key_value_cache,
                         attention,
                         ratio: float):
    """
    Assumes Q and V in cache are in shape (batch, num_heads, seq_len, head_dim),
    which matches the Qwen you showed (torch.Size([1,4,6,128])).
    Returns:
      Ke_keys: (num_Ke, head_dim)
      Kc_keys: (num_Kc, head_dim)
      Ve_vals: (num_Ke, head_dim)
      Vc_vals: (num_Kc, head_dim)
      sim_matrix: (num_Ke, num_Kc)
      Ke_idx: list indices
      Kc_idx: list indices
    """
    K_matrix = key_value_cache[layer_idx][0]  # (b, h, s, d)
    V_matrix = key_value_cache[layer_idx][1]  # (b, h, s, d)

    # assume batch=1
    K_of_head = K_matrix[0, head_idx, :, :]  # (s, d)
    V_of_head = V_matrix[0, head_idx, :, :]  # (s, d)

    s, head_dim = K_of_head.shape
    num_Ke = max(1, int(s * ratio))
    num_Kc = s - num_Ke

    # attention matrix for the head: (s, s)
    attn_mat = attention[layer_idx][0, head_idx]  # (s, s)
    # total attention received by each token (sum of columns)
    token_scores = attn_mat.sum(dim=0)
    # choose tokens with lowest attention (Ke)
    _, Ke_idx_t = torch.topk(-token_scores, num_Ke)
    Ke_idx = Ke_idx_t.tolist()
    Kc_idx = [i for i in range(s) if i not in Ke_idx]

    Ke_tokens_keys = K_of_head[Ke_idx]  # (num_Ke, head_dim)
    Kc_tokens_keys = K_of_head[Kc_idx]  # (num_Kc, head_dim)
    Ve_tokens_values = V_of_head[Ke_idx]  # (num_Ke, head_dim)
    Vc_tokens_values = V_of_head[Kc_idx]  # (num_Kc, head_dim)

    # normalize for cosine similarity
    if Ke_tokens_keys.size(0) == 0 or Kc_tokens_keys.size(0) == 0:
        sim_matrix = torch.empty((Ke_tokens_keys.size(0), Kc_tokens_keys.size(0)), device=K_of_head.device)
    else:
        Ke_norm = F.normalize(Ke_tokens_keys, p=2, dim=1)  # (num_Ke, head_dim)
        Kc_norm = F.normalize(Kc_tokens_keys, p=2, dim=1)  # (num_Kc, head_dim)
        sim_matrix = Ke_norm @ Kc_norm.T  # (num_Ke, num_Kc)

    return Ke_tokens_keys, Kc_tokens_keys, Ve_tokens_values, Vc_tokens_values, sim_matrix, Ke_idx, Kc_idx

def merge_Ke_to_Kc_qwen(model,
                        layer_idx: int,
                        head_idx: int,
                        key_value_cache,
                        attention,
                        hidden_states: List[torch.Tensor],
                        ratio: float,
                        rope_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
    Ke, Kc, _, _, sim_matrix, Ke_idx, Kc_idx = divide_by_layer_qwen(layer_idx, head_idx, key_value_cache, attention, ratio)
    if sim_matrix.numel() == 0:
        return sim_matrix, Kc, Ke_idx, Kc_idx
    _, row_idx = torch.max(sim_matrix, dim=1)  # for each Ke -> index in Kc
    row_idx = row_idx.tolist()
    Kc_new = Kc.clone()

    q_t = get_qt_qwen(layer_idx, head_idx, model, hidden_states, rope_fn=rope_fn)
    for i in range(Ke.size(0)):
        tgt = row_idx[i]
        # merge by token using our tensor-safe formula
        w_1, w_2 = stable_pair_weight(q_t, Ke[i], Kc[tgt])
    for i in range(Ke.size(0)):
        tgt = row_idx[i]
        # merge by token using our tensor-safe formula
        w_e, w_c = stable_pair_weight(q_t, Ke[i], Kc[tgt])
        Kc_new[tgt] = merge_by_token(Ke[i], Kc_new[tgt], w_e, w_c)
    return sim_matrix, Kc_new, Ke_idx, Kc_idx

def merge_Ve_to_Vc_qwen(model,
                        layer_idx: int,
                        head_idx: int,
                        key_value_cache,
                        attention,
                        hidden_states: List[torch.Tensor],
                        ratio: float,
                        rope_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
    Ke, Kc, Ve, Vc, sim_matrix, Ke_idx, Kc_idx = divide_by_layer_qwen(layer_idx, head_idx, key_value_cache, attention, ratio)
    if sim_matrix.numel() == 0:
        return sim_matrix, Vc, Ke_idx, Kc_idx

    _, row_idx = torch.max(sim_matrix, dim=1)
    row_idx = row_idx.tolist()
    Ve_num = Ve.size(0)

    # get q_t in same space; pass rope_fn if you need to apply RoPE to q
    q_t = get_qt_qwen(layer_idx, head_idx, model, hidden_states, rope_fn=rope_fn)  # (1, head_dim)

    Vc_new = Vc.clone()
    for i in range(Ve_num):
        tgt = row_idx[i]
        # stable weights
        w_e, w_c = stable_pair_weight(q_t, Ke[i], Kc[tgt])
        denom = (w_e + w_c) if (w_e + w_c) != 0.0 else 1e-6
        Vc_new[tgt] = (w_e * Ve[i] + w_c * Vc_new[tgt]) / denom
    return sim_matrix, Vc_new, Ke_idx, Kc_idx

# ---------------------- main Repair entry ----------------------
def Repair_keepkv(model,
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
        # number of heads from cache
        num_kv_heads = key_value_cache[i][0].shape[1]

        # build Kc for the layer
        Kc_layer_list = []
        for j in range(num_kv_heads):
            _, Kc_head, _, _ = merge_Ke_to_Kc_qwen(model, i, j, key_value_cache, attention, hidden_states, ratio, rope_fn=rope_fn)
            Kc_layer_list.append(Kc_head)
        # stack into (num_heads, seq_len, head_dim) then add batch dim -> (1, num_heads, seq_len, head_dim)
        Kc_layer = torch.stack(Kc_layer_list, dim=0).unsqueeze(0)

        # build Vc for the layer
        Vc_layer_list = []
        for k in range(num_kv_heads):
            _, Vc_head, _, _ = merge_Ve_to_Vc_qwen(model, i, k, key_value_cache, attention, hidden_states, ratio, rope_fn=rope_fn)
            Vc_layer_list.append(Vc_head)
        Vc_layer = torch.stack(Vc_layer_list, dim=0).unsqueeze(0)

        past_key_values_merged.append((Kc_layer, Vc_layer))

    return past_key_values_merged
