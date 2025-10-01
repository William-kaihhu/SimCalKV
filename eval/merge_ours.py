import torch
import torch.nn.functional as F
import math

def get_qt(layer_idx, head_idx, my_model, hidden_states):
    attn_module = my_model.model.layers[layer_idx].self_attn  # Works for both LLaMA and Mistral
    Q = attn_module.q_proj(hidden_states[layer_idx])  # (b, s, num_heads * head_dim)
    b, s, _ = Q.shape
    num_heads = attn_module.num_heads
    d_h = Q.shape[-1] // num_heads
    Q = Q.view(b, s, num_heads, d_h).transpose(1, 2)  # (b, num_heads, s, d_h), get Q
    q_t = Q[0, head_idx, -1].unsqueeze(0)  # select the first sample, head_idx-th head, last token
    return q_t

def get_w(q, k):
    """
    Compute attention weight between q and k.
    
    Args:
        q: (1, dim)
        k: (1, dim)
    Returns:
        Scalar attention weight
    """
    dim = q.shape[1]
    w = torch.exp((q @ k.transpose(0, -1)) / math.sqrt(dim))  # (1, 1)
    return w.item()

def merge_by_token(key1: torch.Tensor, key2: torch.Tensor) -> torch.Tensor:
    """
    Merge two token keys using the formula:
        (key1 + key2)/2 + ln(2) + (key1 - key2)^2 / 8
    
    Args:
        key1: [batch, num_heads, 1, head_dim]
        key2: [batch, num_heads, 1, head_dim]
    Returns:
        key_merge: [batch, num_heads, 1, head_dim]
    """
    key_merge = (key1 + key2) / 2 + math.log(2)
    return key_merge

def divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio):
    K_matrix = key_value_cache[layer_idx][0]  # Key matrix of layer_idx, shape: (b, num_heads, s, head_dim)
    V_matrix = key_value_cache[layer_idx][1]

    K_of_head = K_matrix[:, head_idx, :, :]  # Keys for the head, shape: (b, s, head_dim)
    K_of_head = K_of_head[0]  # assuming batch size = 1

    V_of_head = V_matrix[:, head_idx, :, :]
    V_of_head = V_of_head[0]

    s, head_dim = K_of_head.shape
    num_Ke = int(s * ratio)
    num_Kc = s - num_Ke

    # Attention matrix of this head: (s, s)
    attn_matrix = attention[layer_idx][0, head_idx]

    # Sum over columns to get total attention received by each token
    token_scores = attn_matrix.sum(dim=0)  
    _, Ke_idx = torch.topk(-token_scores, num_Ke)  # tokens with lowest attention scores
    Ke_idx = Ke_idx.tolist()
    Kc_idx = [i for i in range(s) if i not in Ke_idx]

    Ke_tokens_keys = K_of_head[Ke_idx]  # keys of Ke tokens
    Kc_tokens_keys = K_of_head[Kc_idx]  # keys of Kc tokens
    Ve_tokens_values = V_of_head[Ke_idx]
    Vc_tokens_values = V_of_head[Kc_idx]  # values corresponding to keys

    # Normalize for cosine similarity
    Ke_norm = F.normalize(Ke_tokens_keys, p=2, dim=1)  # [num_Ke, head_dim]
    Kc_norm = F.normalize(Kc_tokens_keys, p=2, dim=1)  # [num_Kc, head_dim]

    # similarity_matrix[i, j] = cosine similarity between Ke[i] and Kc[j]
    similarity_matrix = Ke_norm @ Kc_norm.T  
    return Ke_tokens_keys, Kc_tokens_keys, Ve_tokens_values, Vc_tokens_values, similarity_matrix

def merge_Ke_to_Kc(layer_idx, head_idx, key_value_cache, attention, ratio):
    Ke, Kc, _, _, sim_matrix = divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio)  # Only need Ke, Kc and similarity matrix
    row_max, row_idx = torch.max(sim_matrix, dim=1)  # max similarity index for each Ke
    row_idx = row_idx.tolist() 

    Ke_num = Ke.size(0)
    for i in range(Ke_num):
        to_be_merged_idx = row_idx[i]
        Kc[to_be_merged_idx] = merge_by_token(Ke[i], Kc[to_be_merged_idx])
    return sim_matrix, Kc

def merge_Ve_to_Vc(my_model, layer_idx, head_idx, key_value_cache, attention, hidden_states, ratio):
    Ke, Kc, Ve, Vc, sim_matrix = divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio)  # Only need Ve, Vc, similarity matrix
    row_max, row_idx = torch.max(sim_matrix, dim=1)  # max similarity index for each Ke
    row_idx = row_idx.tolist() 
    Ve_num = Ve.size(0)

    q_t = get_qt(layer_idx, head_idx, my_model, hidden_states)

    for i in range(Ve_num):
        to_be_merged_idx = row_idx[i]
        w_e = get_w(q_t, Ke[i])
        w_c = get_w(q_t, Kc[to_be_merged_idx])
        Vc[to_be_merged_idx] = (w_e * Ve[i] + w_c * Vc[to_be_merged_idx]) / (w_e + w_c)
    return sim_matrix, Vc

def Repair(model, key_value_cache, attention, hidden_states, ratio):
    nums_layer = len(model.model.layers)
    past_key_values_merged = []

    for i in range(nums_layer):
        num_kv_heads = key_value_cache[i][0].shape[1]

        # ---- build Kc ----
        Kc_layer_list = []
        for j in range(num_kv_heads):
            _, Kc_head = merge_Ke_to_Kc(i, j, key_value_cache, attention, ratio)
            Kc_layer_list.append(Kc_head)
        Kc_layer = torch.stack(Kc_layer_list, dim=0).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]

        # ---- build Vc ----
        Vc_layer_list = []
        for k in range(num_kv_heads):
            _, Vc_head = merge_Ve_to_Vc(model, i, k, key_value_cache, attention, hidden_states, ratio)
            Vc_layer_list.append(Vc_head)
        Vc_layer = torch.stack(Vc_layer_list, dim=0).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]

        # ---- Append as tuple to the list ----
        past_key_values_merged.append((Kc_layer, Vc_layer))

      # ratio = ratio + 0.01  # optional pyramid increment
    return past_key_values_merged
