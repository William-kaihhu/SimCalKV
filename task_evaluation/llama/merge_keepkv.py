import torch
import torch.nn.functional as F
import math

eps = 1e-8  

def get_qt(layer_idx, head_idx, my_model, hidden_states):
    attn_module = my_model.model.layers[layer_idx].self_attn
    Q = attn_module.q_proj(hidden_states[layer_idx])
    b, s, _ = Q.shape
    num_heads = attn_module.num_heads
    d_h = Q.shape[-1] // num_heads
    Q = Q.view(b, s, num_heads, d_h).transpose(1, 2)
    q_t = Q[0, head_idx, -1].unsqueeze(0)
    return q_t

def get_w(q, k):
    dim = q.shape[1]
    w = torch.exp((q @ k.transpose(0, -1)) / math.sqrt(dim))
    w = torch.clamp(w, min=eps)
    return w.item()

def merge_by_token(key1, key2, w1, w2):
    w1 = max(w1, eps)
    w2 = max(w2, eps)
    numerator = w1 * key1 + w2 * key2
    denominator = 0
    denominator = w1 * math.log(w1) + w2 * math.log(w2)
    denominator = max(denominator, eps)
    key_merge = math.log((w1 + w2) / 2 + eps) * numerator / denominator
    return numerator / (w1 + w2)

def divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio):
    K_matrix = key_value_cache[layer_idx][0][0]  # shape: [num_heads, seq_len, head_dim]
    V_matrix = key_value_cache[layer_idx][1][0]

    K_of_head = K_matrix[head_idx]  # [seq_len, head_dim]
    V_of_head = V_matrix[head_idx]

    s, head_dim = K_of_head.shape
    num_Ke = max(1, int(s * ratio))
    num_Kc = s - num_Ke

    attn_matrix = attention[layer_idx][0, head_idx]  # [seq_len, seq_len]
    token_scores = attn_matrix.sum(dim=0)
    _, Ke_idx = torch.topk(-token_scores, num_Ke)
    Ke_idx = Ke_idx.tolist()
    Kc_idx = [i for i in range(s) if i not in Ke_idx]

    Ke_tokens_keys = K_of_head[Ke_idx]
    Kc_tokens_keys = K_of_head[Kc_idx]
    Ve_tokens_values = V_of_head[Ke_idx]
    Vc_tokens_values = V_of_head[Kc_idx]

    Ke_norm = F.normalize(Ke_tokens_keys, p=2, dim=1)
    Kc_norm = F.normalize(Kc_tokens_keys, p=2, dim=1)
    similarity_matrix = Ke_norm @ Kc_norm.T

    return Ke_tokens_keys, Kc_tokens_keys, Ve_tokens_values, Vc_tokens_values, similarity_matrix

def merge_Ke_to_Kc(my_model, layer_idx, head_idx, key_value_cache, attention, hidden_states, ratio):
    Ke, Kc, _, _, sim_matrix = divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio)
    row_max, row_idx = torch.max(sim_matrix, dim=1)
    row_idx = row_idx.tolist()
    Ke_num = Ke.size(0)
    q_t = get_qt(layer_idx, head_idx, my_model, hidden_states)
    for i in range(Ke_num):
        to_be_merged_idx = row_idx[i]
        w_1 = get_w(q_t, Ke[i])
        w_2 = get_w(q_t, Kc[to_be_merged_idx])
        
    for i in range(Ke_num):
        to_be_merged_idx = row_idx[i]
        w_e = get_w(q_t, Ke[i])
        w_c = get_w(q_t, Kc[to_be_merged_idx])
        Kc[to_be_merged_idx] = merge_by_token(Ke[i], Kc[to_be_merged_idx], w_e, w_c)
    return sim_matrix, Kc

def merge_Ve_to_Vc(my_model, layer_idx, head_idx, key_value_cache, attention, hidden_states, ratio):
    Ke, Kc, Ve, Vc, sim_matrix = divide_by_layer(layer_idx, head_idx, key_value_cache, attention, ratio)
    row_max, row_idx = torch.max(sim_matrix, dim=1)
    row_idx = row_idx.tolist()
    Ve_num = Ve.size(0)
    q_t = get_qt(layer_idx, head_idx, my_model, hidden_states)

    for i in range(Ve_num):
        to_be_merged_idx = row_idx[i]
        w_e = get_w(q_t, Ke[i])
        w_c = get_w(q_t, Kc[to_be_merged_idx])
        Vc[to_be_merged_idx] = (w_e * Ve[i] + w_c * Vc[to_be_merged_idx]) / (w_e + w_c + eps)
    return sim_matrix, Vc

def KeepKV(model, key_value_cache, attention, hidden_states, ratio):
    nums_layer = len(model.model.layers)
    past_key_values_merged = []

    for i in range(nums_layer):
        num_kv_heads = key_value_cache[i][0].shape[1]
        Kc_layer_list = []
        Vc_layer_list = []

        for j in range(num_kv_heads):
            _, Kc_head = merge_Ke_to_Kc(model, i, j, key_value_cache, attention, hidden_states, ratio)
            Kc_layer_list.append(Kc_head)

            _, Vc_head = merge_Ve_to_Vc(model, i, j, key_value_cache, attention, hidden_states, ratio)
            Vc_layer_list.append(Vc_head)

        Kc_layer = torch.stack(Kc_layer_list, dim=0).unsqueeze(0)
        Vc_layer = torch.stack(Vc_layer_list, dim=0).unsqueeze(0)

        past_key_values_merged.append((Kc_layer, Vc_layer))
        # ratio = ratio + 0.01  # pyramid style

    return past_key_values_merged
