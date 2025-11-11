import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import nn

@dataclass
class CaMPress(object): 
    compression_ratio: float = 0.2
    eps: float = 1e-8

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # --- 0. Get attention weights ---
        if attentions is None:
            raise ValueError(
                "CaMPress (Standalone) requires the 'attentions' tensor "
                "to be provided, as kvpress dependencies were removed."
            )

        if self.compression_ratio <= 0 or self.compression_ratio >= 1:
            return keys, values

        b, h_kv, s, d = keys.shape
        h_q = attentions.size(1)
        device, dtype = keys.device, keys.dtype

        if s < 2:  # At least 2 tokens are needed for merging
            return keys, values

        # --- 1. Compute cumulative attention (A_bar) ---
        # A_bar: (b, h_q, s)
        # Represents the total attention each K/V token (s) receives from all queries (q)
        A_bar_q = attentions.sum(dim=2) 
        # Aggregate Q-heads into KV-heads
        num_kv_groups = h_q // h_kv
        # A_bar: (b, h_kv, s)
        A_bar = A_bar_q.view(b, h_kv, num_kv_groups, s).mean(dim=2)

        # --- 2. Identify Ke (to be evicted) and Kc (to be kept) ---
        # (same selection logic as SimCalKVPressRepair)
        num_Ke = max(1, int(s * self.compression_ratio))
        num_Kc = s - num_Ke
        m = num_Kc  # In CaM, 'm' is the size of the kept set

        # Ke_idx: (b, h_kv, num_Ke) - indices of tokens with the lowest scores
        _, Ke_idx = torch.topk(-A_bar, num_Ke, dim=-1) 
        # Kc_mask: (b, h_kv, s) - True means keep (Kc), False means evict (Ke)
        Kc_mask = torch.ones(b, h_kv, s, dtype=torch.bool, device=device)
        Kc_mask.scatter_(2, Ke_idx, False)

        # --- 3. Extract Key cache (K) ---
        # CaM only merges Values. Keys are extracted by masking.
        # Kc_new: (b, h_kv, num_Kc, d)
        Kc_new = keys[Kc_mask].view(b, h_kv, num_Kc, d)

        # --- 4. Extract Values and scores needed for CaM ---
        # 4.1. Extract evicted values Ve (V_i) and their scores
        # Ve: (b, h_kv, num_Ke, d)
        Ve = torch.gather(values, 2, Ke_idx.unsqueeze(-1).expand(b, h_kv, num_Ke, d))
        # A_bar_e: (b, h_kv, num_Ke)
        A_bar_e = torch.gather(A_bar, 2, Ke_idx)

        # 4.2. Extract kept values Vc (V_j..j+m) and their scores
        # Vc_new: (b, h_kv, num_Kc, d)
        Vc_new = values[Kc_mask].view(b, h_kv, num_Kc, d)
        # A_bar_c: (b, h_kv, num_Kc)
        A_bar_c = A_bar[Kc_mask].view(b, h_kv, num_Kc)

        # --- 5. Compute CaM merge probability p (Eq. 14) ---
        # avg_A_bar_c: (b, h_kv, 1)
        avg_A_bar_c = torch.mean(A_bar_c, dim=-1, keepdim=True)
        # p = \bar{A}_i / avg(\bar{A}_{j:j+m})
        # p: (b, h_kv, num_Ke)
        p = A_bar_e / avg_A_bar_c.clamp(min=self.eps)
        p = p.clamp(0.0, 1.0)  # Apply clamping

        # --- 6. Sample merge mask M (Eq. 14) ---
        # M: (b, h_kv, num_Ke)
        M = torch.bernoulli(p)
        # Adjust M's shape for broadcasting
        # M: (b, h_kv, num_Ke, 1)
        M = M.unsqueeze(-1)

        # --- 7. Compute merge increment (Eq. 15) ---
        # V_to_merge_i = M_i * (V_i / m)
        # (b, h_kv, num_Ke, d)
        V_to_merge_i = M * (Ve / m)
        # Sum all contributions from the V_i to be merged
        # V_total_delta: (b, h_kv, 1, d)
        V_total_delta = torch.sum(V_to_merge_i, dim=2, keepdim=True)

        # --- 8. In-place merge update ---
        # \bar{V}_k = V_k + \sum_{i} (M_i * V_i / m)
        # (b, h_kv, num_Kc, d) + (b, h_kv, 1, d) -> broadcast
        Vc_new = Vc_new + V_total_delta

        # --- 9. Return compressed K/V ---
        return Kc_new, Vc_new
