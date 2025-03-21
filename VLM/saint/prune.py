import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

def iterative_drop_full_graph(x, keys, keep_num=64, r=16):
    cls_token = x[:, :1, :] 
    tokens = x[:, 1:, :][0]
    keys = keys[:, 1:, :][0]

    norm_keys = F.normalize(keys, p=2, dim=-1)
    sim = norm_keys @ norm_keys.transpose(0, 1)

    cand_idx = torch.arange(tokens.shape[0], device=x.device)

    while cand_idx.shape[0] > (keep_num - 1):
        N = cand_idx.shape[0]
        cand_sim = sim[cand_idx][:, cand_idx]
        scores = cand_sim.sum(dim=-1)
        r = min(N - (keep_num - 1), N // r)
        drop_order = scores.argsort(descending=True)
        drop_local_indices = drop_order[:r]
        mask = torch.ones(N, dtype=torch.bool, device=x.device)
        mask[drop_local_indices] = False
        cand_idx = cand_idx[mask]

    kept_tokens = tokens[cand_idx, :].unsqueeze(0)
    x_new = torch.cat([cls_token, kept_tokens], dim=1)
    return x_new