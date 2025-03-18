from typing import Callable, Optional
import torch

def saint_drop(
    metric: torch.Tensor,
    prune_mode: Optional[str],
    sim_threshold: float = 0.7,
    K: int = 5,
    gamma: float = 10,
    class_token: bool = False,
    distill_token: bool = False,
) -> Callable:
    
    if prune_mode is None:
        return lambda x: x

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    if protected > 0:
        metric = metric[:,protected:]
        

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        valid_mask = scores >= sim_threshold
        valid_counts = valid_mask.sum(dim=-1)
        row_mask = valid_counts >= K
        r = int(row_mask.float().sum(dim=-1).mean().item())

        if r <= 0:
            return lambda x: x
        
        score_candidate = (scores * valid_mask).sum(dim=-1) / valid_counts.clamp(min=1)
        score_candidate = valid_counts * torch.exp(gamma * (score_candidate - sim_threshold))
        score_alternative = scores.sum(dim=-1) / scores.size(-1)
        final_scores = torch.where(valid_counts > 0, score_candidate, score_alternative)
        
        sorted_indices = final_scores.argsort(dim=-1, descending=True).unsqueeze(-1)
        unm_idx = sorted_indices[..., r:, :]

    def drop_func(x: torch.Tensor) -> torch.Tensor:
        if protected > 0:
            special_tokens = x[:, :protected]
            x = x[:, protected:]
        
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        
        if protected > 0:
            return torch.cat([special_tokens, unm, dst], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)
            
    return drop_func
