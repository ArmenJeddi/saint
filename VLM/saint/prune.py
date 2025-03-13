# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F


def do_nothing(x):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    prune_mode: Optional[str],
    sim_threshold: float = 0.9,
    class_token: bool = False,
    distill_token: bool = False,
) -> Callable:
    
    if prune_mode is None:
        return do_nothing
    
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
        
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        
        
        # scores = scores[:, 1:]
        # print(f"  Mean Key Sim: {scores.mean().item():.4f}")
        
        # r = 0
        
        percentage = (node_max > sim_threshold).float().mean().item()
        r = int(node_max.size(1) * percentage)

        if r <= 0:
            return do_nothing
        
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]


    def drop_func(x: torch.Tensor) -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)
        
    def deadpool_func(x: torch.Tensor) -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
        deadpool = src.mean(1, keepdim=True)
        
        mean_norm = src.norm(p=2, dim=-1).max(dim=-1, keepdim=True).values.unsqueeze(-1)
        deadpool = F.normalize(deadpool, p=2, dim=-1) * mean_norm

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:], deadpool], dim=1)
        else:
            return torch.cat([unm, dst, deadpool], dim=1)
        
    def merge_func(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)
        
    def mlerp_func(x: torch.Tensor) -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape

        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            
        dst_selected = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        mean = ( src + dst_selected ) / 2

        mean_norm = mean.norm(dim=-1, keepdim=True) + 1e-8  # 
        mean_normalized = mean / mean_norm
        src_norm = src.norm(dim=-1, keepdim=True)
        dst_selected_norm = dst_selected.norm(dim=-1, keepdim=True)
        max_norm = torch.max(src_norm, dst_selected_norm)
        scaled_mean = mean_normalized * max_norm

        dst.scatter_(-2, dst_idx.expand(n, r, c), scaled_mean)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)
        
    if prune_mode == 'drop':
        return drop_func
    elif prune_mode == 'merge':
        return merge_func
    elif prune_mode == 'mlerp':
        return mlerp_func
    elif prune_mode == 'deadpool':
        return deadpool_func


def random_drop_last(x, keep_num, class_token: bool = False):
    if class_token:
        cls = x[:, :1, :]
        x = x[:, 1:]
        keep_num = keep_num - 1
    
    B, N, C = x.shape
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)

    ids_keep = ids_shuffle[:, :keep_num]
    x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
    
    if class_token:
        x = torch.cat((cls, x), dim=1)
    
    return x


def batched_kmeans_itself(X, num_clusters, num_iters=5):
    B, N, C = X.shape
    centroids = torch.stack([X[b, torch.randperm(N, device=X.device)[:num_clusters]] for b in range(B)], dim=0)
    for _ in range(num_iters):
        diff = X.unsqueeze(2) - centroids.unsqueeze(1)
        dists = (diff ** 2).sum(-1)
        assignments = dists.argmin(-1)
        new_centroids = torch.zeros_like(centroids)
        for b in range(B):
            for j in range(num_clusters):
                mask = assignments[b] == j
                if mask.sum() > 0:
                    new_centroids[b, j] = X[b][mask].mean(0)
                else:
                    farthest_idx = torch.argmax(torch.min(dists[b], dim=1)[0])
                    new_centroids[b, j] = X[b, farthest_idx]
        centroids = new_centroids
    return assignments, centroids

def cluster_and_select_tokens_itself(x, k, num_iters=20, return_centroids=False):
    B, N, C = x.shape
    cls_token = x[:, :1]
    tokens = x[:, 1:]
    assignments, centroids = batched_kmeans_itself(tokens, k - 1, num_iters)
    if return_centroids:
        return torch.cat([cls_token, centroids], dim=1)
    reps = []
    for b in range(B):
        reps_b = []
        for j in range(k - 1):
            mask = assignments[b] == j
            if mask.sum() == 0:
                idx = torch.randint(0, tokens.shape[1], (1,), device=x.device).item()
                reps_b.append(tokens[b, idx:idx + 1])
            else:
                idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                d = ((tokens[b][mask] - centroids[b, j]) ** 2).sum(-1)
                best = d.argmin().item()
                chosen = idxs[best].item()
                reps_b.append(tokens[b, chosen:chosen + 1])
        reps.append(torch.cat(reps_b, dim=0))
    reps = torch.stack(reps)
    return torch.cat([cls_token, reps], dim=1)


def kmeans_cluster_with_cls(x, k, num_iters=3):
    B, N, C = x.shape
    cls_token = x[:, :1, :]
    tokens = x[:, 1:, :]#.float()
    M = tokens.shape[1]
    centroids = torch.stack([tokens[b, torch.randperm(M, device=tokens.device)[:k-1]] for b in range(B)], dim=0)
    for _ in range(num_iters):
        diff = tokens.unsqueeze(2) - centroids.unsqueeze(1)
        dists = (diff ** 2).sum(dim=-1)
        assignments = dists.argmin(dim=-1)
        one_hot = F.one_hot(assignments, num_classes=k-1).float()
        counts = one_hot.sum(dim=1)
        new_centroids = torch.bmm(one_hot.transpose(1, 2).float(), tokens.float()) / (counts.unsqueeze(-1) + 1e-8)
        mask = counts < 1
        if mask.sum().item() > 0:
            rand_idx = torch.randint(0, M, (B, k-1), device=tokens.device)
            batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(B, k-1)
            random_tokens = tokens[batch_idx, rand_idx]
            new_centroids = torch.where(mask.unsqueeze(-1), random_tokens, new_centroids)
        centroids = new_centroids
    return torch.cat([cls_token, centroids], dim=1).half()




def iterative_token_drop(x, metric, keep_num=64, class_token=True, r=16):
    
    while x.shape[1] > keep_num:
        with torch.no_grad():
            
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)
            # print(f"iterative_token_drop - {keep_num} - {r} - {metric.shape}")
            
            # a, b = metric[..., ::2, :], metric[..., 1::2, :]
            # scores = -torch.cdist(a, b, p=2)
            

            if class_token:
                scores[..., 0, :] = -math.inf

            node_max, _ = scores.max(dim=-1)
        
            r = min(r, x.shape[1] - keep_num)
            
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]

            if class_token:
                unm_idx = unm_idx.sort(dim=1)[0]

            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            x = torch.cat([unm, dst], dim=1)
            
            src_metric, dst_metric = metric[..., ::2, :], metric[..., 1::2, :]
            n, t1, c = src_metric.shape
            unm = src_metric.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            metric = torch.cat([unm, dst_metric], dim=1)

    return x


def iterative_token_merge(x, metric, keep_num=64, class_token=True, r=32):
    
    while x.shape[1] > keep_num:
        with torch.no_grad():
            
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)
            
            # a, b = metric[..., ::2, :], metric[..., 1::2, :]
            # scores = -torch.cdist(a, b, p=2)
            

            if class_token:
                scores[..., 0, :] = -math.inf

            node_max, node_idx = scores.max(dim=-1)
        
            r = min(r, x.shape[1] - keep_num)
            
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

            if class_token:
                unm_idx = unm_idx.sort(dim=1)[0]

            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='mean')
            
            x = torch.cat([unm, dst], dim=1)
            
            src_metric, dst_metric = metric[..., ::2, :], metric[..., 1::2, :]
            n, t1, c = src_metric.shape
            unm = src_metric.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            metric = torch.cat([unm, dst_metric], dim=1)

    return x


def iterative_token_drop_merge(x, metric, keep_num=64, class_token=True, r=32):
    
    while x.shape[1] > keep_num:
        with torch.no_grad():
            
            # metric = metric / metric.norm(dim=-1, keepdim=True)
            # a, b = metric[..., ::2, :], metric[..., 1::2, :]
            # scores = a @ b.transpose(-1, -2)
            
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = -torch.cdist(a, b, p=2)
            

            if class_token:
                scores[..., 0, :] = -math.inf

            node_max, node_idx = scores.max(dim=-1)
        
            r = min(r, x.shape[1] - keep_num)
            
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

            if class_token:
                unm_idx = unm_idx.sort(dim=1)[0]

            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            if x.shape[1] < 2 * keep_num:
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='mean')
            
            x = torch.cat([unm, dst], dim=1)
            
            src_metric, dst_metric = metric[..., ::2, :], metric[..., 1::2, :]
            n, t1, c = src_metric.shape
            unm = src_metric.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            metric = torch.cat([unm, dst_metric], dim=1)

    return x

def iterative_token_drop_with_l2(x, metric, keep_num=64, class_token=True, r=32, alpha=0.1):
    while x.shape[1] > keep_num:
        with torch.no_grad():
            norm_metric = metric / (metric.norm(dim=-1, keepdim=True) + 1e-8)
            
            a, b = norm_metric[..., ::2, :], norm_metric[..., 1::2, :]
            
            scores = a @ b.transpose(-1, -2)  # shape: [B, N/2, N/2]
            
            if class_token:
                scores[..., 0, :] = -float('inf')
            
            redundancy_score, _ = scores.max(dim=-1)  # shape: [B, N/2]
            
            importance = metric[..., ::2, :].norm(dim=-1)  # shape: [B, N/2]
            
            combined_score = redundancy_score - alpha * importance  # higher means more expendable
            
            r_drop = min(r, x.shape[1] - keep_num)
            
            edge_idx = combined_score.argsort(dim=-1, descending=True)[..., None]  # shape: [B, N/2, 1]
            
            keep_idx = edge_idx[..., r_drop:, :]

            if class_token:
                keep_idx = keep_idx.sort(dim=1)[0]
            
            src = x[..., ::2, :]  # even tokens
            n, t1, c = src.shape
            kept_src = src.gather(dim=-2, index=keep_idx.expand(n, t1 - r_drop, c))
            dst = x[..., 1::2, :]  # odd tokens
            
            # Concatenate the kept even tokens with all odd tokens.
            x = torch.cat([kept_src, dst], dim=1)
            
            # Do the same for the metric.
            src_metric = metric[..., ::2, :]
            n, t1, c = src_metric.shape 
            kept_src_metric = src_metric.gather(dim=-2, index=keep_idx.expand(n, t1 - r_drop, c))
            dst_metric = metric[..., 1::2, :]
            metric = torch.cat([kept_src_metric, dst_metric], dim=1)
    return x




def cluster_with_cls_keys(x, keys, k, metric="euclidean", num_iters=5):
    B, N, C = x.shape
    cls_token_x = x[:, :1, :]             # [B, 1, C] → CLS token (values)
    tokens_x = x[:, 1:, :]                # [B, N-1, C] → non-CLS values
    tokens_keys = keys[:, 1:, :].float()           # [B, N-1, C] → non-CLS keys
    M = tokens_keys.shape[1]
    if metric == "cosine":
        tokens_keys = F.normalize(tokens_keys, p=2, dim=-1)
    centroids = torch.stack([
        tokens_keys[b, torch.randperm(M, device=tokens_keys.device)[:k-1]]
        for b in range(B)
    ], dim=0)  # [B, k-1, C]
    
    for _ in range(num_iters):
        if metric == "cosine":
            centroids = F.normalize(centroids, p=2, dim=-1)
        if metric == "euclidean":
            diff = tokens_keys.unsqueeze(2) - centroids.unsqueeze(1)  # [B, M, k-1, C]
            dists = (diff ** 2).sum(dim=-1)                           # [B, M, k-1]
        elif metric == "cosine":
            sim = torch.bmm(tokens_keys, centroids.transpose(1, 2))   # [B, M, k-1]
            dists = 1 - sim
        else:
            raise ValueError("Unsupported metric")
        assignments = dists.argmin(dim=-1)                            # [B, M]
        one_hot = F.one_hot(assignments, num_classes=k-1).float()       # [B, M, k-1]
        counts = one_hot.sum(dim=1)                                    # [B, k-1]
        new_centroids = torch.bmm(one_hot.transpose(1, 2).float(), tokens_keys) / (counts.unsqueeze(-1) + 1e-8)
        if metric == "cosine":
            new_centroids = F.normalize(new_centroids, p=2, dim=-1)
        mask = counts < 1
        if mask.sum() > 0:
            rand_idx = torch.randint(0, M, (B, k-1), device=tokens_keys.device)
            batch_idx = torch.arange(B, device=tokens_keys.device).unsqueeze(1).expand(B, k-1)
            random_tokens = tokens_keys[batch_idx, rand_idx]
            if metric == "cosine":
                random_tokens = F.normalize(random_tokens, p=2, dim=-1)
            new_centroids = torch.where(mask.unsqueeze(-1), random_tokens, new_centroids)
        centroids = new_centroids

    # For each cluster, choose the token key closest to its centroid and take corresponding x.
    rep_x_list = []
    for b in range(B):
        reps = []
        for j in range(k-1):
            cluster_mask = (assignments[b] == j)
            if cluster_mask.sum() == 0:
                chosen_idx = torch.randint(0, M, (1,), device=tokens_keys.device).item()
            else:
                indices = torch.nonzero(cluster_mask, as_tuple=False).squeeze(-1)
                if metric == "euclidean":
                    diff = tokens_keys[b][cluster_mask] - centroids[b, j]
                    d = (diff ** 2).sum(dim=-1)
                else:  # cosine
                    sims = torch.matmul(tokens_keys[b][cluster_mask], centroids[b, j])
                    d = 1 - sims
                best = d.argmin().item()
                chosen_idx = indices[best].item()
            reps.append(tokens_x[b, chosen_idx:chosen_idx+1])
        rep_x = torch.cat(reps, dim=0)  # [k-1, C]
        rep_x_list.append(rep_x)
    rep_x_tensor = torch.stack(rep_x_list, dim=0)  # [B, k-1, C]
    return torch.cat([cls_token_x, rep_x_tensor], dim=1).half()  # [B, k, C]






def dbscan_with_cls_batch(x, keys, eps):
    # x, keys: [1, N, C]
    B, N, C = x.shape
    assert B == 1, "This implementation assumes batch size 1" 
    
    cls_x = x[:, :1, :]        # [1, 1, C]
    k_data = keys[:, 1:, :]      # [1, N-1, C]
    x_data = x[:, 1:, :]         # [1, N-1, C]
    
    # Remove batch dim for processing (since B==1)
    k_data = k_data[0]         # [M, C] with M = N-1
    x_data = x_data[0]         # [M, C]
    M = k_data.shape[0]
    
    # Compute pairwise Euclidean distances among non-CLS tokens.
    dists = torch.cdist(k_data, k_data, p=2)  # [M, M]
    # Build an adjacency matrix (True if distance < eps)
    A = dists < eps  # [M, M]
    I = torch.eye(M, dtype=torch.bool, device=A.device)
    R = A | I  # initial reachability matrix
    
    # Iteratively compute transitive closure
    while True:
        newR = R | ((R.float() @ R.float()) > 0)
        if torch.equal(newR, R):
            break
        R = newR
    
    # Assign a label to each token: use the minimal index in its connected component.
    labels = torch.empty(M, dtype=torch.long, device=R.device)
    for i in range(M):
        comp = torch.nonzero(R[i], as_tuple=True)[0]
        labels[i] = comp.min()
    
    # Map labels to consecutive indices.
    _, final_labels = torch.unique(labels, sorted=True, return_inverse=True)
    
    # For each cluster, compute the centroid and select the token whose key is closest.
    reps = []
    num_clusters = final_labels.max().item() + 1
    for cid in range(num_clusters):
        idx = (final_labels == cid).nonzero(as_tuple=True)[0]
        cluster_keys = k_data[idx]             # [n, C]
        centroid = cluster_keys.mean(dim=0, keepdim=True)  # [1, C]
        d = torch.norm(cluster_keys - centroid, dim=1)     # [n]
        best = idx[d.argmin()].unsqueeze(0)      # [1]
        reps.append(x_data[best])                # [1, C]
    
    if reps:
        reps = torch.cat(reps, dim=0)             # [num_clusters, C]
    else:
        reps = torch.empty((0, C), device=x.device)
    
    # Concatenate the CLS token with the cluster representatives.
    # cls_x: [1, 1, C] and reps (unsqueezed): [1, num_clusters, C]
    out = torch.cat([cls_x, reps.unsqueeze(0)], dim=1)
    return out





def dbscan_with_cls_cosine(x, keys, eps):
    # x, keys: [1, N, C] where the first token (index 0) is the CLS token.
    B, N, C = x.shape
    assert B == 1, "This implementation assumes batch size 1." 
    
    cls_x = x[:, :1, :]          # [1, 1, C]
    k_data = keys[:, 1:, :]        # [1, N-1, C]
    x_data = x[:, 1:, :]           # [1, N-1, C]
    M = k_data.shape[1]
    
    # Remove batch dim for clustering
    k_data = k_data[0]             # [M, C]
    x_data = x_data[0]             # [M, C]
    
    # Normalize keys and compute cosine similarity matrix
    k_norm = F.normalize(k_data, p=2, dim=-1)
    sim = k_norm @ k_norm.t()      # [M, M]
    
    # Build adjacency: tokens are neighbors if cosine similarity > eps.
    A = sim > eps
    I = torch.eye(M, dtype=torch.bool, device=A.device)
    R = A | I
    
    # Compute transitive closure (reachability matrix)
    while True:
        newR = R | ((R.float() @ R.float()) > 0)
        if torch.equal(newR, R):
            break
        R = newR
    
    # Label connected components using the minimal index in each component.
    labels = torch.empty(M, dtype=torch.long, device=R.device)
    for i in range(M):
        comp = torch.nonzero(R[i], as_tuple=True)[0]
        labels[i] = comp.min()
    
    # Map labels to consecutive integers.
    _, final_labels = torch.unique(labels, sorted=True, return_inverse=True)
    
    reps = []
    num_clusters = final_labels.max().item() + 1
    for cid in range(num_clusters):
        idx = (final_labels == cid).nonzero(as_tuple=True)[0]
        cluster_keys = k_data[idx]  # [n, C]
        centroid = cluster_keys.mean(dim=0, keepdim=True)  # [1, C]
        # For cosine, select the token with maximum cosine similarity to the centroid.
        cluster_norm = F.normalize(cluster_keys, p=2, dim=-1)
        centroid_norm = F.normalize(centroid, p=2, dim=-1)
        cos_sim = (cluster_norm * centroid_norm).sum(dim=1)  # [n]
        best = idx[cos_sim.argmax()].unsqueeze(0)
        reps.append(x_data[best])
        
    reps = torch.cat(reps, dim=0) if reps else torch.empty((0, C), device=x.device)
    # Concatenate the CLS token with the representatives.
    return torch.cat([cls_x, reps.unsqueeze(0)], dim=1)



def iterative_drop_full_graph(x, keys, keep_num=64, r=16):
    cls_token = x[:, :1, :] 
    tokens = x[:, 1:, :][0]
    keys = keys[:, 1:, :][0]
    
    # tokens = tokens[1::2, :]
    # keys = keys[1::2, :]

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