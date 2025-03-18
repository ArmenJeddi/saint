from typing import List, Union

def parse_prune_mode(num_layers: int, prune_mode: Union[List[str], str]) -> List[str]:
    if isinstance(prune_mode, list):
        if len(prune_mode) < num_layers:
            prune_mode = prune_mode + [None] * (num_layers - len(prune_mode))
        return list(prune_mode)

    return [prune_mode for _ in range(num_layers)]

def parse_sim_threshold(num_layers: int, sim_threshold: Union[List[float], float]) -> List[float]:
    if isinstance(sim_threshold, list):
        if len(sim_threshold) < num_layers:
            sim_threshold = sim_threshold + [1.0] * (num_layers - len(sim_threshold))
        return list(sim_threshold)

    return [sim_threshold for _ in range(num_layers)]
