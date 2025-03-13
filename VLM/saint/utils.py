import time
from typing import List, Tuple, Union

import torch
from tqdm import tqdm


def benchmark(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model (channels, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    input = torch.rand(batch_size, *input_size, device=device)
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()

                model(input)
                total += batch_size

    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start

    throughput = total / elapsed

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")

    return throughput


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
