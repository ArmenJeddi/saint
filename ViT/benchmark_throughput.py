import time
import torch
from tqdm import tqdm

def benchmark_throughput(model, data_loader, device):
    warm_up = 20
    total_samples = 0
    
    model.to(device)
    model.eval()
    
    start = None  # Initialize start time
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, desc="Benchmarking Throughput")):
            if i == warm_up:
                torch.cuda.synchronize()
                total_samples = 0
                start = time.time()

            input, _ = data
            input = input.to(device) 
            
            model(input) 
            total_samples += input.shape[0] 
    
    torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    throughput = total_samples / elapsed

    return throughput
