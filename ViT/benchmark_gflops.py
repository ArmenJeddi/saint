import torch
from torchprofile import profile_macs
from tqdm import tqdm

def benchmark_gflops(model, data_loader, device):
    total_macs = 0
    total_samples = 0
    
    model.to(device)
    model.eval()

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Benchmarking GFLOPs"):
            x = data[0].to(device)
            macs = profile_macs(model, x)
            total_macs += macs / 1e9  # Convert to GFLOPs
            total_samples += x.shape[0]  # Accumulate number of samples

    gflops = total_macs / total_samples if total_samples > 0 else 0
    return gflops
