"""
Comprehensive Profiling Script for Deep Neural Network Training
Uses PyTorch's torch.profiler for detailed CPU/GPU performance analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.profiler import profile, record_function, ProfilerActivity
import time
import sys

# Import local modules
import Verlet as ver
import dataloader as dl

# ==================== CONFIGURATION ====================
PROFILE_DEVICE = "cpu"  # "cpu" or "cuda:0"
BATCH_SIZE = 64  # Smaller batch for profiling to reduce memory
NUM_EPOCHS = 1
PROFILE_BATCHES = 10  # Only profile first N batches
USE_MIXED_PRECISION = False  # torch.cuda.is_available()
# =====================================================

def load_mnist_data(batch_size=64):
    """Load MNIST dataset with optimized dataloader"""
    print("Loading MNIST data with optimized dataloader...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = torchvision.datasets.MNIST(
        './data/MNIST', 
        train=True, 
        download=False, 
        transform=transform
    )
    
    # Use optimized dataloader with multi-threaded prefetching
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  # Multi-threaded loading
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader


def profile_training_loop():
    """Profile the complete training loop with torch.profiler"""
    
    device = torch.device(PROFILE_DEVICE)
    print(f"\n{'='*80}")
    print(f"PYTORCH PROFILER - DNN TRAINING OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Profiling first {PROFILE_BATCHES} batches")
    print(f"{'='*80}\n")
    
    # ==================== DATA LOADING ====================
    print("[1/3] Loading data...")
    train_loader = load_mnist_data(BATCH_SIZE)
    
    num_features = 28 * 28
    num_classes = 10
    in_channels = 1
    N = 32  # Reduced for faster profiling
    
    # ==================== MODEL SETUP ====================
    print("[2/3] Building model...")
    model = ver.Verlet(
        device, N, num_features, num_classes,
        torch.nn.ReLU(), F.softmax,
        None, None, False, True, False, True, in_channels, 6
    ).to(device)
    
    error_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    # ==================== PROFILING ====================
    print("[3/3] Starting profiling...")
    print()
    
    # Warm up GPU/cache
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx >= 2:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            if not model.conv:
                inputs = inputs.view(-1, num_features)
            _ = model(inputs, 0.01)
    
    # ==================== MAIN PROFILING ====================
    profiler_activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        profiler_activities.append(ProfilerActivity.CUDA)
    
    profile_dict = {
        'forward_times': [],
        'backward_times': [],
        'data_load_times': [],
        'optimizer_times': []
    }
    
    batch_start = time.perf_counter()
    
    with profile(
        activities=profiler_activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx >= PROFILE_BATCHES:
                break
            
            # ========== DATA LOADING TIMING ==========
            data_start = time.perf_counter()
            inputs, labels = inputs.to(device), labels.to(device)
            if not model.conv:
                inputs = inputs.view(-1, num_features)
            data_time = time.perf_counter() - data_start
            profile_dict['data_load_times'].append(data_time)
            
            # ========== FORWARD PASS ==========
            optimizer.zero_grad()
            
            forward_start = time.perf_counter()
            with record_function("forward_pass"):
                if USE_MIXED_PRECISION and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs, 0.01)
                        loss = error_func(outputs, labels)
                else:
                    outputs = model(inputs, 0.01)
                    loss = error_func(outputs, labels)
            forward_time = time.perf_counter() - forward_start
            profile_dict['forward_times'].append(forward_time)
            
            # ========== BACKWARD PASS ==========
            backward_start = time.perf_counter()
            with record_function("backward_pass"):
                loss.backward()
            backward_time = time.perf_counter() - backward_start
            profile_dict['backward_times'].append(backward_time)
            
            # ========== OPTIMIZER STEP ==========
            opt_start = time.perf_counter()
            with record_function("optimizer_step"):
                optimizer.step()
            opt_time = time.perf_counter() - opt_start
            profile_dict['optimizer_times'].append(opt_time)
            
            if (batch_idx + 1) % max(1, PROFILE_BATCHES // 4) == 0:
                print(f"Profiled {batch_idx + 1}/{PROFILE_BATCHES} batches "
                      f"(Loss: {loss.item():.4f})")
    
    total_profile_time = time.perf_counter() - batch_start
    
    # ==================== PRINT RESULTS ====================
    print("\n" + "="*80)
    print("PROFILING RESULTS")
    print("="*80)
    
    print("\nPer-Batch Timing (milliseconds):")
    print("-"*80)
    print(f"  Data Loading:    {np.mean(profile_dict['data_load_times'])*1000:.3f}ms "
          f"(±{np.std(profile_dict['data_load_times'])*1000:.3f}ms)")
    print(f"  Forward Pass:    {np.mean(profile_dict['forward_times'])*1000:.3f}ms "
          f"(±{np.std(profile_dict['forward_times'])*1000:.3f}ms)")
    print(f"  Backward Pass:   {np.mean(profile_dict['backward_times'])*1000:.3f}ms "
          f"(±{np.std(profile_dict['backward_times'])*1000:.3f}ms)")
    print(f"  Optimizer Step:  {np.mean(profile_dict['optimizer_times'])*1000:.3f}ms "
          f"(±{np.std(profile_dict['optimizer_times'])*1000:.3f}ms)")
    
    total_avg = (np.mean(profile_dict['data_load_times']) +
                 np.mean(profile_dict['forward_times']) +
                 np.mean(profile_dict['backward_times']) +
                 np.mean(profile_dict['optimizer_times']))
    
    print(f"\n  Total per Batch: {total_avg*1000:.3f}ms")
    print(f"  Total Profile:   {total_profile_time:.2f}s")
    
    # Time Breakdown Percentage
    print("\nTime Distribution (%):")
    print("-"*80)
    data_pct = (np.mean(profile_dict['data_load_times']) / total_avg) * 100
    fwd_pct = (np.mean(profile_dict['forward_times']) / total_avg) * 100
    bwd_pct = (np.mean(profile_dict['backward_times']) / total_avg) * 100
    opt_pct = (np.mean(profile_dict['optimizer_times']) / total_avg) * 100
    
    print(f"  Data Loading:    {data_pct:6.2f}% {'█' * int(data_pct/2)}")
    print(f"  Forward Pass:    {fwd_pct:6.2f}% {'█' * int(fwd_pct/2)}")
    print(f"  Backward Pass:   {bwd_pct:6.2f}% {'█' * int(bwd_pct/2)}")
    print(f"  Optimizer Step:  {opt_pct:6.2f}% {'█' * int(opt_pct/2)}")
    
    # ==================== TORCH PROFILER TABLE ====================
    print("\n" + "="*80)
    print("PYTORCH PROFILER SUMMARY (Top 10 Operations)")
    print("="*80 + "\n")
    
    # Use key_avg for aggregated view
    prof.key_avg().table(sort_by="cpu_time_total", row_limit=10)
    
    # ==================== MEMORY PROFILING ====================
    print("\n" + "="*80)
    print("MEMORY USAGE")
    print("="*80)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model Size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024:.2f} MB")
    
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
    
    # ==================== EXPECTED EPOCH TIME ====================
    print("\n" + "="*80)
    print("ESTIMATED TRAINING TIMES")
    print("="*80)
    
    batches_per_epoch = 60000 // BATCH_SIZE  # MNIST has 60,000 samples
    epoch_time = total_avg * batches_per_epoch
    
    print(f"Batches per Epoch: {batches_per_epoch}")
    print(f"Estimated Time per Epoch: {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes)")
    print(f"Estimated Time for 50 Epochs: {epoch_time*50:.2f} seconds ({epoch_time*50/60:.2f} minutes)")
    
    print("\n" + "="*80)
    print("OPTIMIZATION TIPS")
    print("="*80)
    print("""
1. Data Loading Bottleneck?
   - Increase num_workers (currently using 4)
   - Use pin_memory=True for GPU training
   - Pre-cache dataset in RAM

2. Forward/Backward Pass Slow?
   - Enable GPU acceleration (use CUDA)
   - Enable mixed precision (AMP)
   - Reduce model size (N parameter)

3. Optimizer Overhead?
   - Use fused optimizers (e.g., torch.optim.RAdam)
   - Accumulate gradients
   - Use gradient checkpointing
    """)
    
    print("="*80)
    print("Profiling complete! See above for detailed breakdown.")
    print("="*80 + "\n")


if __name__ == "__main__":
    profile_training_loop()
