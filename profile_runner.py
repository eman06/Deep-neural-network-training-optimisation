import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dataloader as dl
import Verlet as ver
import time

# Set random seeds
np.random.seed(11)
torch.manual_seed(11)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu")  # Force CPU
print(f"Device: {device}")

# Configuration
dataset_name = "MNIST"
batch_size = 256
N = 64
learn_rate = 0.001
step = 0.01
epochs = 5
begin = 0
end = 10000
reg_f = True
reg_c = False
alpha_f = 0.0001
alpha_c = 0.01
gpu = True
conv = False

# Setup
error_func = nn.CrossEntropyLoss()    
func_f = torch.nn.ReLU()
func_c = F.softmax

print("Loading data...")
dataloader_obj = dl.InMemDataLoader(dataset_name)
train_loader = dataloader_obj.getDataLoader(batch_size, shuffle=True, num_workers=0, pin_memory=False, train=True)
num_features, num_classes, in_channels = dl.getDims(dataset_name)

print(f"Data loaded - Features: {num_features}, Classes: {num_classes}, Channels: {in_channels}")
print("Building model...")
model = ver.Verlet(device, N, num_features, num_classes, func_f, func_c, None, None, gpu, True, conv, True, in_channels, 6)

print(f"Starting training for {epochs} epochs...")
train_time = time.perf_counter()
model.train(train_loader, error_func, learn_rate, epochs, begin, end, step, reg_f, alpha_f, reg_c, alpha_c, False)
train_time = time.perf_counter() - train_time

print(f"\nTraining completed in {train_time:.2f} seconds")
print(f"Average time per epoch: {train_time/epochs:.2f} seconds")
