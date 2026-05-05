import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
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
gpu = False
conv = False

# Setup
error_func = nn.CrossEntropyLoss()    
func_f = torch.nn.ReLU()
func_c = F.softmax

print("Loading MNIST data...")
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = torchvision.datasets.MNIST('./data/MNIST', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

num_features = 28 * 28  # MNIST flattened
num_classes = 10
in_channels = 1

print(f"Data loaded - Features: {num_features}, Classes: {num_classes}, Channels: {in_channels}")
print("Building model...")
model = ver.Verlet(device, N, num_features, num_classes, func_f, func_c, None, None, gpu, True, conv, True, in_channels, 6)

print(f"Starting training for {epochs} epochs...")
train_time = time.perf_counter()
model.train(train_loader, error_func, learn_rate, epochs, begin, end, step, reg_f, alpha_f, reg_c, alpha_c, False)
train_time = time.perf_counter() - train_time

print(f"\nTraining completed in {train_time:.2f} seconds")
print(f"Average time per epoch: {train_time/epochs:.2f} seconds")
