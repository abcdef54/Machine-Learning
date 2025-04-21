import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Callable


def train_test_split(test_size: float = 0.2):
    # Data Generation
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 1 + 2 * x + .1 * np.random.randn(100, 1)

    # Shuffles the indices
    idx = np.arange(100)
    np.random.shuffle(idx)

    test_count = int(100 * test_size)
    train_count = 100 - test_count

    # Uses first 80 random indices for train
    train_idx = idx[:train_count]
    # Uses the remaining indices for validation
    val_idx = idx[train_count:]

    # Generates train and validation sets
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    return x_train, x_val, y_train, y_val

def get_train_tensors(device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    x_train, _, y_train, _ = train_test_split()
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    return x_train_tensor, y_train_tensor

def get_test_tensors(device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    _, x_test, _, y_test = train_test_split()
    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)
    return x_test_tensor, y_test_tensor

def get_train_test_loader(test_size: float = 0.2, train_batch_size: int = 16, test_batch_size: int = 20) -> Tuple[DataLoader, DataLoader]:
    x_train_tensor, y_train_tensor = get_train_tensors(device='cpu')
    
    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    total_size = len(dataset)
    
    test_len = int(test_size * total_size)
    train_len = total_size - test_len
    
    torch.manual_seed(42)  # For reproducibility
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, test_loader

def make_train_step(model: nn.Module, loss_fn: Callable[[torch.tensor, torch.Tensor], torch.Tensor], optimizer: optim.Optimizer) -> Callable:
    def train_step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Sets model to TRAIN mode
        model.train()
        # Step 1: Makes predictions
        yhat = model(x)
        # Step 2: Computes loss
        loss = loss_fn(y, yhat)
        # Step 3: Computes gradients
        loss.backward()
        # Step 4: Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    return train_step