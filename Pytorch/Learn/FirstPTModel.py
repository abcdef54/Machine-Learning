import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Callable, Tuple, Any
from torch.utils.data import Dataset, TensorDataset


# custom train test split / it works
from utils import train_test_split, get_train_tensors

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):
    def __init__(self, x_train_tensor: torch.Tensor, y_train_tesor: torch.Tensor) -> None:
        super().__init__()
        self.x_train_tensor = x_train_tensor
        self.y_train_tensor = y_train_tesor
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.x_train_tensor[index], self.y_train_tensor[index])
    
    def __len__(self) -> int:
        return len(self.x_train_tensor)
    
def testing():
    '''
    We don’t want our whole training data to be loaded into GPU tensors,
    as we have been doing in our example so far, because it takes up space in our precious graphics card’s RAM.
    '''
    x_train_tensor, y_train_tensor = get_train_tensors(device='cpu')
    
    train_data = CustomDataset(x_train_tensor, y_train_tensor)
    print(train_data[0])
    
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    print(train_data[0])

class ManualLinearRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.a: torch.Tensor = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b: torch.Tensor = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computes the outputs / predictions
        return self.a + self.b * x


class SimpleLinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    

    
def SigmaBoiSequential():
    model = nn.Sequential(
    nn.Linear(2, 10),  # Input layer: 2 features to 10 neurons
    nn.ReLU(),         # Activation function
    nn.Linear(10, 5),  # Hidden layer: 10 neurons to 5 neurons
    nn.ReLU(),         # Activation function
    nn.Linear(5, 1),   # Output layer: 5 neurons to 1 output
    nn.Sigmoid()       # Activation function for binary classification
    )

    # Example input tensor with 2 features
    x = torch.tensor([[0.5, -1.2]], dtype=torch.float32)

    # Forward pass through the model
    output = model(x).to(device)

    print("Model output:", output)
    
if __name__ == '__main__':
    testing()