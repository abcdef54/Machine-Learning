import torch
import torch.nn as nn
from typing import Tuple

class TinyVGGv2(nn.Module):
    def __init__(self, in_features: int, hidden_units: int, out_features: int) -> None:
        super().__init__()
        DROP_RATE = 0.4
        self.block1 = nn.Sequential( # Input images shape (3,64,64)
            nn.Conv2d(in_channels=in_features, out_channels=hidden_units, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout2d(p=DROP_RATE)
        )
        self.block2 = nn.Sequential( # images shape (hidden_units,32,32)
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
        )
        self.linear = nn.Sequential( # images shape (hidden_units,16,16)
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16, out_features=out_features)
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(self.block2(self.block1(X))) # Operation fusion
    

class TinyVGG(nn.Module):
    def __init__(self, in_features: int, hidden_units: int, out_features: int, image_size: Tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dynamically determine the input features for the linear layer
        with torch.inference_mode():
            dummy_input = torch.zeros(1, in_features, image_size[0], image_size[1])
            x = self.block1(dummy_input)
            x = self.block2(x)
            flattened_size = x.view(1, -1).size(1)

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_size, out_features=out_features)
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(self.block2(self.block1(X)))