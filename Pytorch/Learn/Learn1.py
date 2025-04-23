import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Tuple, Callable

BATCH_SIZE = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def download() -> Tuple[Dataset, Dataset]:
    train_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return train_data, test_data

def make_data_loader(dataset: Dataset, batch_size: int, shuffle = True) -> DataLoader:
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

class NeuralNetWork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(train_dataloader: DataLoader, model: nn.Module, loss_fn: Callable, optimizer: optim.Optimizer) -> None:
    size = len(train_dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'loss: {loss:.7f}, current: {current:>5}, size: {size:>5}')

def test(test_dataloader: DataLoader, model: nn.Module, loss_fn: Callable) -> None:
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test error:\nAccuracy: {100*correct:0.1f}%\n'
          f'Avg loss: {test_loss:>8f}\n')

if __name__ == '__main__':
    train_dataset, test_dataset = download()
    train_dataloader = make_data_loader(train_dataset)
    test_dataloader = make_data_loader(test_dataset)

    model = NeuralNetWork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
