from models.Pytorch import TinyVGG
from utils.ufuncs import train

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_path = 'data/pizza_steak_sushi_20_percent/train'
test_path = 'data/pizza_steak_sushi_20_percent/test'

train_transformation = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])
test_transformation = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

def make_dataset():
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transformation,
    )
    test_dataset = datasets.ImageFolder(
        root=test_path,
        transform=test_transformation
    )
    return train_dataset, test_dataset

def make_dataloader():
    train_dataset, test_dataset = make_dataset()

    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=os.cpu_count(), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=os.cpu_count(), shuffle=True)
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    train_dataloader, test_dataloader = make_dataloader()

    model = TinyVGG(1, 10, 3)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train(model, train_dataloader, test_dataloader, optimizer, loss_fn, device, epochs=10, progress=False, verbose=True, every=2)
