import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

import os
import zipfile
from pathlib import Path
import requests

from typing import Dict, List, Tuple
from tqdm.auto import tqdm


def download_20percent_data():
    # Setup path to data folder
    data_path_20 = Path("data/")
    image_path_20 = data_path_20 / "pizza_steak_sushi_20_percent"

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path_20.is_dir():
        print(f"{image_path_20} directory exists.")
    else:
        print(f"Did not find {image_path_20} directory, creating one...")
        image_path_20.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        with open(data_path_20 / "pizza_steak_sushi_20_percent.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path_20 / "pizza_steak_sushi_20_percent.zip", "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...") 
            zip_ref.extractall(image_path_20)


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
    plt.show()


def train_step(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module, device: str = None) -> Tuple[float, float]:
  """Make a train step on the specified device (default cpu) and return mean train loss and mean train accuracy"""
  device = device or torch.device('cpu')
  model.train()
  train_loss, train_acc = 0.0, 0.0
  total = 0

  for xb, yb in dataloader:
    xb = xb.to(device)
    yb = yb.to(device)

    logits = model(xb)
    loss = loss_fn(logits, yb)
    train_loss += loss.item() * xb.size(0)
    train_acc += (yb == torch.argmax(logits, dim=1)).sum().item()
    total += yb.size(0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss /= total
  train_acc /= total
  return train_loss, train_acc


def test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: str = None) -> Tuple[float, float]:
  """Make a train step on the specified device (default cpu) and return mean test loss and mean test accuracy"""
  device = device or torch.device('cpu')
  model.eval()
  test_loss, test_acc = 0.0, 0.0
  total = 0

  with torch.inference_mode():
    for xb, yb in dataloader:
      xb = xb.to(device)
      yb = yb.to(device)

      logits = model(xb)
      loss = loss_fn(logits, yb)
      test_loss += loss.item() * xb.size(0)
      test_acc += (yb == torch.argmax(logits, dim=1)).sum().item()
      total += yb.size(0)

    test_loss /= total
    test_acc /= total
  return test_loss, test_acc


def train(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module,
          device: str = None, epochs: int = 5, progress: bool = True, verbose: bool = True, every: int = 5) -> Dict[str, List[float]]:
  device = device or torch.device('cpu')
  result = {'train_loss' : [],
            'train_acc' : [],
            'test_loss' : [],
            'test_acc' : []}

  if progress:
    for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model, train_dataloader, optimizer, loss_fn, device)
      test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

      if verbose and (epoch + 1) % every == 0:
        print(f'Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.2f} | Test loss: {test_loss:.4f} | Test acc: {test_acc*100:.2f}')

      result["train_loss"].append(train_loss)
      result["train_acc"].append(train_acc)
      result["test_loss"].append(test_loss)
      result["test_acc"].append(test_acc)
  
  else:
    for epoch in range(epochs):
      train_loss, train_acc = train_step(model, train_dataloader, optimizer, loss_fn, device)
      test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

      if verbose and (epoch + 1) % every == 0:
        print(f'Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.2f} | Test loss: {test_loss:.4f} | Test acc: {test_acc*100:.2f}')

      result["train_loss"].append(train_loss)
      result["train_acc"].append(train_acc)
      result["test_loss"].append(test_loss)
      result["test_acc"].append(test_acc)

  return result


def make_image_dataset(train_root: str, train_transform: nn.Module, test_root: str, test_transform: nn.Module) -> Tuple[torch.utils.data.Dataset]:
    train_dataset = datasets.ImageFolder(
        root=train_root,
        transform=train_transform,
    )
    test_dataset = datasets.ImageFolder(
        root=test_root,
        transform=test_transform
    )
    return train_dataset, test_dataset


def make_image_dataloader(train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset,
                    num_workers: int = 1, batch_size: int = 32, shuffle: bool = True) -> Tuple[torch.utils.data.DataLoader]:
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return train_dataloader, test_dataloader


def _to_tensors(*args, dtype: torch.dtype) -> Tuple[torch.Tensor]:
        """Turn python lists or numpy ndarrays into tensors with the specified pytorch data type"""
        tensors = []
        for lis in args:
            if isinstance(lis, list):
                tensors.append(torch.tensor(lis, dtype=dtype))
            elif isinstance(lis, np.ndarray):
                tensors.append(torch.from_numpy(lis).type(dtype=dtype))
            else:
                raise TypeError('Unsupport array type, only support list and numpy ndarray.')

        return tuple(tensors)