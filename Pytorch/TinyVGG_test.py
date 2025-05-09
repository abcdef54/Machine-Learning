import sys
import os
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Pytorch.TinyVGG import TinyVGGv2
from utils.ufuncs import train, plot_loss_curves, make_image_dataloader, make_image_dataset

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import os

import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    train_dataset, test_dataset = make_image_dataset(train_path, train_transformation, test_path, test_transformation)
    train_dataloader, test_dataloader = make_image_dataloader(train_dataset, test_dataset, num_workers=os.cpu_count(), batch_size=32, shuffle=True)

    model = TinyVGGv2(3, 24, 3).to(device)
    lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model_result = train(model, train_dataloader, test_dataloader, optimizer, loss_fn, device, epochs=30, progress=False, verbose=True, every=2)

    plot_loss_curves(model_result)
    plt.show()
