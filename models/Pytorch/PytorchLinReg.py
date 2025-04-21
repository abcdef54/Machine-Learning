import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from typing import Callable, Tuple, List
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler


class LinearRegressor:
    def __init__(self, n_epoch: int = 1000, lr: float = 1e-2) -> None:
        self.lr = lr
        self.n_epoch = n_epoch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = nn.Sequential(nn.Linear(1, 1)).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(reduction='mean')

    def train(self, train_loader: DataLoader) -> List[torch.Tensor]:
        losses = []
        for epoch in range(self.n_epoch):
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss = self._train_step(x_batch, y_batch)
                epoch_loss += loss
            average_loss = epoch_loss / len(train_loader)
            losses.append(average_loss)
        return losses

    def predict(self, test_tensor: torch.Tensor) -> torch.Tensor:
        test_tensor = test_tensor.to(self.device)

        with torch.inference_mode():
            return self.model(test_tensor).squeeze()

    def score(self, test_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        n_samples = len(test_loader.dataset)
        if n_samples < 2:
            raise ValueError('Need at least two test samples for R²')

        self.model.eval()
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_true_all.append(y_batch.cpu().numpy())
                y_pred_all.append(self.model(x_batch).cpu().numpy())

        y_true = np.concatenate(y_true_all).flatten()
        y_pred = np.concatenate(y_pred_all).flatten()

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            r2 = 1.0
        else:
            r2 = 1 - ss_res / ss_tot

        self.model.train()
        return r2, y_true, y_pred

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.model.train()

        yhat = self.model(x)
        loss = self.loss_fn(y, yhat)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

class CustomDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, test_size: float = .2, train_batch_size: int = 16,
    test_batch_size: int = 20, shuffle: bool = True)\
    -> None:
        self.data = x
        self.target = y
        self.train_batches = train_batch_size
        self.test_batches = test_batch_size
        self.test_size = test_size
        self.shuffle = shuffle

    def get_train_test_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        dataset: Dataset = TensorDataset(self.data, self.target)
        total_size = len(dataset)
        test_len = int(self.test_size * total_size)
        train_len = total_size - test_len
        train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_batches, shuffle=self.shuffle)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.test_batches, shuffle=self.shuffle)

        return train_loader, test_loader

def main():
    # Load and preprocess the dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = CustomDataset(X_tensor, y_tensor)
    train_loader, test_loader = dataset.get_train_test_dataloader()

    # Initialize the model
    model = LinearRegressor(n_epoch=1000, lr=1e-3)
    model.model = nn.Sequential(nn.Linear(10, 1)).to(model.device)
    model.optimizer = optim.SGD(model.model.parameters(), lr=model.lr)

    # Train the model
    losses = model.train(train_loader)

    # Plot training curve
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.grid(True)
    plt.savefig('training_curve.png')
    plt.close()

    # Evaluate the model
    r2, y_true, y_pred = model.score(test_loader)
    print(f"R² on test set: {r2:.4f}")

    # Print learned parameters
    w = model.model[0].weight.data.cpu().numpy()
    b = model.model[0].bias.data.cpu().numpy()
    print(f"Learned weights: {w}")
    print(f"Learned intercept: {b}")

    # Plot actual vs predicted values to visualize the linear model's fit
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Data Points")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Fit (y=x)")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values - Linear Model Fit")
    plt.legend()
    plt.grid(True)
    plt.savefig('actual_vs_predicted.png')
    plt.close()

if __name__ == "__main__":
    main()