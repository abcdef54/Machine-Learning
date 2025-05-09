import torch
import torch.nn as nn
import torch.optim as optim

from utils.ufuncs import to_tensors, make_train_test_dataloader
from typing import Iterable


class CustomLinearModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_layers: int = 0, hidden_units: int = 16) -> None:
        super().__init__()
        if hidden_layers < 0:
            raise ValueError('Hidden layer can not be smaller than 0.')
        elif hidden_units <= 0:
            raise ValueError('Hidden units must be at least 1 or bigger.')

        if hidden_layers > 0:
            sequence = [nn.Linear(in_features, hidden_units)]
            for _ in range(hidden_layers):
                sequence.append(nn.Linear(hidden_units, hidden_units))
            sequence.append(nn.Linear(hidden_units, out_features))

            self.model = nn.Sequential(*sequence)
        else:
            self.model = nn.Linear(in_features, out_features)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class LinearModel:
    def __init__(self, in_features: int, out_features: int, hidden_layers: int = 0, hidden_units: int = 16,
                 lr: float = 1e-3, device: torch.device = None, random_state: int = 42, optimizer: str = 'SGD',
                 loss_function: str = 'MSE', task: str = 'regression') -> None:
        torch.manual_seed(random_state)
        self.model = CustomLinearModel(in_features, out_features, hidden_layers, hidden_units).to(device)

        self.lr = lr
        self.random_state = random_state
        self.device = device or torch.device('cpu')

        if task.lower() not in ['classification', 'regression']:
            raise ValueError('Invalid task type. Support "regression" and "classification".')
        self.task = task.lower()

        if loss_function.lower() == 'mse':
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif loss_function.lower() == 'mae':
            self.loss_fn = nn.L1Loss()
        else: raise ValueError('Not supported loss function.')
        
        if optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else: raise ValueError('Not supported optimizer.')

    def fit(self, data: Iterable = None, labels: Iterable = None, train_split: float = .2, max_epochs: int = 100, verbose: bool = False,
            batch_size: int = 32, shuffle: bool = True, test_every: int = 10) -> None:
        torch.manual_seed(self.random_state)

        if len(data) <= 0 or len(labels):
            raise ValueError('Training or testing data is empty.')
        elif len(data) != len(labels):
            raise ValueError('Mismatch length of data and labels.')
        
        if max_epochs <= 0:
            raise ValueError('Please train more than 0 epoch.')
        
        # Turn the inputs to tensors
        data = to_tensors(data, dtype=torch.float32)
        
        if self.task == 'regression':
            labels = to_tensors(labels, dtype=torch.float32)
        elif self.task == 'classification':
            labels = to_tensors(labels, dtype=torch.long)
        
        # Make train test data loader
        train_dataloader, test_dataloader = make_train_test_dataloader(data, labels, train_split, batch_size, shuffle=shuffle)
        
        # Fit the model
        for epoch in range(max_epochs):
            total_train_loss = 0.0
            total_test_loss = 0.0

            for batch, (xb, yb) in enumerate(train_dataloader):
                total_train_loss += self._train_step(xb, yb)

            if (epoch + 1) % test_every == 0:
                for xb, yb in test_dataloader:
                    total_test_loss += self._test_step(xb, yb)
            
            if not verbose: continue
            mean_train_loss = total_train_loss / len(train_dataloader)
            mean_test_loss = total_test_loss / len(test_dataloader)
            if (epoch + 1) % test_every != 0:
                print(f'Epoch: {epoch + 1} | Train loss: {mean_train_loss}')
            else:
                print(f'Epoch: {epoch + 1} | Train loss: {mean_train_loss} | Test loss: {mean_test_loss}')


    def predict(self, X_test: Iterable, y_test: Iterable) -> Iterable:
        torch.manual_seed(self.random_state)
        if len(X_test) <= 0 or len(y_test):
            raise ValueError('Training or testing data is empty.')
        elif len(X_test) != len(y_test):
            raise ValueError('Mismatch length of data and labels.')
        
    def score():
        pass

    @torch.jit.script
    def _train_step(self, xb: torch.Tensor, yb: torch.Tensor) -> float:
        """Return total loss of the train step"""
        self.model.train()
        train_loss = 0.0

        xb = xb.to(self.device)
        yb = yb.to(self.device)

        logits = self.model(xb)
        loss = self.loss_fn(logits, yb)
        train_loss += loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return train_loss

    def _test_step(self, xb: torch.Tensor, yb: torch.Tensor) -> float:
        """Return total loss of the test step"""
        self.model.eval()
        test_loss = 0.0
        with torch.inference_mode():
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            test_loss += loss
        
        return test_loss
