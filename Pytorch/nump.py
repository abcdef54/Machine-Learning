import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from typing import Callable, Tuple

from utils import get_train_tensors, make_train_step, train_test_split, get_train_test_loader
from FirstPTModel import CustomDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def np_linreg():
    x_train, x_test, y_train, y_test = train_test_split()

    np.random.seed(42)
    a = np.random.randn(1)  # Line 3
    b = np.random.randn(1)

    eta = 1e-1

    n_epochs = 1000

    for epoch in range(n_epochs):
        # Step 1: Computes our model's predicted output
        yhat = a + b * x_train

        # Step 2
        # How wrong is our model? That's the error!
        error = y_train - yhat
        loss = (error ** 2).mean()

        # Step 3
        # Computes gradients for both "a" and "b" parameters
        a_grad = -2 * error.mean()
        b_grad = -2 * (x_train * error).mean()

        # Step 4
        # Updates parameters using gradients and the learning rate
        a = a - eta * a_grad
        b = b - eta * b_grad

    print(a, b)

    from sklearn.linear_model import LinearRegression
    linr = LinearRegression()
    linr.fit(x_train, y_train)
    print(linr.intercept_, linr.coef_[0])

def torch_linreg():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_train, x_test, y_train, y_test = train_test_split()

    # Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
    # and then we send them to the chosen device
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)

    # Here we can see the difference - notice that .type() is more useful
    # since it also tells us WHERE the tensor is (device)
    #print(type(x_train), type(x_train_tensor), x_train_tensor.type())
    #x_train_tensor = x_train_tensor.cpu().numpy()
    #print(type(x_train_tensor))
    
    '''
    # We can either create regular tensors and send them to the device (as we did with our data)
    # and THEN set them as requiring gradients...
    
    a = torch.randn(1, dtype=torch.float).to(device)
    b = torch.randn(1, dtype=torch.float).to(device)
    a.requires_grad_(requires_grad =True)
    b.requires_grad_(requires_grad =True)
    print(a, b)
    '''
    
    lr = 1e-1
    n_epoch = 1000
    # We can specify the device at the moment of creation - RECOMMENDED!
    torch.manual_seed(42)
    a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    
    for epoch in range(n_epoch):
        yhat = a + b * x_train_tensor
        
        error = y_train_tensor - yhat
        loss = (error ** 2).mean()
        
        loss.backward()
        
        # We need to use NO_GRAD to keep the update out of the gradient computation
        # Why is that? It boils down to the DYNAMIC GRAPH that PyTorch uses...
        with torch.no_grad():
            a -= lr * a.grad
            b -= lr * b.grad
            
        a.grad.zero_()
        b.grad.zero_()
        
    print(a, b)
    
def torch_linreg2():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train, x_test, y_train, y_test = train_test_split()
    
    x_train_tensor: torch.Tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor: torch.Tensor = torch.from_numpy(y_train).float().to(device)
    
    torch.manual_seed(42)
    a: torch.Tensor = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b: torch.Tensor = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    
    lr = 1e-1
    n_epoch = 1000
    optimizer: optim.Optimizer = optim.SGD([a, b], lr=lr)
    
    for epoch in range(n_epoch):
        yhat: torch.Tensor = a + b * x_train_tensor
        error: torch.Tensor = y_train_tensor - yhat
        loss: torch.Tensor = (error ** 2).mean()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (epoch + 1) % 100 == 0:
            print(f'Learned parameter: a = {a.item():.4f}, b = {b.item():.4f}')
    print(a,b)
        
def torch_linreg3():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_train, x_test, y_train, y_test = train_test_split()
    
    x_train_tensor: torch.Tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor: torch.Tensor = torch.from_numpy(y_train).float().to(device)
    
    torch.manual_seed(42)
    a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    
    lr = 1e-1
    n_epoch = 1000
    optimizer: optim.Optimizer = optim.SGD([a, b], lr=lr)
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(reduction='mean')
    
    for epoch in range(n_epoch):
        yhat: torch.Tensor = a + b * x_train_tensor
        loss: torch.Tensor = loss_fn(y_train_tensor, yhat)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{n_epoch}], Loss: {loss.item():.4f}')
    
    print(f'Learned parameters: a = {a.item():.4f}, b = {b.item():.4f}')
    
def torch_linreg4():
    x_train, x_test, y_train, y_test = train_test_split()
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    
    torch.manual_seed(42)
    model = nn.Linear(1,1).to(device)
    print(model.state_dict())
    
    lr = 1e-1
    n_epoch = 1000
    loss_fn: Callable = nn.MSELoss(reduction='mean')
    optimizer: optim.Optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(n_epoch):
        model.train()
        
        yhat = model(x_train_tensor)
        loss = loss_fn(y_train_tensor, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(model.state_dict())
    

def torch_linreg5():
    x_train_tensor, y_train_tensor = get_train_tensors()
    lr = 1e-1
    n_epoch = 1000
    model: nn.Module = nn.Linear(1,1).to(device='cuda')
    optimizer: optim.Optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn: Callable = nn.MSELoss(reduction='mean')
    
    train_step = make_train_step(model, loss_fn, optimizer)
    losses = []
    
    for epoch in range(n_epoch):
        losses.append(train_step(x_train_tensor, y_train_tensor))
    
    print(model.state_dict())

def get_dataset() -> Dataset:
    x_train_tensor, y_train_tensor = get_train_tensors(device='cpu')
    return CustomDataset(x_train_tensor, y_train_tensor)

def torch_linreg6():
    train_data = get_dataset()
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    
    lr = 1e-1
    n_epoch = 1000
    torch.manual_seed(42)
    model: nn.Module = nn.Sequential(nn.Linear(1, 1)).to(device)
    loss_fn: Callable = nn.MSELoss(reduction='mean')
    optimizer: optim.Optimizer = optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    train_step = make_train_step(model, loss_fn, optimizer)
    
    for epoch in range(n_epoch):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)
    print(model.state_dict())

def torch_linreg_final():
    train_loader, test_loader = get_train_test_loader()
    
    lr = 1e-1
    n_epoch = 1000
    torch.manual_seed(42)
    model: nn.Module = nn.Sequential(nn.Linear(1, 1)).to(device)
    loss_fn: Callable = nn.MSELoss(reduction='mean')
    optimizer: optim.Optimizer = optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    val_losses = []
    train_step = make_train_step(model, loss_fn, optimizer)
    
    for epoch in range(n_epoch):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)
        
        with torch.inference_mode():
            for x_val, y_val in test_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                model.eval()
                yhat = model(x_val)
                val_loss = loss_fn(y_val, yhat)
                val_losses.append(val_loss.item())
    return losses, val_losses
                
                

if __name__ == '__main__':
    losses, val_losses = torch_linreg_final()
    print(f'Losses: {losses}')