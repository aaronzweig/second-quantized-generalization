import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train(model, loader, lr = 0.003, iterations = 20, criterion = nn.MSELoss(), verbose = False, device = torch.device("cpu")):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for i in range(iterations):
        batch_losses = []
        for data in loader:

            data_device = data.to(device)

            output = model(data_device)

            optimizer.zero_grad()
            loss = criterion(output, data_device.y.double())
            loss.backward()
            optimizer.step()
            
            del data_device
            torch.cuda.empty_cache()

            batch_losses.append(loss.item())
            
        batch_loss = np.mean(np.array(batch_losses))
        losses.append(batch_loss)

        if verbose:
            print("timestep: {}, loss: {}".format(i, batch_loss))
    
    model.eval()
    return losses

def test(model, loader, criterion = nn.MSELoss(), device = torch.device("cpu")):
    model.eval()

    losses = []
    for data in loader:

        data_device = data.to(device)

        output = model(data_device)

        loss = criterion(output, data_device.y.double())
        losses.append(loss.item())
    
    return np.mean(np.array(losses))
