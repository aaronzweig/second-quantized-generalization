import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train(model, loader, lr = 0.003, iterations = 20, criterion = nn.MSELoss(), verbose = False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for i in range(iterations):
        batch_losses = []
        for data in loader:

            output = model(data)

            optimizer.zero_grad()
            loss = criterion(output, data.y.double())
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            
        batch_loss = np.mean(np.array(batch_losses))
        losses.append(batch_loss)

        if verbose:
            print("timestep: {}, loss: {}".format(i, batch_loss))
    
    model.eval()
    return losses

def test(model, loader, criterion = nn.MSELoss()):
    model.eval()

    losses = []
    for data in loader:

        output = model(data)

        loss = criterion(output, data.y.double())
        losses.append(loss.item())
    
    return np.mean(np.array(losses))