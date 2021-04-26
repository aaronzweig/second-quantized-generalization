import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy import sparse

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_max_pool, GlobalAttention, GatedGraphConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import softmax
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data, DataLoader


from pyscf import gto, scf, tools, ao2mo

from graph_model import SecondNet, SimpleNet, THCNet
from preprocess import build_qm7, build_thc_graph
from thc import THCContainer
from utils import khatri_rao

def train(model, loader, lr = 0.003, iterations = 10, verbose = False, lamb = 1.0, device = torch.device("cpu")):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for i in range(iterations):
        batch_losses = []
        for data in loader:

            E_THC = data.con.E_THC[0] # first term means the J term
            E_THC = torch.from_numpy(E_THC).to(device)
            
            E_hat = model(data.to(device))[data.E_mask.to(device)].reshape(E_THC.shape)
            E_pred = E_hat + E_THC
            #E_hat = torch.atan(E_hat) * (2.0 / np.pi) * lamb
            #E_pred = E_THC * E_hat

            E_true = data.con.E[2] # 2 means the total MP2
            E_true = torch.from_numpy(E_true).to(device)

            loss = torch.norm(E_true - E_pred) / torch.norm(E_true) #Scale regularization
            scalar_loss = torch.abs(torch.sum(E_true) - torch.sum(E_pred))
            dummy_loss = torch.abs(torch.sum(E_true) - torch.sum(E_THC))

#             zero = torch.tensor([0]).to(device)
#             one = torch.tensor([1]).to(device)
#             target = torch.where(E_true > E_THC, one, zero).double()
#             loss = torch.nn.BCELoss()(torch.sigmoid(E_hat), target)

            optimizer.zero_grad()
            loss.backward()


            optimizer.step()

            batch_losses.append((loss.item(), scalar_loss.item(), dummy_loss.item()))

            if i  == iterations-1:
                print("final comp")
                #print("THC loss: {:e}".format(torch.norm(E_true - E_THC)))
                #print("our loss: {:e}".format(torch.norm(E_true - E_pred)))

        batch_loss = np.mean(np.array([l[0] for l in batch_losses]))
        losses.append(batch_loss)
        if verbose:
            for loss, scalar_loss, dummy_loss in batch_losses:
                print("timestep: {}, loss: {:e}, scalar_loss: {:e}, dummy_loss: {:e}".format(i, loss, scalar_loss, dummy_loss))

    model.eval()
    return losses

def get_args():
    parser = argparse.ArgumentParser(description='THC')
    
    parser.add_argument('--grid_points_per_atom', type=int, default=100)
    parser.add_argument('--epsilon_qr', type=float, default=1e-15)
    parser.add_argument('--epsilon_inv', type=float, default=1e-15)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--basis', default='sto-3g')
    parser.add_argument('--dataset_size', type=int, default=5)
    parser.add_argument('--heads', type=int, default=2)

    parser.add_argument('--hidden_dim', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lamb', type=float, default=1e-2)
    parser.add_argument('--iterations', type=int, default=200)

    
    parser.add_argument('--no-cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    return args


if __name__ == "__main__":
    
    args = get_args()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    mols = build_qm7(args.basis)
    mols = mols[:args.dataset_size]
    kwargs = {'grid_points_per_atom': args.grid_points_per_atom,
              'epsilon_qr': args.epsilon_qr,
              'epsilon_inv': args.epsilon_inv,
              'verbose': args.verbose}
    mol_data = [THCContainer(mol, kwargs) for mol in mols]

    dataset = []
    for con in mol_data:

        if args.verbose:
            print("E_J loss", np.linalg.norm(con.E[0] - con.E_THC[0]))
            print("E loss", np.linalg.norm(con.E[2] - con.E_THC[2]))
            print("MP2_J loss", np.linalg.norm(con.MP2[0] - con.MP2_THC[0]))
            print("MP2 loss", np.linalg.norm(con.MP2[2] - con.MP2_THC[2]))
            print(con.E[2].shape)
            print("")
            
        data = build_thc_graph(con)

        dataset.append(data)

    vertex_dim = dataset[0].x.shape[1]
    edge_dim = dataset[0].edge_attr.shape[1]
    model = THCNet(vertex_dim, edge_dim, args.hidden_dim, args.heads).double()
    
    model.to(device)

    losses = train(model, dataset, iterations = args.iterations, lr = args.lr, verbose = args.verbose, lamb = args.lamb,
                   device = device)
