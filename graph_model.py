import numpy as np

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, GraphConv, GlobalAttention, NNConv, global_add_pool, global_mean_pool
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch_geometric.utils import softmax

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

from torch_geometric.nn.inits import glorot, zeros


class PoolNet(nn.Module):
    def __init__(self, input_dim, hidden_dim = None):
        super(PoolNet, self).__init__()
        
        if hidden_dim is None:
            self.nn = nn.Linear(input_dim, 1)
        else:
            self.nn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.pool = GlobalAttention(self.nn)

    def forward(self, inputs, batch):
        return self.pool(inputs, batch)

#edited version
from cgconv import CGConv
    
class SecondNet(nn.Module):
    def __init__(self, vertex_dim, edge_dim, hidden_dim, p = 0.0):        
        super(SecondNet, self).__init__()
    
        #TODO: Try NNConv / our edge-attention network
    
        self.conv1 = CGConv(vertex_dim, hidden_dim, edge_dim)
        self.conv2 = CGConv(hidden_dim, hidden_dim, edge_dim)

        self.dropout = nn.Dropout(p)
        
        self.pool = PoolNet(hidden_dim, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)
                        
    def forward(self, data):

        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = self.conv2(x, data.edge_index, data.edge_attr)
        
#         x = self.pool(x, data.batch)
#         x = global_add_pool(x, data.batch)
        x = F.relu(self.l1(x))
        x = self.dropout(x)
        x = self.l2(x)
        
        return x.squeeze(1)
    
    def forward_energy(self, data):
        x = self.forward(data)
        x = global_add_pool(x * data.mask, data.batch)
        return 0.25 * x
    
class SimpleNet(nn.Module):
    def __init__(self, vertex_dim, edge_dim, hidden_dim, p = 0.0):        
        super(SimpleNet, self).__init__()
        
        self.l1 = nn.Linear(vertex_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)
                
        self.dropout = nn.Dropout(p)
                        
    def forward(self, data):
        
        x = F.relu(self.l1(data.x))
        x = self.dropout(x)
        x = F.relu(self.l2(x))
        x = self.dropout(x)
        x = F.relu(self.l3(x))
        x = self.l4(x)
                
        return x.squeeze(1)
    
    def forward_energy(self, data):
        x = self.forward(data)
        x = global_add_pool(x * data.mask, data.batch)
        return 0.25 * x

class THCNet(nn.Module):
    def __init__(self, vertex_dim, edge_dim, hidden_dim, heads = 1):        
        super(THCNet, self).__init__()

        self.l1 = nn.Linear(vertex_dim, hidden_dim)
        self.conv1 = TransformerConv(hidden_dim, hidden_dim, heads = heads, edge_dim = edge_dim)
        self.l2 = nn.Linear(hidden_dim * heads, hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, heads = heads, edge_dim = edge_dim)
        self.l3 = nn.Linear(hidden_dim * heads, 1)
                
#         self.conv1 = CGConv(vertex_dim, hidden_dim, edge_dim)
#         self.l1 = nn.Linear(hidden_dim, hidden_dim)
#         self.conv2 = CGConv(hidden_dim, hidden_dim, edge_dim)
#         self.l2 = nn.Linear(hidden_dim, 1)
 
    def forward(self, data):

        x = F.relu(self.l1(data.x))
        x = F.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.relu(self.l2(x))
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        x = self.l3(x)
        
        return x.squeeze(-1)
