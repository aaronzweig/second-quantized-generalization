import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_max_pool, GlobalAttention, GatedGraphConv
from torch_geometric.data import Data
from torch_geometric.utils import softmax

    
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
    
    
class GCNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, normalize = False):        
        super(GCNNet, self).__init__()
    
        self.conv1 = GCNConv(input_dim, hidden_dim, normalize = normalize)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize = normalize)
        self.pool = PoolNet(hidden_dim, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)
                        
    def forward(self, data):
        
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = self.pool(x, data.batch)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x.squeeze(1)
            
    