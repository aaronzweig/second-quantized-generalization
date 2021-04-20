import numpy as np
from pyscf import gto, scf, tools, ao2mo, mp
import scipy.linalg
import pickle
import os.path
from os import path
from collections import namedtuple

def khatri_rao(X, Y):
    return np.einsum("ij,kj->ikj",X,Y).reshape((X.shape[0] * Y.shape[0], -1))

def torch_khatri_rao(X, Y):
    C = torch.einsum("ij,kj->ikj",X,Y)
    C = C.view(X.shape[0] * Y.shape[0], -1)
    return C