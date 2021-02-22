import numpy as np
import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy import sparse
from scipy.io import loadmat
from pyscf import gto, scf, tools, ao2mo

def build_qm7(basis):
    x = loadmat('qm7.mat')
    X = x['X'] # Coulomb matrix
    R = x['R'] # Coordinates
    Z = x['Z'] # Atomic charge
    T = x['T'] # Atomization Energies (useless for us?)
    
    element_dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}
    
    mols = []
    for i in range(R.shape[0]):
        mol = gto.Mole(symmetry=False)

        atoms = []
        for j in range(R.shape[1]):
            if Z[i,j] == 0:
                continue
            element = element_dict[Z[i,j]]
            coor = tuple(R[i,j])
            atom = [element, coor]
            atoms.append(atom)

        mol.atom = atoms
        mol.verbose = 0
        mol.symmetry = False
        mol.spin = 0
        mol.basis = basis
        mol.unit = "Bohr" #QM7b coordinate unit

        mols.append(mol)
    return mols


def vertex_index(i, j, n):
    r = n * (n+1) / 2 - (n-i) * (n-i+1) / 2
    return int(r + j - i)

# TEST:
# M = 5
# for i in range(M):
#     for j in range(i, M+1):
#         print(i,j, vertex_index(i,j,M+1))

def concat(arr):
    arr = [x if isinstance(x, np.ndarray) else np.array([x]) for x in arr]
    return np.concatenate(arr)


#M: total number of orbitals
#N: total number of particles

#A is potential matrix: M x M
#U is coulumb 4-tensor: M x M x M x M
#X is additional orbital feature matrix: M x F_1
#Y is additional pairwise orbital feature matrix: M x M x F_2

def build_graph(A, U, X, Y, epsilon = 0.0):
    
    M = X.shape[0]
    F_1 = X.shape[1]
    F_2 = Y.shape[2]

    V = int((M+1)*(M+2)/2)
    W = np.zeros((V, V, F_2 + 2))
    # (pair features, potential, coulomb)
    
    Z = np.zeros((V, F_1 + F_2 + 3))
    # (singleton features, pair features, potential, single-orbital indicator, duplicate-orbital indicator)

    
    #(i, j) indicates a vertex for orbital pair i-j where i < j
    #(i, M) indicates a vertex for the single orbital i
    
    
    for i in range(M):
        for j in range(i, M+1):
            u = vertex_index(i,j,M+1)
            u_single = (j == M)
            
            if not u_single:
                duplicate = 1 if i == j else 0
                features = [np.zeros(F_1), Y[i,j], A[i,j], 0, duplicate]
            else:
                features = [X[i], np.zeros(F_2), 0, 1, 0]
            Z[u] = concat(features)
                
            for k in range(M):
                for l in range(k, M+1):
                    v = vertex_index(k,l,M+1)
                    v_single = (l == M)

                    if not u_single and not v_single:
                        coulomb = 0 if np.abs(U[i,j,k,l]) < epsilon else U[i,j,k,l]
                        features = [np.zeros(F_2), 0, coulomb]
                    elif u_single and v_single:
                        features = [Y[i,k], A[i,k], 0]
                                                
                    elif u_single and not v_single:
                        if i == k:
                            features = [np.zeros(F_2), 0, 0]
                        elif i == l:
                            features = [np.zeros(F_2), 0, 0]
                        else:
                            features = np.zeros(F_2 + 2)
                    elif not u_single and v_single:
                        if k == i:
                            features = [np.zeros(F_2), 0, 0]
                        elif k == j:
                            features = [np.zeros(F_2), 0, 0]
                        else:
                            features = np.zeros(F_2 + 2)
                    
                    W[u,v] = concat(features)
                    
    
    #TODO: Time optimization + Sparse optimization
    
    x = torch.from_numpy(Z)
    all_edge = np.max(np.abs(W), axis = 2)
    all_edge = np.where(all_edge > 0, 1, 0)
    edge_index, _ = from_scipy_sparse_matrix(sparse.coo_matrix(all_edge))
    
    edge_attr = W[(edge_index[0], edge_index[1])]
    edge_attr = torch.from_numpy(edge_attr)
    
    return x, edge_index, edge_attr
