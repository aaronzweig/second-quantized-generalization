import numpy as np
import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy import sparse
from scipy.io import loadmat
from pyscf import gto, scf, tools, ao2mo
from torch_geometric.data import Data, DataLoader

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
        mol.unit = "Bohr" #QM7 coordinate unit

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
#n: number of grid points

#X is THC matrix: M x n
#Z is THC matrix: n x n

#F1 is additional orbital feature matrix: M x L1
#F2 is additional pairwise orbital feature matrix: M x M x L2
#F3 is additional pairwise grid point feature matrix: n x n x L3

#Vertex type:
#Type 0 = v_ij (two MOs)
#Type 1 = v_iP (one MO, one grid point)
#Type 2 = v_PQ (two grid points)


#E is pair correlation energies: N x (M - N)
#MP2 is total correlation energy

#NOTE: Assume the first N orbitals are the occupied ones
#TODO: Sparisfy the edge building
#TODO: Make sure same-index features (i.e. identity) are included in F2 and F3
#NOTE: Careful about getting X and Z masks, and reshaping them out of the indices at the end

def build_thc_graph(X, Z, F1, F2, F3, T, E, MP2):
    
    M = X.shape[0]
    N = E.shape[0]
    n = Z.shape[0]
    
    L1 = F1.shape[1]
    L2 = F2.shape[2]
    L3 = F3.shape[2]
    
    def thc_vertex_index(i, j, typ):
        if typ == 0:
            i, j = min(i,j), max(i,j)
            index = vertex_index(i, j, M)
        elif typ == 1:
            P = j
            index = M*(M+1)/2 + i * n + P
        elif typ == 2:
            P, Q = min(i,j), max(i,j)
            index = M*(M+1)/2 + M * n + vertex_index(P, Q, n)
        else:
            index = -999
        return int(index)
        
### TEST
#     for i in range(M):
#         for j in range(i,M):
#             v = thc_vertex_index(i,j,0)
#             print(v)

#     for i in range(M):
#         for P in range(n):
#             v = thc_vertex_index(i,P,1)
#             print(v)

#     for P in range(n):
#         for Q in range(P,n):
#             v = thc_vertex_index(Q, P, 2)
#             print(v)
    
    V_F = L1 + L1 + L2 + L3 + 3 + 1 + 1
    # F1_i, F1_j, F2, F3, type, X, Z
    E_F = L2 + L3 + 1 + 1
    # F2, F3, X, Z
    
    
    V = int(M*(M+1)/2 + M * n + n*(n+1)/2)
    
    G_V = np.zeros((V, V_F))
    G_E = np.zeros((V, V, E_F))
    
    X_mask = np.zeros(V, dtype=bool)
    Z_mask = np.zeros(V, dtype=bool)
    
    ### build vertices ###
    
    typ_arr = np.array([1, 0, 0])
    for i in range(M):
        for j in range(i,M):
            v = thc_vertex_index(i, j, 0)
            features = [F1[i], F1[j], F2[i,j], np.zeros(L3), typ_arr, 0, 0]
            G_V[v] = concat(features)
    
    typ_arr = np.array([0, 1, 0])
    for i in range(M):
        for P in range(n):
            v = thc_vertex_index(i, P, 1)
            features = [F1[i], np.zeros(L1), np.zeros(L2), np.zeros(L3), typ_arr, X[i,P], 0]
            G_V[v] = concat(features)
            X_mask[v] = True
    
    typ_arr = np.array([0, 0, 1])
    for P in range(n):
        for Q in range(P,n):
            v = thc_vertex_index(P, Q, 2)
            features = [np.zeros(L1), np.zeros(L1), np.zeros(L2), F3[P,Q], typ_arr, 0, Z[P,Q]]
            G_V[v] = concat(features)
            Z_mask[v] = True

            
            
    ### build edges ###
    #TODO: Probably could squeeze some factors of 2 by not being redundant with edges?
    
    for i in range(M):
        for j in range(M):
            v = thc_vertex_index(i, j, 0)
            for k in range(M):
                #(i,j) - (j,k)
                u = thc_vertex_index(j, k, 0)
                features = [F2[i,k], np.zeros(L3), 0, 0]
                G_E[u,v] = G_E[v,u] = concat(features)
            for P in range(n):
                #(i,j) - (j,P)
                u = thc_vertex_index(j, P, 1)
                features = [np.zeros(L2), np.zeros(L3), X[i,P], 0]
                G_E[u,v] = G_E[v,u] = concat(features)

    for P in range(n):
        for Q in range(n):
            v = thc_vertex_index(P, Q, 2)
            for R in range(n):
                #(P,Q) - (Q,R)
                u = thc_vertex_index(Q, R, 2)
                features = [np.zeros(L2), F3[P,R], 0, Z[P,R]]
                G_E[u,v] = G_E[v,u] = concat(features)
            for i in range(M):
                #(P,Q) - (i,Q)
                u = thc_vertex_index(i, Q, 1)
                features = [np.zeros(L2), np.zeros(L3), X[i,P], 0]
                G_E[u,v] = G_E[v,u] = concat(features)
                
    for P in range(n):
        for i in range(M):
            v = thc_vertex_index(i, P, 1)
            for Q in range(n):
                #(i,P) - (P,Q)
                u = thc_vertex_index(i, Q, 1)
                features = [np.zeros(L2), F3[P,Q], 0, Z[P,Q]]
                G_E[u,v] = G_E[v,u] = concat(features)
            for j in range(M):
                #(i,P) - (j, P)
                u = thc_vertex_index(j, P, 1)
                features = [F2[i,j], np.zeros(L3), 0, 0]
                G_E[u,v] = G_E[v,u] = concat(features)


                
                
    x = torch.from_numpy(G_V)
    all_edge = np.max(np.abs(G_E), axis = 2)
    all_edge = np.where(all_edge > 0, 1, 0)
    edge_index, _ = from_scipy_sparse_matrix(sparse.coo_matrix(all_edge))
    
    edge_attr = G_E[(edge_index[0], edge_index[1])]
    edge_attr = torch.from_numpy(edge_attr)
    
    data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, MP2 = MP2, E = E, X = torch.from_numpy(X),
                Z = torch.from_numpy(Z), X_mask = X_mask, Z_mask = Z_mask, T = torch.from_numpy(T))
    
    return data