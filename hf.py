import numpy as np
from pyscf import gto, scf, tools, ao2mo, mp
import scipy.linalg
import pickle
import os.path
from os import path
from collections import namedtuple
from thc import THCContainer



def thc_cheat(mol, mo_coeff, eri_ao, eri_mo, r = 20, grid_points_per_atom = 600, 
              epsilon_qr = 1e-15, epsilon_inv = 1e-15, verbose = False):
    N = mol.nao_nr()
    n = grid_points_per_atom * len(mol.atom)
    r = min(N, r)
    
    ### grid evaluation ###
    
    coords = []
    for atom in mol.atom:
        pos = np.array(atom[1])
        local_coords = pos + np.random.normal(scale = 3.0, size = (grid_points_per_atom, 3))
        coords.append(local_coords)
    coords = np.concatenate(coords, axis = 0)
        
    ao_value = mol.eval_gto("GTOval_sph", coords)
    
    ### Fourier projection ###
    
    rho = np.zeros((N * N, n))
    M = np.zeros((N * N, n), dtype=complex)
    for mu in range(n):
        v = np.outer(ao_value[mu], ao_value[mu]).flatten()
        rho[:,mu] = v
        M[:,mu] = np.fft.fft(v)

    rows = np.random.choice(N * N, N * r, replace = False)
    M = M[rows]
    
    ### QR pivot and threshold ###
    
    Q, R, E = scipy.linalg.qr(M, pivoting = True)
    # print(np.linalg.norm(M[:,E] - Q.dot(R)))

    E_inv = np.zeros(E.size, dtype=np.int32)
    for i in np.arange(E_inv.size):
        E_inv[E[i]] = i

    d = np.abs(R.diagonal()) / np.abs(R[0,0])
    N_aux = np.count_nonzero(d > epsilon_qr)

    
    ### Restrict to selected grid points and compute approximation by cheating
    
    coords_aux = coords[E[:N_aux]]
    X = ao_value[E[:N_aux]].T
    
    C = rho[:,E[:N_aux]]
    P = np.linalg.inv(R[:N_aux, :N_aux]).dot(R[:N_aux, E_inv])    
    
    B = np.linalg.pinv(C.T.dot(C), rcond = epsilon_inv).dot(C.T)
    T_ao = eri_ao.reshape(N*N, N*N)
    T_mo = eri_mo.reshape(N*N, N*N)

    Z = B @ T_ao @ B.T
    
    # T_ao \approx C @ Z @ C.T
    
    X_mo = mo_coeff.T.dot(X)
    
    C_mo = khatri_rao(X_mo, X_mo)
    T_approx = C_mo @ Z @ C_mo.T
    
    
    
        
    if verbose:
        print("rho L2:", np.linalg.norm(rho - C.dot(P)))
        print("Sanity C as khatri-rao product:",
              np.linalg.norm(C - np.einsum("ij,kj->ikj",X,X).reshape((X.shape[0] ** 2, -1))))
        print("T_ao L_infinity:", np.max(np.abs(T_ao - C @ Z @ C.T)))
        print("T_mo L_infinity:", np.max(np.abs(T_mo - T_approx)))
    
    return X, Z, coords_aux, T_approx

def thc_to_eri_torch(X, U, T_ao, epsilon = 1e-15, Y = None):
    if Y is None:
        Y = X
    C = torch_khatri_rao(X, Y)
    B = torch.pinverse(C.T @ C, rcond = epsilon) @ C.T
    Z = B @ T_ao @ B.T
    
    X_mo = U.T @ X
    Y_mo = U.T @ Y
    C_mo = torch_khatri_rao(X_mo, Y_mo)
    T_approx = C_mo @ Z @ C_mo.T
    return T_approx
    
    
#     with open(filename, 'wb') as f:
#         pickle.dump(data, f)
    

# def load_data(filename):
    
#     with open(filename, 'rb') as f:
#         data = pickle.load(f)
    
#     return data



#########################################################################################################

def get_orthoAO(S, LINDEP_CUTOFF=1e-14):
    sdiag, Us = np.linalg.eigh(S)
    X = Us[:,sdiag>LINDEP_CUTOFF] / np.sqrt(sdiag[sdiag>LINDEP_CUTOFF])
    return X


def get_data(mol, orbital_type = "AO", predict_correlation_pairs = True):
    
    mol.build()    
    mf = scf.RHF(mol)
    na, nb = mol.nelec
    mf.kernel()
    
    E = mf.energy_elec()
    E = E[0] + E[1]
    
    # get AO integrals
    nbsf = mol.nao_nr() # the number of AOs
    hcore = mf.get_hcore() # one-body in AO
    eri_ao = mol.intor('int2e', aosym='s1').reshape([nbsf]*4) # two-body in AO
    rdm1_ao = mf.make_rdm1() # 1rdm in AO
    fock_ao = mf.get_fock()
    
#     # get OAO integrals
#     S = mol.intor('int1e_ovlp') # AO metric
#     X = get_orthoAO(S, 1e-6)
#     Xinv = np.linalg.inv(X)
#     hoao = X.T.dot(hcore).dot(X) # one-body in OAO
#     eri_oao = np.einsum("ip,jq,kr,ls,ijkl->pqrs",X,X,X,X,eri_ao,optimize=True) # one-body in OAO
#     rdm1_oao = Xinv.dot(rdm1_ao).dot(Xinv.T) # 1rdm in OAO, trace (rdm1_oao) = na + nb
    
#     # get MO integrals
#     orb = mf.mo_coeff
#     hmo = mf.mo_coeff.T.dot(hcore).dot(mf.mo_coeff) # one-body in MO
#     eri_mo = ao2mo.kernel(mol, orb, compact=False).reshape([nbsf]*4) # two-body in MO
#     rdm1_mo = mf.mo_coeff.T.dot(S).dot(rdm1_ao).dot(S).dot(mf.mo_coeff) # 1rdm in MO
    
#     print("Nelec check: {}, {}, {}, {}".format(rdm1_ao.dot(S).trace(), rdm1_oao.trace(), rdm1_mo.trace(), na+nb))

    if orbital_type == "AO":
        return (hcore, eri_ao, None, np.stack([rdm1_ao, fock_ao], axis = 2), E)
#     elif orbital_type == "OAO":
#         return 
#     elif orbital_type == "MO":
#         return (hmo, eri_mo, None, np.stack([rdm1_mo, fock_ao], axis = 2), E)
        #TODO: Fock matrix returned here?
    else:
        return None
    