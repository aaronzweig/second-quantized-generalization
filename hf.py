import numpy as np
from pyscf import gto, scf, tools, ao2mo
import scipy.linalg
import pickle
import os.path
from os import path

#TODO: Becke quadrature + weights + CVT grid point sampling?
#TODO: Stop cheating, switch to DF approximation

def thc_cheat(mol, mo_coeff, eri_mo, r = 20, grid_points_per_atom = 600, epsilon_qr = 1e-15, epsilon_inv = 1e-15, verbose = False):
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
    mo_value = ao_value.dot(mo_coeff)
    
    ### Fourier projection ###
    
    rho = np.zeros((N * N, n))
    M = np.zeros((N * N, n), dtype=complex)
    for mu in range(n):
        v = np.outer(mo_value[mu], mo_value[mu]).flatten()
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
    X = mo_value[E[:N_aux]].T

    C = rho[:,E[:N_aux]]
    # C = scipy.linalg.khatri_rao(X, X)
    P = np.linalg.inv(R[:N_aux, :N_aux]).dot(R[:N_aux, E_inv])    
    
    B = np.linalg.pinv(C.T.dot(C), rcond = epsilon_inv).dot(C.T)
    T = eri_mo.reshape(N*N, N*N)
    Z = B @ T @ B.T
    
    T_approx = C @ Z @ C.T
    
    if verbose:
        print("rho L2:", np.linalg.norm(rho - C.dot(P)))
        print("T L_infinity:", np.max(np.abs(T - T_approx)))
        print("C-X approx:", np.linalg.norm(C - scipy.linalg.khatri_rao(X, X)))
    
    return X, Z, coords_aux


def get_orthoAO(S, LINDEP_CUTOFF=1e-14):
    sdiag, Us = np.linalg.eigh(S)
    X = Us[:,sdiag>LINDEP_CUTOFF] / np.sqrt(sdiag[sdiag>LINDEP_CUTOFF])
    return X

def save_data(mols, filename, force = False, kwargs = None):
    
    if path.exists(filename):
        if not force:
            return
        else:
            os.remove(filename)
    
    data = []
    
    for mol in mols:
        
        mol.build()    
        mf = scf.RHF(mol)
        na, nb = mol.nelec
        mf.kernel()
        
        etotal = mf.energy_elec()
        etotal = etotal[0] + etotal[1]

        # get AO integrals
        nbsf = mol.nao_nr() # the number of AOs
        hcore = mf.get_hcore() # one-body in AO
        eri_ao = mol.intor('int2e', aosym='s1').reshape([nbsf]*4) # two-body in AO
        rdm1_ao = mf.make_rdm1() # 1rdm in AO
        fock_ao = mf.get_fock()

        # get MO integrals
        S = mol.intor('int1e_ovlp') # AO metric
        orb = mf.mo_coeff
        hmo = mf.mo_coeff.T.dot(hcore).dot(mf.mo_coeff) # one-body in MO
        eri_mo = ao2mo.kernel(mol, orb, compact=False).reshape([nbsf]*4) # two-body in MO
        rdm1_mo = mf.mo_coeff.T.dot(S).dot(rdm1_ao).dot(S).dot(mf.mo_coeff) # 1rdm in MO
        
        
        #TODO: boilerplate from PySCF written for UHF; is this correct?  How do we even check

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        mo_coeff = mf.mo_coeff
        o = mo_coeff[:,mo_occ>0]
        v = mo_coeff[:,mo_occ==0]
        eo = mo_energy[mo_occ>0]
        ev = mo_energy[mo_occ==0]
        no = o.shape[1]
        nv = v.shape[1]
        noa = sum(mo_occ>0)
        nva = sum(mo_occ==0)
        eri = ao2mo.general(mf.mol, (o,v,o,v)).reshape(no,nv,no,nv)
        eri[:noa,nva:] = eri[noa:,:nva] = eri[:,:,:noa,nva:] = eri[:,:,noa:,:nva] = 0
        g = eri - eri.transpose(0,3,2,1)
        eov = eo.reshape(-1,1) - ev.reshape(-1)
        de = 1/(eov.reshape(-1,1) + eov.reshape(-1)).reshape(g.shape)
        
        E = np.einsum('iajb,iajb,iajb->ia', g, g, de)
        emp2 = .25 * np.einsum('iajb,iajb,iajb->', g, g, de)
        
        
        ###THC with cheating###
        X, Z, coords_aux = thc_cheat(mol, mf.mo_coeff, eri_mo, **kwargs)
        
        dump = (hmo, eri_mo, rdm1_mo, mo_occ, mo_energy, X, Z, coords_aux, E, etotal, emp2)
        data.append(dump)
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    

def load_data(filename):
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    new_data = []
    
    for dump in data:
        hmo, eri_mo, rdm1_mo, mo_occ, mo_energy, X, Z, coords_aux, E, etotal, emp2 = dump
        
        M = hmo.shape[0]
        N_aux = coords_aux.shape[0]
        
        F1 = np.stack([mo_energy, mo_occ], axis = 1)
        F2 = np.stack([hmo, rdm1_mo, np.eye(M)], axis = 2)
        F3 = np.zeros((N_aux, N_aux, 2))
        for i in range(N_aux):
            for j in range(N_aux):
                F3[i,j,0] = np.linalg.norm(coords_aux[i] - coords_aux[j])
        F3[:,:,1] = np.eye(N_aux)
        
        new_dump = (X, Z, F1, F2, F3, eri_mo, E, emp2)
        new_data.append(new_dump)
    return new_data



#########################################################################################################


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
    