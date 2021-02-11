import numpy as np
from pyscf import gto, scf, tools, ao2mo
import scipy.linalg

def get_orthoAO(S, LINDEP_CUTOFF=1e-14):
    sdiag, Us = np.linalg.eigh(S)
    X = Us[:,sdiag>LINDEP_CUTOFF] / np.sqrt(sdiag[sdiag>LINDEP_CUTOFF])
    return X


def get_data(mol, orbital_type = "AO", predict_correlation = True):
    
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

#     if predict_correlation:
#         cisolver = fci.FCI(mol, mf.mo_coeff)
#         cisolver.kernel()
#         E_0 = mf.energy_elec()
#         E_0 = E_0[0] + E_0[1]
#         E -= E_0

    if orbital_type == "AO":
        return (hcore, eri_ao, None, np.stack([rdm1_ao, fock_ao], axis = 2), E)
#     elif orbital_type == "OAO":
#         return 
#     elif orbital_type == "MO":
#         return
    else:
        return None
    