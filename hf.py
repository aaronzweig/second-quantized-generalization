import numpy as np
from pyscf import gto, scf, tools, ao2mo
import scipy.linalg
import pickle
import os.path
from os import path


def get_orthoAO(S, LINDEP_CUTOFF=1e-14):
    sdiag, Us = np.linalg.eigh(S)
    X = Us[:,sdiag>LINDEP_CUTOFF] / np.sqrt(sdiag[sdiag>LINDEP_CUTOFF])
    return X

def save_data(mols, filename, force = False):
    
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
        
        E = mf.energy_elec()
        E = E[0] + E[1]

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
        
        pair = np.einsum('iajb,iajb,iajb->ij', g, g, de)
                
        #TODO: Need MO fock matrix?  And overlap
        
        emp2 = .25 * np.einsum('iajb,iajb,iajb->', g, g, de)
        
        dump = (hcore, eri_ao, rdm1_ao, fock_ao, hmo, eri_mo, rdm1_mo, mo_occ, mo_energy, pair, E, emp2)
        data.append(dump)
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    

def load_data(filename, orbital_type = "AO"):
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    new_data = []
    
    for dump in data:
        hcore, eri_ao, rdm1_ao, fock_ao, hmo, eri_mo, rdm1_mo, mo_occ, mo_energy, pair, E, emp2 = dump
        
        if orbital_type == "AO":
            new_dump = (hcore, eri_ao, None, np.stack([rdm1_ao, fock_ao], axis = 2), E)
        else:
            mo_occ[mo_occ>0] = 1
            double_occupied = np.outer(mo_occ, mo_occ)
            new_dump = (hmo, eri_mo, np.stack([mo_occ, mo_energy], axis = 1), np.stack([rdm1_mo, fock_ao, double_occupied], axis = 2), pair, emp2, mo_occ)
        new_data.append(new_dump)
    return new_data






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
    