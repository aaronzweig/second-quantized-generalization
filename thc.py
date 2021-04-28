import numpy as np
import scipy.linalg
from pyscf import gto, scf, tools, ao2mo, mp
from utils import khatri_rao

def get_mp2(eri, mo_energy, mo_occ):
    
    eo = mo_energy[mo_occ>0]
    ev = mo_energy[mo_occ==0]
    
    eov = eo.reshape(-1,1) - ev.reshape(-1)
    de = 1/(eov.reshape(-1,1) + eov.reshape(-1)).reshape(eri.shape)
    
    E_J = 2 * np.einsum('iajb,iajb,iajb->ia', eri, eri, de)
    E_K = -1 * np.einsum('iajb,ibja,iajb->ia', eri, eri, de)
    E = E_J + E_K
    
    MP2_J = 2 * np.einsum('iajb,iajb,iajb->', eri, eri, de)
    MP2_K = -1 * np.einsum('iajb,ibja,iajb->', eri, eri, de)
    MP2 = MP2_J + MP2_K
    
    return (MP2_J, MP2_K, MP2), (E_J, E_K, E)

#TODO: Becke quadrature + weights + CVT grid point sampling?
#TODO: Stop cheating, switch to DF approximation

def build_thc(mol, mo_coeff, eri_ao, eri_mo, r = 20, grid_points_per_atom = 600, 
                  epsilon_qr = 1e-15, epsilon_inv = 1e-15, verbose = False, scale = 3.0, z_max = -1.0):
        N = mol.nao_nr()
        n = grid_points_per_atom * len(mol.atom)
        r = min(N, r)

        ### grid evaluation ###

        coords = []
        for atom in mol.atom:
            pos = np.array(atom[1])
            local_coords = pos + np.random.normal(scale = scale, size = (grid_points_per_atom, 3))
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
            print("T_ao L_infinity:", np.max(np.abs(T_ao - C @ Z @ C.T)))
            print("T_mo L_infinity:", np.max(np.abs(T_mo - T_approx)))
            print("T_mo L_2:", np.linalg.norm(T_mo - T_approx))

        if z_max > 0:
            c = np.max(np.abs(Z)) / z_max
            Z /= c
            X *= c ** 0.25
            X_mo *= c ** 0.25

        return X, X_mo, Z, coords_aux, T_approx

    
    
    
class THCContainer():
    def __init__(self, mol, kwargs):
        
        mol.build()
        mf = scf.RHF(mol)
        mf.kernel()
        
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
        
        # Calculate energies + THC

        self.MP2, self.E = get_mp2(eri, mo_energy, mo_occ)
        
#         mp_obj = mp.mp2.RMP2(mf)
#         MP2_b = mp_obj.kernel(mo_energy = mo_energy, mo_coeff = mo_coeff)[0]
        
        self.X, self.X_mo, self.Z, self.coords, T_THC = build_thc(mol, mo_coeff, eri_ao, eri_mo, **kwargs)
        
        M = mol.nao_nr()
        eri_THC = T_THC.reshape((M, M, M, M))
        eri_THC = eri_THC[mo_occ>0][:,mo_occ==0][:,:,mo_occ>0][:,:,:,mo_occ==0]
        
        self.MP2_THC, self.E_THC = get_mp2(eri_THC, mo_energy, mo_occ)
        
        # save features
#         self.mol = mol
        self.hmo = hmo
        self.rdm1_mo = rdm1_mo
        self.S = S
        self.mo_energy = mo_energy
        self.mo_occ = mo_occ
        self.mo_coeff = mo_coeff
        self.N = np.count_nonzero(mo_occ)
        self.M = mol.nao_nr()
        self.N_aux = self.coords.shape[0]
        
    def get_features(self, tau = 1.0):

        F1 = np.stack([self.mo_energy, self.mo_occ], axis = 1)
        
        E_THC_J_full = np.zeros((self.M, self.M))
        E_THC_J_full[:self.N, self.N:] = self.E_THC[0]
        E_THC_J_full[self.N:, :self.N] = self.E_THC[0].T
        F2 = np.stack([self.hmo, self.rdm1_mo, E_THC_J_full, np.eye(self.M)], axis = 2)
        
        F3 = np.zeros((self.N_aux, self.N_aux, 2))
        for i in range(self.N_aux):
            for j in range(self.N_aux):
                F3[i,j,0] = np.exp(-1.0 * tau * np.linalg.norm(self.coords[i] - self.coords[j]))
        F3[:,:,1] = np.eye(self.N_aux)

        return F1, F2, F3
    
