import numpy as np
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis
from RFE import robust_gap_estimate

pauli = {1: sigmax(), 2: sigmay(), 3: sigmaz()}
_P = {'I': qeye(2), 'X': sigmax(), 'Y': sigmay(), 'Z': sigmaz()}
ORDER = ["XI","YI","ZI","IX","IY","IZ",
         "XX","XY","XZ","YX","YY","YZ","ZX","ZY","ZZ"]
# for s=1 only!
Oc_table = {1: sigmaz(), 2: sigmaz(), 3: sigmax()}
Os_table = {1: sigmay(), 2: -sigmax(), 3: -sigmay()}
# (k, beta)
Oc_table_2q = {(0,1): tensor(sigmaz(),qeye(2)), (1,1): tensor(qeye(2),sigmaz()), (0,2): tensor(sigmaz(),qeye(2)),(1,2): tensor(qeye(2),sigmaz()), (0,3): tensor(sigmax(),qeye(2)), (1,3):tensor(qeye(2), sigmax())}
Os_table_2q = {(0,1): tensor(sigmay(),qeye(2)), (1,1): tensor(qeye(2),sigmay()), (0,2): tensor(-sigmax(),qeye(2)), (1,2): tensor(qeye(2),-sigmax()), (0,3): tensor(-sigmay(),qeye(2)), (1,3): tensor(qeye(2),-sigmay())}

def spectral_gap(H_tot):
    evals = H_tot.eigenenergies()
    evals = np.sort(evals)       
    return evals[1] - evals[0]

# Coeffs are lambdas in the paper
def H_0(coeffs, order=ORDER):
    H = 0
    for c, lab in zip(coeffs, order):
        if c != 0:
            H += c * tensor(_P[lab[0]], _P[lab[1]])
    if isinstance(H, int): # for case with all zero coeffs
        H = 0 * tensor(qeye(2), qeye(2))
    return H


def single_qubit_eigenstate(s, beta):
    if beta == 1: 
        ket_plus  = (basis(2, 0) + basis(2, 1)).unit()
        ket_minus = (basis(2, 0) - basis(2, 1)).unit()
        return ket_plus if s == 0 else ket_minus
    if beta == 2:  
        ket_plus_i  = (basis(2, 0) + 1j * basis(2, 1)).unit()
        ket_minus_i = (basis(2, 0) - 1j * basis(2, 1)).unit()
        return ket_plus_i if s == 0 else ket_minus_i
    if beta == 3:  
        return basis(2, s)   
    
def check(): # sanity check for single_qubit_eigenstate, passed
    for beta in (1,2,3):
        for s in (0,1):
            ket = single_qubit_eigenstate(s, beta)
            ev  = (-1)**s
            err = (pauli[beta]*ket - ev*ket).norm()
            print(f"beta={beta}, s={s}, residual={float(err):.3e}")    

def two_qubit_eigenstate(s1, s2,beta1, beta2):
    return tensor(single_qubit_eigenstate(s1, beta1),
                  single_qubit_eigenstate(s2, beta2)).unit()
    
def def_phi_plus_2q(k,s1, s2, beta1, beta2):
    if k==0:
        return (two_qubit_eigenstate(s1, s2, beta1, beta2)+two_qubit_eigenstate(1-s1, s2, beta1, beta2)).unit()
    else:
        return (two_qubit_eigenstate(s1, 1-s2, beta1, beta2)+two_qubit_eigenstate(s1,  s2,beta1, beta2)).unit()

def on_qubit(op, k, n):
    #Put single-qubit operator on qubit j, I elsewhere.
    ops = [qeye(2)] * n
    ops[k] = op
    return tensor(ops)

def H_ctrl_func(k, s1, s2, beta_1, beta_2):
    """
    Observe that k=0, H_ctrl = 1/2*s_1*pauli(beta_1) x I + s_2*I x pauli(beta_2)
    Similarly, if k=1, H_ctrl = s_1*pauli(beta_1) x I + 1/2*s_2* I x pauli(beta_2) 
    """
    H_ctrl = 0
    coeff1 = 0.5 if k == 0 else 1.0
    H_ctrl += coeff1 *(-1)**(s1)* on_qubit(pauli[beta_1], 0, 2)
    coeff2 = 0.5 if k == 1 else 1.0
    H_ctrl += coeff2 *(-1)**(s2)* on_qubit(pauli[beta_2], 1, 2)
    if isinstance(H_ctrl, int): # for case with all zero coeffs
        H_ctrl = 0 * tensor(qeye(2), qeye(2))
    return H_ctrl


def check_conditions(Oc, Os, phi0, phi1):
    # print(Os*phi0,1j*phi1)
    return ((Oc*phi0-phi1).norm(), (Oc*phi1-phi0).norm(), (Os*phi0+1j*phi1).norm(), (Os*phi1-1j*phi0).norm())

def Oc_Os_decider(k,s_1,s_2,beta_1,beta_2):
    if k==0:
        beta = beta_1
        if s_1 == 1:
            O_c = Oc_table_2q[(k,beta)]
            O_s = -Os_table_2q[(k,beta)]
        else:
            O_c = Oc_table_2q[(k,beta)]
            O_s = Os_table_2q[(k,beta)]
    else:
        beta = beta_2
        if s_2==1:
            O_c = Oc_table_2q[(k,beta)]
            O_s = -Os_table_2q[(k,beta)]
        else:
            O_c = Oc_table_2q[(k,beta)]
            O_s = Os_table_2q[(k,beta)]
    return O_c, O_s

# phi_0 = single_qubit_eigenstate(1,3)
# phi_1 = single_qubit_eigenstate(1,3)
# O_c = Oc_table[3]
# O_s = Oc_table[3]
# print(check_conditions(O_c, O_s, phi_0, phi_1))

if __name__ == "__main__":
    lambda_vec = [0.1,0.2,0.3, 0.5,0.6,0.3,0.2,0.1,0.1, 0.2,0.1,0.1, 0.3,0.22,0.15] #Sanity checked
    H_true = H_0(coeffs=lambda_vec) #Sanity checked
    # H_true = 0      
    for k in [0,1]:
        for s1 in [0,1]:
            for s2 in [0,1]:
                for beta_1 in [1,2,3]:
                    for beta_2 in [1,2,3]:
                        nu=10
                        H_ctrl = H_ctrl_func(k,s1,s2,beta_1,beta_2) 
                        H_tot = H_true - nu * H_ctrl
                        if k==0:
                            beta = beta_1
                        else:
                            beta = beta_2
                        O_c, O_s = Oc_Os_decider(k, s1, s2, beta_1, beta_2)   
                        if k==0:
                            phi_0 = two_qubit_eigenstate(s1, s2, beta_1, beta_2)
                            phi_1 = two_qubit_eigenstate(1-s1, s2, beta_1, beta_2)     
                        else:
                            phi_0 = two_qubit_eigenstate(s1, s2, beta_1, beta_2)
                            phi_1 = two_qubit_eigenstate(s1, 1-s2, beta_1, beta_2)         
                        # phi_plus = def_phi_plus_2q(k,s1,s2, beta_1, beta_2)
                        phi_plus = (phi_0 + phi_1).unit()
                        # print(phi_0)
                        gap_est = robust_gap_estimate(phi_plus,H_tot,O_c,O_s,upper=nu+2.0 * H_true.norm() ,eps=1e-4,N_shots=20)
                        gap_true = spectral_gap(H_tot)
                        if np.abs(gap_true-gap_est)>10e-4:
                            print('--------------------')
                            print(k,s1,s2,beta_1, beta_2)
                            print('energy: ' + str(phi_0.dag()*H_tot*phi_0))
                            print('Oc+Os conditions: '+str(check_conditions(O_c, O_s, phi_0, phi_1)))
                            
                            print(gap_est,gap_true)
                            print('diff: '+str(np.abs(gap_true-gap_est)))