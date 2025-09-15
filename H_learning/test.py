import numpy as np
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis

pauli = {1: sigmax(), 2: sigmay(), 3: sigmaz()}
_P = {'I': qeye(2), 'X': sigmax(), 'Y': sigmay(), 'Z': sigmaz()}
ORDER = ["II","XI","YI","ZI","IX","IY","IZ",
         "XX","XY","XZ","YX","YY","YZ","ZX","ZY","ZZ"]


# Coeffs are lambdas in the paper
def H_0(coeffs, order=ORDER):
    coeffs = np.asarray(coeffs).reshape(-1)
    if coeffs.size != 16:
        raise ValueError("Need 16 coefficients for two-qubit Pauli basis.")
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
    
def two_qubit_eigenstate(s1, beta1, s2, beta2):
    return tensor(single_qubit_eigenstate(s1, beta1),
                  single_qubit_eigenstate(s2, beta2))

def on_qubit(op, k, n):
    #Put single-qubit operator on qubit j, I elsewhere.
    ops = [qeye(2)] * n
    ops[k] = op
    return tensor(ops)

def H_ctrl(k, s1, s2, beta_1, beta_2):
    """
    Observe that k=0, H_ctrl = 1/2*s_1*pauli(beta_1) x I + s_2*I x pauli(beta_2)
    Similarly, if k=1, H_ctrl = s_1*pauli(beta_1) x I + 1/2*s_2* I x pauli(beta_2) 
    """
    H_ctrl = 0
    if s1:
        coeff1 = 0.5 if k == 0 else 1.0
        H_ctrl += coeff1 * on_qubit(pauli[beta_1], 0, 2)
    if s2:
        coeff2 = 0.5 if k == 1 else 1.0
        H_ctrl += coeff2 * on_qubit(pauli[beta_2], 1, 2)
    if isinstance(H_ctrl, int): # for case with all zero coeffs
        H_ctrl = 0 * tensor(qeye(2), qeye(2))
    return H_ctrl


if __name__ == "__main__":
    vec = [1.0, 0,0,0, 0,0,0, 0.2,0,0, 0,0,0, 0,0,0] #Sanity checked
    H_true = H_0(vec) #Sanity checked
    H_ctrl(0,1,1,1,1) #Sanity checked
    
    