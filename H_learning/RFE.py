import numpy as np
# from numpy.random import default_rng
from qutip import basis, sigmax, sigmay, sigmaz, sesolve, expect, Qobj
import matplotlib.pyplot as plt

pauli = {1: sigmax(), 2: sigmay(), 3: sigmaz()}
# phi_plus = (basis(2, 0) + basis(2, 1)).unit()

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
def def_phi_plus(s, beta):
    return (single_qubit_eigenstate(s, beta) + single_qubit_eigenstate(1 - s, beta)).unit()

def spectral_gap(H_tot):
    evals = H_tot.eigenenergies()
    evals = np.sort(evals)        
    return evals[1] - evals[0]

def run_shots(state,operator,N= 54,seed = None):
    rng = np.random.default_rng(seed)
    exp_val = expect(operator, state)          
    p_plus   = (1.0 + exp_val) / 2.0 
    # print(exp_val)  
    # print('pplus'+ str(p_plus))      
    # Draw N Bernoulli trials: True = +1, False = −1
    outcomes = rng.random(N) < p_plus          
    # Convert True/False to +1/−1 and take the mean
    # print('mean'+str((2.0 * outcomes.astype(float) - 1.0).mean()))
    return (2.0 * outcomes.astype(float) - 1.0).mean()
    
  

def decide_low_or_high(a, b, mean_x, mean_y):
    z     = mean_x + 1j * mean_y            # complex sample
    phase = np.exp(-1j * (a + b) * np.pi / (2 * (b - a)))
    f     = np.imag(z * phase)              # Lemma 8 test function
    return 1 if f > 0 else 0

temp = []
def robust_gap_estimate(phi_plus,H_tot,O_c,O_s,upper,eps,N_shots,seed: None = None):
    a, b = 0.0, upper
    rng = np.random.default_rng(seed)
    # helper that returns the sign (+1 or −1) of the empirical mean
    def shot_sign(state, op):
        mean = run_shots(state, op, N=N_shots, seed=None)
        # return 1.0 if mean >= 0 else -1.0
        return mean
    while b - a > eps:
        # print(b-a)
        t = np.pi / (b - a)
        U = (-1j * H_tot * t).expm()
        psi_t = U * phi_plus
        # print(expect(O_c, psi_t))
        sgn_X = shot_sign(psi_t, O_c)
        sgn_Y = shot_sign(psi_t, O_s)
        keep = decide_low_or_high(a, b, sgn_X, sgn_Y)
        if keep == 0:               # keep low interval
            b = (a + 2 * b) / 3.0
        else:                       # keep high interval
            a = (2 * a + b) / 3.0
    return 0.5 * (a + b)


H_true = 0.3 * sigmaz() + 0.1 * sigmax() + 0.5*sigmay()
# H_true = 0
nu = 10
Oc_table = {1: sigmaz(), 2: sigmaz(), 3: sigmax()}
Os_table = {1: sigmay(), 2: -sigmax(), 3: -sigmay()}
E_delta_vec = []
E_delta_true =[]
for s1 in [1]:
    for beta in [1,2,3]:
        # print('parameters',s1, beta)
        H_ctrl = 0.5 * s1 * pauli[beta]
        H_tot = H_true - nu * H_ctrl
        if s1 == 0:
            O_c, O_s = sigmax(), sigmay()
        else:
            O_c = Oc_table[beta]
            O_s = Os_table[beta]
        phi_plus = def_phi_plus(s1,beta)
        # phi_plus = def_phi_plus(1,3)
        # print(expect(O_c, phi_plus))  
        # phi_plus = (basis(2, 0) + basis(2, 1)).unit()
        print(expect(O_c,phi_plus))
        print(expect(O_s,phi_plus))
        # print(phi_plus)
        gap_est = robust_gap_estimate(phi_plus,H_tot,O_c,O_s,upper=nu,eps=1e-3,N_shots=540)
        E_delta_vec.append(gap_est)
        E_delta_true.append(float(spectral_gap(H_tot)))

print(E_delta_vec)
print(E_delta_true)
