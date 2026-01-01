import numpy as np
from qutip import basis, sigmax, sigmay, sigmaz,  expect
import matplotlib.pyplot as plt

pauli = {1: sigmax(), 2: sigmay(), 3: sigmaz()}
Oc_table = {1: sigmaz(), 2: sigmaz(), 3: sigmax()}
Os_table = {1: sigmay(), 2: -sigmax(), 3: -sigmay()}

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

def run_shots(state,operator,N= 54,seed = None,p=0.05):
    rng = np.random.default_rng(seed)
    exp_val = expect(operator, state)          
    p_plus   = (1.0 + exp_val) / 2.0    
    # Draw N Bernoulli trials: True = +1, False = −1
    outcomes = rng.random(N) < p_plus          
    # Convert True/False to +1/−1 and take the mean
    results = 2.0 * outcomes.astype(float) - 1.0
    
    # apply noise: flip sign with probability p
    noise_mask = rng.random(N) < p
    results[noise_mask] *= -1
    
    
    return results.mean()
    
def decide_low_or_high(a, b, mean_x, mean_y):
    z     = mean_x + 1j * mean_y            # complex sample
    phase = np.exp(-1j * (a + b) * np.pi / (2 * (b - a)))
    f     = np.imag(z * phase)              # Lemma 8 test function
    return 1 if f > 0 else 0

def robust_gap_estimate(phi_plus,H_tot,O_c,O_s,upper,eps,N_shots):
    a, b = 0.0, upper
    while b - a > eps:
        t = np.pi / (b - a)
        U = (-1j * H_tot * t).expm()
        psi_t = U * phi_plus
        X = run_shots(psi_t, O_c, N=N_shots, seed=None)
        Y = run_shots(psi_t, O_s, N=N_shots, seed=None)
        keep = decide_low_or_high(a, b, X, Y)
        if keep == 0:               # keep low interval
            b = (a + 2 * b) / 3.0
        else:                       # keep high interval
            a = (2 * a + b) / 3.0
    return 0.5 * (a + b)

def delta_E_RFE(lambda_1, lambda_2, lambda_3, nu, beta):
    H_true = lambda_1 * sigmaz() + lambda_2 * sigmax() + lambda_3*sigmay()
    H_ctrl = 0.5 * pauli[beta]
    H_tot = H_true - nu * H_ctrl
    O_c = Oc_table[beta]
    O_s = Os_table[beta]
    phi_plus = def_phi_plus(1, beta)
    gap_est = robust_gap_estimate(phi_plus,H_tot,O_c,O_s,upper=nu,eps=1e-3,N_shots=10)
    return gap_est


if __name__ == "__main__":
    H_true = 0.3 * sigmaz() + 0.1 * sigmax() + 0.5*sigmay()
    nu = 10
    E_delta_vec = []
    E_delta_true =[]
    for s1 in [1]:
        for beta in [1,2,3]:
            H_ctrl = 0.5 * (2*s1-1) * pauli[beta]
            H_tot = H_true - nu * H_ctrl
            if s1 == 0:
                O_c, O_s = sigmax(), sigmay()
            else:
                O_c = Oc_table[beta]
                O_s = Os_table[beta]
            phi_plus = def_phi_plus(s1,beta)
            gap_est = robust_gap_estimate(phi_plus,H_tot,O_c,O_s,upper=nu,eps=1e-3,N_shots=540)
            E_delta_vec.append(gap_est)
            E_delta_true.append(float(spectral_gap(H_tot)))

    print(E_delta_vec)
    print(E_delta_true)
