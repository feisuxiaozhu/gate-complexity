import numpy as np
# from numpy.random import default_rng
from qutip import basis, sigmax, sigmay, sigmaz, sesolve, expect, Qobj

pauli = {1: sigmax(), 2: sigmay(), 3: sigmaz()}
phi_plus = (basis(2, 0) + basis(2, 1)).unit()

def spectral_gap(H_tot):
    evals = H_tot.eigenenergies()
    evals = np.sort(evals)        
    return evals[1] - evals[0]

def run_shots(state,operator,N= 54,seed = None):
    rng = np.random.default_rng(seed)
    exp_val = expect(operator, state)          
    p_plus   = (1.0 + exp_val) / 2.0         
    # Draw N Bernoulli trials: True = +1, False = −1
    outcomes = rng.random(N) < p_plus          
    # Convert True/False to +1/−1 and take the mean
    return (2.0 * outcomes.astype(float) - 1.0).mean()

def decide_low_or_high(a, b, mean_x, mean_y):
    z     = mean_x + 1j * mean_y            # complex sample
    phase = np.exp(-1j * (a + b) * np.pi / (2 * (b - a)))
    f     = np.imag(z * phase)              # Lemma 8 test function
    return 1 if f > 0 else 0

def robust_gap_estimate(H_tot,O_c,O_s,upper,eps,N_shots,seed: None = None):
    a, b = 0.0, upper
    rng = np.random.default_rng(seed)
    # helper that returns the sign (+1 or −1) of the empirical mean
    def shot_sign(state, op):
        mean = run_shots(state, op, N=N_shots, seed=rng.integers(1 << 32))
        return 1.0 if mean >= 0 else -1.0
    while b - a > eps:
        print(b-a)
        t = np.pi / (b - a)
        U = (-1j * H_tot * t).expm()
        psi_t = U * phi_plus
        sgn_X = shot_sign(psi_t, O_c)
        sgn_Y = shot_sign(psi_t, O_s)
        keep = decide_low_or_high(a, b, sgn_X, sgn_Y)
        if keep == 0:               # keep low interval
            b = (a + 2 * b) / 3.0
        else:                       # keep high interval
            a = (2 * a + b) / 3.0
    return 0.5 * (a + b)


H_true = 0.3 * sigmaz() + 0.1 * sigmax()
nu = 1.0
Oc_table = {1: sigmaz(), 2: sigmax(), 3: sigmax()}
Os_table = {1: sigmay(), 2: sigmaz(), 3: sigmay()}
E_delta_vec = []
E_delta_true =[]
for s1 in (0,1):
    for beta in (1, 2, 3):
        print('parameters',s1, beta)
        H_ctrl = 0.5 * s1 * pauli[beta]
        H_tot = H_true - nu * H_ctrl
        if s1 == 0:
            O_c, O_s = sigmax(), sigmay()
        else:
            O_c = Oc_table[beta]
            O_s = Os_table[beta]

        gap_est = robust_gap_estimate(H_tot,O_c,O_s,upper=nu,eps=1e-4,N_shots=54)
        E_delta_vec.append(gap_est)
        E_delta_true.append(float(spectral_gap(H_tot)))

print(E_delta_vec)
print(E_delta_true)
