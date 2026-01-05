import numpy as np
from qutip import sigmax, sigmay, sigmaz
from RFE import  pauli, spectral_gap,  def_phi_plus, run_shots, decide_low_or_high
import pandas as pd
from scipy.optimize import least_squares
import os

# experiments = [(nu, 3), (nu, 2), (nu, 1)]

pauli = {1: sigmax(), 2: sigmay(), 3: sigmaz()}
Oc_table = {1: sigmaz(), 2: sigmaz(), 3: sigmax()}
Os_table = {1: sigmay(), 2: -sigmax(), 3: -sigmay()}

def H_from_lambda(lmb):
    l1, l2, l3 = lmb
    return l1 * sigmaz() + l2 * sigmax() + l3 * sigmay()

def H_total(nu, beta, H_0):
    return H_0 - nu * (0.5 * pauli[beta])

def robust_gap_estimate(phi_plus,H_tot,O_c,O_s,upper,eps,N_shots):
    a, b = 0.0, upper
    T = 0.0
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
        T += N_shots * t
    return 0.5 * (a + b), T

def residuals(lmb):
    vals = []
    counter = 0
    # for (nu, beta), tgt in zip(experiments, targets):
    #     gap = spectral_gap(H_total(nu, beta, H))
    #     vals.append(gap - tgt)
    for beta in [1,2,3]:
        H_ctrl = 0.5 * pauli[beta]
        H_tot = H_from_lambda(lmb) - nu* H_ctrl
        gap = spectral_gap(H_tot)
        vals.append(gap - targets[counter])
        counter += 1
    return np.array(vals)

eps_col = [1e-2,1e-3,1e-4,1e-5,1e-6]
N_shots_col = [25,27,29,31,33]
for eps, N_shots in zip( eps_col, N_shots_col):
    x0 = np.array([0.09, 0.51, 0.29])
    lambda_true = np.array([0.1, 0.5, 0.3])
    nu = 3
    repeat = 200
    counter = 0 
    T_all_exp = []
    l2_error_all_exp = []
    l2_error_unfiltered = []
    H_true = H_from_lambda(lambda_true)

    for _ in range(repeat):
        targets = []
        total_T = []
        temp_T = 0
        for beta in [1,2,3]:
            H_ctrl = 0.5 * pauli[beta]
            H_tot = H_true - nu * H_ctrl
            O_c = Oc_table[beta]
            O_s = Os_table[beta]
            phi_plus = def_phi_plus(1,beta)
            gap_est, T_used = robust_gap_estimate(phi_plus,H_tot,O_c,O_s,nu+2.0 * H_true.norm(),eps,N_shots)
            targets.append(gap_est)
            temp_T += T_used
        total_T.append(temp_T)
        res = least_squares(residuals, x0)
        result = res.x
        l2_error = np.linalg.norm(result - lambda_true, ord=2)
        l2_error_unfiltered.append(l2_error)
        if l2_error < eps:
            counter += 1
            T_all_exp.append(np.mean(total_T))
            l2_error_all_exp.append(l2_error)
    
    unfiltered_mean = np.mean(l2_error_unfiltered)
    unfiltered_std = np.std(l2_error_unfiltered)
    T_total_mean = f"{np.mean(T_all_exp):.3e}"
    print(f"nu:  {nu}")
    print(f"repeat: {repeat}")
    print(f"eps: {eps:.3e}")
    print(f"N_shots: {N_shots}")
    print(f"success rate: {counter / repeat:.3f}")
    print("average l2 error (if success): " + f"{np.mean(l2_error_all_exp):.3e}")
    print("80 percentile l2 error: " +f"{np.percentile(l2_error_unfiltered, 80, method='nearest'):.3e}")
    print("average total time: " + f"{np.mean(T_all_exp):.3e}")
    print("average l2 error (unfiltered): " + f"{unfiltered_mean:.3e}")
    print("std (unfiltered): " + f"{unfiltered_std:.3e}")
    print("---------------------------------------------------")
    filename = f"l2_error_nu{nu}_eps{eps}_shots{N_shots}_Ttotal{T_total_mean}.csv"
    filepath = os.path.join("data", filename)
    df = pd.DataFrame({'l2_error_unfiltered': l2_error_unfiltered})
    df.to_csv(filepath, index=False)   
        
        