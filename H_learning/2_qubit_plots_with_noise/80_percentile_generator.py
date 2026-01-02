import numpy as np
from qutip import sigmax, sigmay, sigmaz
from scipy.optimize import least_squares
from RFE import delta_E_RFE, pauli, spectral_gap, run_shots, decide_low_or_high
from RFE_2_qubits import H_0, H_ctrl_func, two_qubit_eigenstate,Oc_Os_decider
import pandas as pd
import os

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
    values = []
    counter = 0
    for k in [0,1]:
        for s1 in [0,1]:
            for s2 in [0,1]:
                for beta_1 in [1,2,3]:
                    for beta_2 in [1,2,3]:
                        H_ctrl = H_ctrl_func(k,s1,s2,beta_1,beta_2) 
                        H_tot = H_0(lmb) - nu * H_ctrl
                        gap = spectral_gap(H_tot)
                        values.append(gap-targets[counter])
                        counter += 1
    # print(values)
    return np.array(values)

if __name__ == "__main__":
    eps_col = [1e-2,1e-3,1e-4,1e-5,1e-6]
    N_shots_col = [25,27,29,31,33]
    for eps, N_shots in zip( eps_col, N_shots_col):
        x0 = np.array([0.11,0.21,0.32, 0.51,0.63,0.31,0.22,0.11,0.11, 0.22,0.11,0.11, 0.33,0.22,0.15])
        lambda_true = np.array([0.1,0.2,0.3, 0.5,0.6,0.3,0.2,0.1,0.1, 0.2,0.1,0.1, 0.3,0.22,0.15])
        nu= 5
        # eps = 1e-2
        # N_shots=25
        # print(eps, N_shots)
        repeat = 200
        counter = 0
        T_all_exp = []
        l2_error_all_exp = []
        l2_error_unfiltered = []
        H_true = H_0(lambda_true)
        
        for _ in range(repeat):
            # targets are RFE results, and should only be computed once and remain fixed in later least square algorithm!
            targets = []
            total_T = []
            temp_T = 1
            # print(eps,_)
            for k in [0,1]:
                    for s1 in [0,1]:
                        for s2 in [0,1]:
                            for beta_1 in [1,2,3]:
                                for beta_2 in [1,2,3]:
                                    H_ctrl = H_ctrl_func(k,s1,s2,beta_1,beta_2) 
                                    H_tot = H_true - nu * H_ctrl
                                    O_c, O_s = Oc_Os_decider(k, s1, s2, beta_1, beta_2)   
                                    if k==0:
                                        phi_0 = two_qubit_eigenstate(s1, s2, beta_1, beta_2)
                                        phi_1 = two_qubit_eigenstate(1-s1, s2, beta_1, beta_2)     
                                    else:
                                        phi_0 = two_qubit_eigenstate(s1, s2, beta_1, beta_2)
                                        phi_1 = two_qubit_eigenstate(s1, 1-s2, beta_1, beta_2)
                                    phi_plus = (phi_0 + phi_1).unit()
                                    gap_est, T_used = robust_gap_estimate(phi_plus,H_tot,O_c,O_s,nu+2.0 * H_true.norm() ,eps,N_shots)
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
        
        
        
        

