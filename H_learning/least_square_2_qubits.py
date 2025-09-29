import numpy as np
from qutip import sigmax, sigmay, sigmaz
from scipy.optimize import least_squares
from RFE import delta_E_RFE, pauli, spectral_gap,robust_gap_estimate
from RFE_2_qubits import H_0, H_ctrl_func, two_qubit_eigenstate,Oc_Os_decider

lambda_true = np.array([0.1,0.2,0.3, 0.5,0.6,0.3,0.2,0.1,0.1, 0.2,0.1,0.1, 0.3,0.22,0.15])
H_true = H_0(lambda_true)
nu=20
eps = 1e-4
N_shots=30

# targets are RFE results, and should only be computed once and remain fixed in later least square algorithm!
targets = []
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
                        gap_est = robust_gap_estimate(phi_plus,H_tot,O_c,O_s,nu+2.0 * H_true.norm() ,eps,N_shots)
                        targets.append(gap_est)


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
    x0 = np.array([0.11,0.21,0.32, 0.51,0.63,0.31,0.22,0.11,0.11, 0.22,0.11,0.11, 0.33,0.22,0.15])
    res = least_squares(residuals, x0)
    print("estimate lambda:", res.x)



