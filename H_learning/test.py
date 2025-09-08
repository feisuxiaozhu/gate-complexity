import numpy as np
from qutip import sigmax, sigmay, sigmaz
from scipy.optimize import least_squares
from RFE import *

lambda_true = np.array([0.1, 0.5, 0.3])

def H_from_lambda(lmb):
    l1, l2, l3 = lmb
    return l1 * sigmaz() + l2 * sigmax() + l3 * sigmay()

# settings = [
#     (8, 1),
#     (10, 2),
#     (12, 3),
# ]

settings = [(6,3),(8,3),(10,3),
            (6,1),(8,1),(10,1),
            (6,2),(8,2),(10,2)]  

H_true = H_from_lambda(lambda_true)
targets = []
for (nu_i, beta_i) in settings:
    H_ctrl_i = 0.5 * pauli[beta_i]
    H_tot_i = H_true - nu_i * H_ctrl_i
    targets.append(float(spectral_gap(H_tot_i)))
targets = np.array(targets)

def residuals(lmb):
    out = []
    for (nu_i, beta_i), gap_i in zip(settings, targets):
        out.append(delta_E_RFE(lmb[0], lmb[1], lmb[2], nu_i, beta_i) - gap_i)
    return np.array(out)

x0 = np.array([0.13, 0.55, 0.24])
res = least_squares(residuals, x0)
# print("status:", res.status, res.message)
print("estimate lambda:", res.x)
# print("residuals:", res.fun, "||res||:", np.linalg.norm(res.fun))



