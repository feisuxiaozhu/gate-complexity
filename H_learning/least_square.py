import numpy as np
from qutip import sigmax, sigmay, sigmaz
from scipy.optimize import least_squares
from RFE import delta_E_RFE, pauli, spectral_gap

lambda_true = np.array([0.1, 0.5, 0.3])
experiments = [(10, 3), (10, 2), (10, 1)]

def H_from_lambda(lmb):
    l1, l2, l3 = lmb
    return l1 * sigmaz() + l2 * sigmax() + l3 * sigmay()

def H_total(nu, beta, H_0):
    return H_0 - nu * (0.5 * pauli[beta])

# targets are RFE results, and should only be computed once and remain fixed in later least square algorithm!
targets = np.array([
    delta_E_RFE(lambda_true[0], lambda_true[1], lambda_true[2], nu, beta)
    for (nu, beta) in experiments
])

def residuals(lmb):
    H = H_from_lambda(lmb)
    values = []
    for (nu, beta), tgt in zip(experiments, targets):
        gap = spectral_gap(H_total(nu, beta, H))
        values.append(gap - tgt)
    return np.array(values)

x0 = np.array([0.09, 0.51, 0.29])
res = least_squares(residuals, x0)
print("estimate lambda:", res.x)
