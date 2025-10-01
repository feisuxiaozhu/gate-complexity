import numpy as np
from qutip import sigmax, sigmay, sigmaz
from scipy.optimize import least_squares
from RFE import (
    pauli, spectral_gap, Oc_table, Os_table,
    def_phi_plus, run_shots, decide_low_or_high
)



def H_from_lambda(lmb):
    l1, l2, l3 = lmb
    return l1 * sigmaz() + l2 * sigmax() + l3 * sigmay()

def H_total(nu, beta, H_0):
    return H_0 - nu * (0.5 * pauli[beta])

def robust_gap_estimate(phi_plus, H_tot, O_c, O_s, upper, eps, N_shots):
    a, b = 0.0, upper
    T = 0.0
    while b - a > eps:
        t = np.pi / (b - a)
        U = (-1j * H_tot * t).expm()
        psi_t = U * phi_plus
        X = run_shots(psi_t, O_c, N=N_shots, seed=None)
        Y = run_shots(psi_t, O_s, N=N_shots, seed=None)
        keep = decide_low_or_high(a, b, X, Y)
        if keep == 0:
            b = (a + 2.0 * b) / 3.0
        else:
            a = (2.0 * a + b) / 3.0
        T += N_shots * t
    return 0.5 * (a + b), T

def delta_E_RFE(lambda_1, lambda_2, lambda_3, nu, beta, eps, N_shots):
    H_true = lambda_1 * sigmaz() + lambda_2 * sigmax() + lambda_3 * sigmay()
    H_ctrl = 0.5 * pauli[beta]
    H_tot = H_true - nu * H_ctrl
    O_c = Oc_table[beta]
    O_s = Os_table[beta]
    phi_plus = def_phi_plus(1, beta)
    gap_est, total_time = robust_gap_estimate(phi_plus, H_tot, O_c, O_s, nu+H_true.norm(), eps, N_shots)
    return gap_est, total_time

def residuals(lmb):
    H = H_from_lambda(lmb)
    vals = []
    for (nu, beta), tgt in zip(experiments, targets):
        gap = spectral_gap(H_total(nu, beta, H))
        vals.append(gap - tgt)
    return np.array(vals)

if __name__ == "__main__":
    lambda_true = np.array([0.1,0.5,0.3])
    x0 = np.array([0.0,0.4,0.1])
    nu=4
    eps = 1e-2
    N_shots = 11
    repeat = 500
    experiments = [(nu, 3), (nu, 2), (nu, 1)]
    counter = 0
    T_all_exp = []
    l2_error_all_exp = []

    for _ in range(repeat):
        total_T = []
        targets = []
        for (nu, beta) in experiments:
            est_gap, T_used = delta_E_RFE(
                lambda_true[0], lambda_true[1], lambda_true[2],
                nu, beta, eps, N_shots
            )
            targets.append(est_gap)
            total_T.append(T_used)

        
        res = least_squares(residuals, x0)
        result = res.x
        l2_error = np.linalg.norm(result - lambda_true, ord=2)

        if l2_error < eps:
            counter += 1
            T_all_exp.append(np.mean(total_T))
            l2_error_all_exp.append(l2_error)
    print(f"nu:  {nu}")
    print(f"repeat: {repeat}")
    print(f"eps: {eps:.3e}")
    print(f"N_shots: {N_shots}")
    print(f"success rate: {counter / repeat:.3f}")
    print("average l2 error: " + f"{np.mean(l2_error_all_exp):.3e}")
    print("average total time: " + f"{np.mean(T_all_exp):.3e}")
