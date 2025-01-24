import qutip as qt
import numpy as np
from PXPsubroutines import *
import matplotlib.pyplot as plt
import pickle

N = 6
H_tilde = H_PXP_tilde(N)
two_qubit_set_tilde = all_two_qubit_set_NN(N)
ancilla_two_qubit_set_tilde = ancilla_two_qubit_set(N)

# rho_tilde = create_minus_zero_state(N)
# rho_tilde = create_CDW_state(N)
rho_tilde = create_spin_state(N,[1,2,3,4])

T_column = []
E_column = []
Gradient_norm_column = []
Second_derivative_column = []
dt = np.pi/100

# eigenenergies, eigenstates = H_tilde.eigenstates()
# ground_state = eigenstates[0]
# print(decompose_into_product_state(ground_state))


# print(rho_tilde.purity())
# print(energy(rho_tilde, H_tilde))

# with open('rho_pxp_puregd_5000.pkl', 'rb') as f:
#     rho_tilde = np.load(f, allow_pickle=True)
# print(compute_overlap_with_ground_state(H_tilde,rho_tilde))

for i in range(1000):
    gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
    rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
    # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
    rho = trace_out_rho_tilde(rho_tilde)
    rho_tilde = rho_to_rho_tilde(rho)

    E = energy(rho_tilde, H_tilde)
    gradient_norm = np.linalg.norm(gradients)
    print(i)
    print(E)
    print(gradient_norm)
    # print(compute_overlap_with_ground_state(H_tilde, rho_tilde))
    gradient_norm = np.linalg.norm(gradients)
    T_column.append(i)
    E_column.append(E)
    Gradient_norm_column.append(gradient_norm)
    # print(rho_tilde.purity())

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(T_column, E_column)
axes[0].set_xlabel('t')
axes[0].set_ylabel('Energy')
axes[0].ticklabel_format(style='plain', axis='y', useOffset=False)

axes[1].plot(T_column, Gradient_norm_column)
axes[1].set_xlabel('t')
axes[1].set_ylabel('Gradient norm (non ancilla gates)')

print(compute_overlap_with_ground_state(H_tilde,rho_tilde))
plt.show()



# file_name = 'rho_pxp.pkl'
# with open(file_name, "wb") as f:
#     pickle.dump(rho, f)











