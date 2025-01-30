import qutip as qt
import numpy as np
from false_vacuum_subroutines import *
import matplotlib.pyplot as plt
import pickle


N=6
H_tilde = H_TFIM(N,hx=0.75,hz=0.25)
two_qubit_set_tilde = all_two_qubit_set_NN(N)
ancilla_two_qubit_set_tilde = ancilla_two_qubit_set(N)

all_down_tilde = create_spin_state(N,[])
up_index = [i for i in range(N)]
all_up_tilde = create_spin_state(N,up_index)

rho_tilde = all_down_tilde
all_rho_tilde = generate_all_spin_states(N)


T_column = []
E_column = []
Gradient_norm_column = []
Second_derivative_column = []
dt = np.pi/100

i=0
for rho_tilde in all_rho_tilde:
    i+=1
    print(i)
    E_final = driver(rho_tilde,H_tilde,two_qubit_set_tilde,ancilla_two_qubit_set_tilde ,dt)
    E_column.append(E_final)
    print(E_final)
print(E_column)
# for i in range(200):
#     gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
#     rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
#     # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#     rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
#     # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#     rho = trace_out_rho_tilde(rho_tilde)
#     rho_tilde = rho_to_rho_tilde(rho)

#     E = energy(rho_tilde, H_tilde)
#     gradient_norm = np.linalg.norm(gradients)
#     print(i)
#     print(E)
#     print(gradient_norm)
#     # print(compute_overlap_with_ground_state(H_tilde, rho_tilde))
#     gradient_norm = np.linalg.norm(gradients)
#     T_column.append(i)
#     E_column.append(E)
#     Gradient_norm_column.append(gradient_norm)
#     # print(rho_tilde.purity())

# fig, axes = plt.subplots(1, 2, figsize=(16, 5))
# axes[0].plot(T_column, E_column)
# axes[0].set_xlabel('t')
# axes[0].set_ylabel('Energy')
# axes[0].ticklabel_format(style='plain', axis='y', useOffset=False)

# axes[1].plot(T_column, Gradient_norm_column)
# axes[1].set_xlabel('t')
# axes[1].set_ylabel('Gradient norm (non ancilla gates)')

# # print(compute_overlap_with_ground_state(H_tilde,rho_tilde))
# plt.show()

