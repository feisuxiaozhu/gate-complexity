import qutip as qt
import numpy as np
from false_vacuum_subroutines import *
import matplotlib.pyplot as plt
import pickle
import random
N = 6
Omega =  2*np.pi * 1
r0 = 8.
Rb = 9.76
C6 = Rb**6*Omega

Delta_glob = 2*np.pi * 2.5
Delta_loc =  2*np.pi *0.625

H_tilde = rydberg_hamiltonian_periodic(N, Omega, C6, r0, Delta_glob,Delta_loc)
two_qubit_set_tilde = all_two_qubit_set_NN(N)
ancilla_two_qubit_set_tilde = ancilla_two_qubit_set(N)

print(list(itertools.product([0, 1], repeat=N))[32])
rho_tilde = create_spin_state(N,[1,3,5])
print(energy(rho_tilde,H_tilde))
all_rho_tilde = generate_all_spin_states(N)
rydberg_landscape(N, Omega, C6, r0, Delta_glob,Delta_loc)

T_column = []
E_column = []
Gradient_norm_column = []
Second_derivative_column = []
dt = np.pi/1000



# all_rho_tilde = random.sample(all_rho_tilde,1)

i=0
for rho_tilde in all_rho_tilde:
    i+=1
    print(i)
    E_final = driver(rho_tilde,H_tilde,two_qubit_set_tilde,ancilla_two_qubit_set_tilde ,dt)
    E_column.append(E_final)
    print(E_final)
print(E_column)



# for i in range(300):
#     gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
#     if i==0:
#         rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
#     else:
#         rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)

#     # rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
#     # rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
#     # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#     rho = trace_out_rho_tilde(rho_tilde)
#     rho_tilde = rho_to_rho_tilde(rho)

#     E = energy(rho_tilde, H_tilde)
#     gradient_norm = np.linalg.norm(gradients)
#     print('iteration: '+str(i))
#     print('energy: ' + str(E))
#     # print(gradient_norm)
#     # print(top_three_spin_configurations(rho_tilde))
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
