import qutip as qt
import numpy as np
from two_d_subroutines import *
import time
import matplotlib.pyplot as plt
import random
M=1
N=6
H_tilde =  ising_2d_hamiltonian(M,N,hx=0,hz=0.25)
two_qubit_set_tilde = all_two_qubit_set_NN(M,N)
ancilla_two_qubit_set_tilde = ancilla_two_qubit_set(M,N)

# rho_tilde = generate_spin_state(M,N, state_type='custom',custom_state = [1, 1, 1, 0,0,0] )

rho_tilde = generate_spin_state(M,N, state_type='custom',custom_state = [1, 1, 1, 1,1,1] )
# rho_tilde = generate_spin_state(M,N)

T_column = []
E_column = []
Gradient_norm_column = []

dt = np.pi/100

# print('initial energy: '+ str(energy(rho_tilde,H_tilde)))

# all_rho_tilde = generate_all_spin_states(M,N)
# # for j in range(40):
# j=0
# for rho_tilde in all_rho_tilde:
#     # rho_tilde = random.choice(all_rho_tilde)
#     j+=1
#     print('checking state: '+str(j))
#     E_final, gradient_norm_final = driver(rho_tilde,H_tilde,two_qubit_set_tilde,ancilla_two_qubit_set_tilde ,dt)
#     E_column.append(E_final)
#     Gradient_norm_column.append(gradient_norm_final)
# print(E_column)
# print(Gradient_norm_column)

for i in range(100):
    start_time = time.time()
    
    gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
    # if i == 0 :
    rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)

    # rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
    # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
    rho = trace_out_rho_tilde(rho_tilde)
    rho_tilde = rho_to_rho_tilde(rho)

    E = energy(rho_tilde, H_tilde)
    gradient_norm = np.linalg.norm(gradients)
    print(i)
    print(E)
    print(gradient_norm)
    # print(compute_overlap_with_ground_state(H_tilde, rho_tilde))
    end_time = time.time()
    elapsed_time = end_time - start_time  # Compute elapsed time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
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
axes[1].set_ylabel('Gradient norm')

plt.show()