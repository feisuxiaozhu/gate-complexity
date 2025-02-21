import qutip as qt
import numpy as np
from false_vacuum_subroutines import *
import matplotlib.pyplot as plt
import pickle
import random
N=6
H_tilde = H_TFIM(N,hx=0,hz=0.25)
two_qubit_set_tilde = all_two_qubit_set_NN(N)
ancilla_two_qubit_set_tilde = ancilla_two_qubit_set(N)
all_down_tilde = create_spin_state(N,[])
up_index = [i for i in range(N)]
all_up_tilde = create_spin_state(N,up_index)

all_rho_tilde = generate_all_spin_states(N)
T_column = []
E_column = []
Gradient_norm_column = []
Second_derivative_column = []
dt = np.pi/100

# all_rho_tilde = random.sample(all_rho_tilde,16)
# i=0
# for rho_tilde in all_rho_tilde:
#     i+=1
#     print(i)
#     E_final = driver(rho_tilde,H_tilde,two_qubit_set_tilde,ancilla_two_qubit_set_tilde ,dt,100)
#     E_column.append(E_final)
#     print(E_final)
# print(E_column)

all_rho_tilde = all_rho_tilde[56:64]
fig, axes = plt.subplots(len(all_rho_tilde), 2, figsize=(16, 5))
for j in range(len(all_rho_tilde)):
    print('working on state: '+ str(j))
    rho_tilde = all_rho_tilde[j]
    top_three, top_state = top_three_spin_configurations(rho_tilde)
    T_column = []
    E_column = []
    Gradient_norm_column = []
    Second_derivative_column = []
    for i in range(75):
        gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
        rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
        # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
        rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
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

    axes[j][0].plot(T_column, E_column)
    # axes[j][0].set_xlabel('t')
    axes[j][0].set_ylabel('Energy')
    axes[j][0].ticklabel_format(style='plain', axis='y', useOffset=False)
    axes[j][0].set_ylim(-7, 6)
    axes[j][0].set_title(top_state.removeprefix(top_state[0]))
    axes[j][0].set_xticks([])

    axes[j][1].plot(T_column, Gradient_norm_column)
    axes[j][1].set_xlabel('t')
    axes[j][1].set_ylabel('Gradient norm ')

# print(compute_overlap_with_ground_state(H_tilde,rho_tilde))
plt.show()

