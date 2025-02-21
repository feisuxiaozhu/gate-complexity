import qutip as qt
import numpy as np
from false_vacuum_subroutines import *
import matplotlib.pyplot as plt
import pickle
import random
import os
N=6
H_tilde = H_TFIM(N,hx=0,hz=0.25)
two_qubit_set_tilde = all_two_qubit_set_NN(N)
ancilla_two_qubit_set_tilde = ancilla_two_qubit_set(N)
all_down_tilde = create_spin_state(N,[])
up_index = [i for i in range(N)]
all_up_tilde = create_spin_state(N,up_index)

full_rho_tilde = generate_all_spin_states(N)
T_column = []
E_column = []
Gradient_norm_column = []
Second_derivative_column = []
dt = np.pi/100


# check metastable state separation
# i=0
# for rho_tilde in all_rho_tilde:
#     i+=1
#     print(i)
#     E_final = driver(rho_tilde,H_tilde,two_qubit_set_tilde,ancilla_two_qubit_set_tilde ,dt,10000)
#     E_column.append(E_final)
#     print(E_final)
# print(E_column)


# run a list of state
# all_rho_tilde = all_rho_tilde[56:64]
# fig, axes = plt.subplots(len(all_rho_tilde), 2, figsize=(16, 5))
# for j in range(len(all_rho_tilde)):
#     print('working on state: '+ str(j))
#     rho_tilde = all_rho_tilde[j]
#     top_three, top_state = top_three_spin_configurations(rho_tilde)
#     T_column = []
#     E_column = []
#     Gradient_norm_column = []
#     Second_derivative_column = []
#     for i in range(75):
#         gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
#         rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
#         # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#         rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
#         # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#         rho = trace_out_rho_tilde(rho_tilde)
#         rho_tilde = rho_to_rho_tilde(rho)

#         E = energy(rho_tilde, H_tilde)
#         gradient_norm = np.linalg.norm(gradients)
#         print(i)
#         print(E)
#         print(gradient_norm)
#         # print(compute_overlap_with_ground_state(H_tilde, rho_tilde))
#         gradient_norm = np.linalg.norm(gradients)
#         T_column.append(i)
#         E_column.append(E)
#         Gradient_norm_column.append(gradient_norm)
#         # print(rho_tilde.purity())

#     axes[j][0].plot(T_column, E_column)
#     # axes[j][0].set_xlabel('t')
#     axes[j][0].set_ylabel('Energy')
#     axes[j][0].ticklabel_format(style='plain', axis='y', useOffset=False)
#     axes[j][0].set_ylim(-7, 6)
#     axes[j][0].set_title(top_state.removeprefix(top_state[0]))
#     axes[j][0].set_xticks([])

#     axes[j][1].plot(T_column, Gradient_norm_column)
#     axes[j][1].set_xlabel('t')
#     axes[j][1].set_ylabel('Gradient norm ')

# # print(compute_overlap_with_ground_state(H_tilde,rho_tilde))
# plt.show()



# single state long run
# rho_tilde =create_spin_state(N,[1,2,3,4])
# top_three, top_state = top_three_spin_configurations(rho_tilde)
# for i in range(5000):
#         gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
#         rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
#         # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#         rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
#         # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#         rho = trace_out_rho_tilde(rho_tilde)
#         rho_tilde = rho_to_rho_tilde(rho)

#         E = energy(rho_tilde, H_tilde)
#         gradient_norm = np.linalg.norm(gradients)
#         if i%100==0:
#             print(i)
#             print(E)
#             print(gradient_norm)
#         # print(compute_overlap_with_ground_state(H_tilde, rho_tilde))
#         gradient_norm = np.linalg.norm(gradients)
#         T_column.append(i)
#         E_column.append(E)
#         Gradient_norm_column.append(gradient_norm)
#         # print(rho_tilde.purity())
# fig, axes = plt.subplots(1, 2, figsize=(16, 5))
# axes[0].plot(T_column, E_column)
# axes[0].set_xlabel('t')
# axes[0].set_ylabel('Energy')
# axes[0].ticklabel_format(style='plain', axis='y', useOffset=False)
# axes[0].set_ylim(-7, 6)
# axes[0].set_title(top_state.removeprefix(top_state[0]))
# axes[0].set_xticks([])

# axes[1].plot(T_column, Gradient_norm_column)
# axes[1].set_xlabel('t')
# axes[1].set_ylabel('Gradient norm ')

# plt.show()




# 1 v 1 for all initial states, comparison between SGD and ancilla method.
for k in range(8): # divide into eight groups, each group contains 8 figures
    all_rho_tilde = full_rho_tilde[k*8:(k+1)*8]
    fig_width = 933 / 100
    fig_height = 1385 / 100
    fig, axes = plt.subplots(len(all_rho_tilde), 2, figsize=(fig_width, fig_height),constrained_layout=True)
    steps = 100
    for j in range(len(all_rho_tilde)):
        rho_tilde= all_rho_tilde[j]
        rho_tilde_SGD = rho_tilde.copy()
        rho_tilde_ancilla = rho_tilde.copy()
        top_three, top_state = top_three_spin_configurations(rho_tilde)
        print('working on state number: '+ str(j+1) + ', group number: '+str(k+1)+ ', with configuration: '+ top_state.removeprefix(top_state[0]))
        T_column= []
        E_column_SGD = []
        Gradient_norm_column_SGD = []
        E_column_ancilla = []
        Gradient_norm_column_ancilla = []

        for i in range(steps):
            if i%10 ==0:
                print('at step: '+str(i+1))
            # Do SGD 
            gradients_SGD = compute_gradient(rho_tilde_SGD, H_tilde, two_qubit_set_tilde)
            rho_tilde_SGD = optimizer_1step_SGD_no_scheduling(rho_tilde_SGD, gradients_SGD, two_qubit_set_tilde, dt)
            # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
            # rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
            rho = trace_out_rho_tilde(rho_tilde_SGD)
            rho_tilde_SGD = rho_to_rho_tilde(rho)
            E = energy(rho_tilde_SGD, H_tilde)
            gradient_norm_SGD = np.linalg.norm(gradients_SGD)
            
            # print(E)
            # print(gradient_norm_SGD)
            T_column.append(i)
            E_column_SGD.append(E)
            Gradient_norm_column_SGD.append(gradient_norm_SGD)
        # for i in range(steps):
            # Do ancilla
            gradients_ancilla = compute_gradient(rho_tilde_ancilla, H_tilde, two_qubit_set_tilde)
            rho_tilde_ancilla = optimizer_1step_SGD_no_scheduling(rho_tilde_ancilla, gradients_ancilla, two_qubit_set_tilde, dt)
            # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
            rho_tilde_ancilla, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde_ancilla, ancilla_two_qubit_set_tilde , dt, H_tilde)
            rho = trace_out_rho_tilde(rho_tilde_ancilla)
            rho_tilde_ancilla = rho_to_rho_tilde(rho)

            E = energy(rho_tilde_ancilla, H_tilde)
            gradient_norm_ancilla = np.linalg.norm(gradients_ancilla)
            # print(i)
            # print(E)
            # print(gradient_norm_ancilla)
            E_column_ancilla.append(E)
            Gradient_norm_column_ancilla.append(gradient_norm_ancilla)

        # print(T_column)
        # print(E_column_SGD)
        axes[j][0].plot(T_column, E_column_SGD, color='blue')
        axes[j][0].plot(T_column, E_column_ancilla,color='red')
        # axes[j][0].set_xlabel('t')
        axes[j][0].set_ylabel('Energy')
        axes[j][0].ticklabel_format(style='plain', axis='y', useOffset=False)
        axes[j][0].set_ylim(-7, 6)
        axes[j][0].set_title(top_state.removeprefix(top_state[0]))
        axes[j][0].set_xticks([])

        axes[j][1].plot(T_column, Gradient_norm_column_SGD,color='blue')
        axes[j][1].plot(T_column, Gradient_norm_column_ancilla,color='red')
        axes[j][1].set_xlabel('t')
        axes[j][1].set_ylabel('Gradient norm ')

    # print(compute_overlap_with_ground_state(H_tilde,rho_tilde))
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script location
    filename = '1v1_hx0_100steps_set'+str(k+1) +'.pdf'
    file_path = os.path.join(script_dir, filename)
    plt.savefig(file_path)