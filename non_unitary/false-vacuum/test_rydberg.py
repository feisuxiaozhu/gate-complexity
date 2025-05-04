import qutip as qt
import numpy as np
from false_vacuum_subroutines import *
import matplotlib.pyplot as plt
import pickle
import random
import os
N = 6
Omega =  3*np.pi 
r0 = 8.
Rb = 9.76
C6 = Rb**6*Omega

Neel = neel_operator(N)

Delta_glob = 2*np.pi * 2.5
Delta_loc =  2*np.pi *0.625

H_tilde = rydberg_hamiltonian_periodic(N, Omega, C6, r0, Delta_glob,Delta_loc)
two_qubit_set_tilde = all_two_qubit_set_NN(N)
ancilla_two_qubit_set_tilde = ancilla_two_qubit_set(N)

# print(list(itertools.product([0, 1], repeat=N))[32])
rho_tilde = create_spin_state(N,[0,2,4])
# print((rho_tilde*Neel).tr().real)
# print(energy(rho_tilde,H_tilde))
full_rho_tilde = generate_all_spin_states(N)
# rydberg_landscape(N, Omega, C6, r0, Delta_glob,Delta_loc)

T_column = []
E_column = []
N_column = []
EN_column = []
Gradient_norm_column = []
Second_derivative_column = []
dt = np.pi/1000

# list_of_energyies=[]
# for i in range(4):
#     Omega = (i+1)*np.pi 
#     H_tilde = rydberg_hamiltonian_periodic(N, Omega, C6, r0, Delta_glob,-Delta_loc)
#     list_of_energyies.append(ground_state_energy(H_tilde))
# print(list_of_energyies)


# list_of_metastable_energies = []
# for i in range(4):
#     Omega = (i+1)*np.pi 
#     rho_tilde = create_spin_state(N,[1,3,5])
#     H_tilde = rydberg_hamiltonian_periodic(N, Omega, C6, r0, Delta_glob,-Delta_loc)
#     for i in range(300):
#         gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
#         rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
#         E = energy(rho_tilde, H_tilde)
#         if i%50==0:
#             print('iteration: '+ str(i) + ' energy: ' +str(E))
#     H_tilde = rydberg_hamiltonian_periodic(N, Omega, C6, r0, Delta_glob,Delta_loc)
#     list_of_metastable_energies.append(energy(rho_tilde,H_tilde))
#     # print(energy(rho_tilde,H_tilde))
# print(list_of_metastable_energies)



# metastable_energies = [-38.74178040308486, -41.63262826035188, -45.61503026245358,-50.521617385184896]



i=0
for rho_tilde in full_rho_tilde:
    i+=1
    print(i)
    E_final, rho_tilde = driver(rho_tilde,H_tilde,two_qubit_set_tilde,ancilla_two_qubit_set_tilde ,dt,150)
    N_final = (rho_tilde*Neel).tr().real
    E_column.append(E_final)
    N_column.append(N_final)
    EN_column.append((E_final, N_final))
    print(E_final)
    print(N_final)
# print(E_column)
# print(N_column)
print(EN_column)



# for i in range(50):
#     gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
#     # if i==0:
#     #     rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
#     # else:
#     #     rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)

#     rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
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
# print(top_three_spin_configurations(rho_tilde))
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




# 1 v 1 for all initial states, comparison between SGD and ancilla method.
# for k in range(8): # divide into eight groups, each group contains 8 figures
#     all_rho_tilde = full_rho_tilde[k*8:(k+1)*8]
#     fig_width = 933 / 100
#     fig_height = 1385 / 100
#     fig, axes = plt.subplots(len(all_rho_tilde), 2, figsize=(fig_width, fig_height),constrained_layout=True)
#     steps = 100
#     for j in range(len(all_rho_tilde)):
#         rho_tilde= all_rho_tilde[j]
#         rho_tilde_SGD = rho_tilde.copy()
#         rho_tilde_ancilla = rho_tilde.copy()
#         top_three, top_state = top_three_spin_configurations(rho_tilde)
#         print('working on state number: '+ str(j+1) + ', group number: '+str(k+1)+ ', with configuration: '+ top_state.removeprefix(top_state[0]))
#         T_column= []
#         E_column_SGD = []
#         Gradient_norm_column_SGD = []
#         E_column_ancilla = []
#         Gradient_norm_column_ancilla = []

#         for i in range(steps):
#             if i%10 ==0:
#                 print('at step: '+str(i+1))
#             # Do SGD 
#             gradients_SGD = compute_gradient(rho_tilde_SGD, H_tilde, two_qubit_set_tilde)
#             rho_tilde_SGD = optimizer_1step_SGD_no_scheduling(rho_tilde_SGD, gradients_SGD, two_qubit_set_tilde, dt)
#             # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#             # rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
#             rho = trace_out_rho_tilde(rho_tilde_SGD)
#             rho_tilde_SGD = rho_to_rho_tilde(rho)
#             E = energy(rho_tilde_SGD, H_tilde)
#             gradient_norm_SGD = np.linalg.norm(gradients_SGD)
            
#             # print(E)
#             # print(gradient_norm_SGD)
#             T_column.append(i)
#             E_column_SGD.append(E)
#             Gradient_norm_column_SGD.append(gradient_norm_SGD)
#         # for i in range(steps):
#             # Do ancilla
#             gradients_ancilla = compute_gradient(rho_tilde_ancilla, H_tilde, two_qubit_set_tilde)
#             rho_tilde_ancilla = optimizer_1step_SGD_no_scheduling(rho_tilde_ancilla, gradients_ancilla, two_qubit_set_tilde, dt)
#             # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#             rho_tilde_ancilla, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde_ancilla, ancilla_two_qubit_set_tilde , dt, H_tilde)
#             rho = trace_out_rho_tilde(rho_tilde_ancilla)
#             rho_tilde_ancilla = rho_to_rho_tilde(rho)

#             E = energy(rho_tilde_ancilla, H_tilde)
#             gradient_norm_ancilla = np.linalg.norm(gradients_ancilla)
#             # print(i)
#             # print(E)
#             # print(gradient_norm_ancilla)
#             E_column_ancilla.append(E)
#             Gradient_norm_column_ancilla.append(gradient_norm_ancilla)

#         # print(T_column)
#         # print(E_column_SGD)
#         axes[j][0].plot(T_column, E_column_SGD, color='blue')
#         axes[j][0].plot(T_column, E_column_ancilla,color='red')
#         axes[j][0].set_xlabel('t')
#         axes[j][0].set_ylabel('Energy')
#         axes[j][0].ticklabel_format(style='plain', axis='y', useOffset=False)
#         axes[j][0].set_ylim(-65, 15)
#         axes[j][0].set_title(top_state.removeprefix(top_state[0]))
#         axes[j][0].set_xticks([])

#         axes[j][1].plot(T_column, Gradient_norm_column_SGD,color='blue')
#         axes[j][1].plot(T_column, Gradient_norm_column_ancilla,color='red')
#         axes[j][1].set_xlabel('t')
#         axes[j][1].set_ylabel('Gradient norm ')

#     # print(compute_overlap_with_ground_state(H_tilde,rho_tilde))
#     script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script location
#     filename = '1v1_omega2pi_100steps_set'+str(k+1) +'.pdf'
#     file_path = os.path.join(script_dir, filename)
#     plt.savefig(file_path)



#1 v 1 for a single state, comparison between SGD and ancilla + SGD method.
# rho_tilde = create_spin_state(N,[1,2,3])
# fig, axes = plt.subplots(1, 2, figsize=(16, 5))
# steps = 300
# rho_tilde_SGD = rho_tilde.copy()
# rho_tilde_ancilla = rho_tilde.copy()
# top_three, top_state = top_three_spin_configurations(rho_tilde)
# E_column_SGD = []
# Gradient_norm_column_SGD = []
# E_column_ancilla = []
# Gradient_norm_column_ancilla = []
# for i in range(steps):
#     if i%10 ==0:
#         print('at step: '+str(i+1))
#     # Do SGD 
#     gradients_SGD = compute_gradient(rho_tilde_SGD, H_tilde, two_qubit_set_tilde)
#     rho_tilde_SGD = optimizer_1step_SGD_no_scheduling(rho_tilde_SGD, gradients_SGD, two_qubit_set_tilde, dt)
#     # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#     # rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
#     rho = trace_out_rho_tilde(rho_tilde_SGD)
#     rho_tilde_SGD = rho_to_rho_tilde(rho)
#     E = energy(rho_tilde_SGD, H_tilde)
#     gradient_norm_SGD = np.linalg.norm(gradients_SGD)
    
#     # print(E)
#     # print(gradient_norm_SGD)
#     T_column.append(i)
#     E_column_SGD.append(E)
#     Gradient_norm_column_SGD.append(gradient_norm_SGD)
# # for i in range(steps):
#     # Do ancilla
#     gradients_ancilla = compute_gradient(rho_tilde_ancilla, H_tilde, two_qubit_set_tilde)
#     rho_tilde_ancilla = optimizer_1step_SGD_no_scheduling(rho_tilde_ancilla, gradients_ancilla, two_qubit_set_tilde, dt)
#     # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#     rho_tilde_ancilla, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde_ancilla, ancilla_two_qubit_set_tilde , dt, H_tilde)
#     rho = trace_out_rho_tilde(rho_tilde_ancilla)
#     rho_tilde_ancilla = rho_to_rho_tilde(rho)

#     E = energy(rho_tilde_ancilla, H_tilde)
#     gradient_norm_ancilla = np.linalg.norm(gradients_ancilla)
#     # print(i)
#     # print(E)
#     # print(gradient_norm_ancilla)
#     E_column_ancilla.append(E)
#     Gradient_norm_column_ancilla.append(gradient_norm_ancilla)

# # print(T_column)
# # print(E_column_SGD)
# axes[0].plot(T_column, E_column_SGD, color='blue')
# axes[0].plot(T_column, E_column_ancilla,color='red')
# axes[0].set_xlabel('t')
# axes[0].set_ylabel('Energy')
# # axes[0].ticklabel_format(style='plain', axis='y')
# axes[0].set_ylim(-65, 15)
# axes[0].set_title(top_state.removeprefix(top_state[0]))
# # axes[0].set_xticks([])

# axes[1].plot(T_column, Gradient_norm_column_SGD,color='blue')
# axes[1].plot(T_column, Gradient_norm_column_ancilla,color='red')
# axes[1].set_xlabel('t')
# axes[1].set_ylabel('Gradient norm ')

# # print(compute_overlap_with_ground_state(H_tilde,rho_tilde))
# # script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script location
# # filename = '1v1_omega2pi_100steps_set'+str(k+1) +'.pdf'
# # file_path = os.path.join(script_dir, filename)
# plt.show()

# print(T_column)
# print(E_column_SGD)
# print(E_column_ancilla)
# print(Gradient_norm_column_SGD)
# print(Gradient_norm_column_ancilla)