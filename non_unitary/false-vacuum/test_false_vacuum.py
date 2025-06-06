import qutip as qt
import numpy as np
from false_vacuum_subroutines import *
import matplotlib.pyplot as plt
import pickle
import random
import os
N=6
H_tilde = H_TFIM(N,hx=1,hz=0.25)
two_qubit_set_tilde = all_two_qubit_set_NN(N)
ancilla_two_qubit_set_tilde = ancilla_two_qubit_set(N)
rho_tilde = create_spin_state(N,[])
up_index = [i for i in range(N)]
all_up_tilde = create_spin_state(N,up_index)

full_rho_tilde = generate_all_spin_states(N)
T_column = []
E_column = []
Gradient_norm_column = []
Second_derivative_column = []
dt = np.pi/100
# print(energy(rho_tilde,H_tilde))

# list_of_energyies=[]
# for i in range(5):
#     hx = i*0.25
#     H_tilde = H_TFIM(N,hx,hz=0.25)
#     list_of_energyies.append(ground_state_energy(H_tilde))
# print(list_of_energyies)

# list_of_metastable_energies = []
# for i in range(4):
#     hx = i*0.25
#     rho_tilde = random.choice(full_rho_tilde)
#     H_tilde = H_TFIM(N,hx,hz=0.25)
#     for i in range(300):
#         gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
#         rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)

#         E = energy(rho_tilde, H_tilde)
#         if i%50==0:
#             print('iteration: '+ str(i) + ' energy: ' +str(E))
#     H_tilde = H_TFIM(N,hx,hz=-0.25)
#     print(energy(rho_tilde,H_tilde))
# print(list_of_metastable_energies)
# list_of_metastable_energies = [-3.5,-3.622984403370049,-4.053763419873856,-4.77616659052858]

# check metastable state separation
all_rho_tilde = [create_spin_state(N,[1])]
i=0
for rho_tilde in all_rho_tilde:
    i+=1
    print(i)
    E_final, rho_fianl = driver(rho_tilde,H_tilde,two_qubit_set_tilde,ancilla_two_qubit_set_tilde ,dt,100)
    E_column.append(E_final)
    # print(E_final)
print(E_column)


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
# rho_tilde =create_spin_state(N,[0,1,2,3,4,5])
# top_three, top_state = top_three_spin_configurations(rho_tilde)
# for i in range(20):
#         gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
#         rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
#         # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#         # rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
#         # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
#         rho = trace_out_rho_tilde(rho_tilde)
#         rho_tilde = rho_to_rho_tilde(rho)

#         E = energy(rho_tilde, H_tilde)
#         gradient_norm = np.linalg.norm(gradients)
#         # if i%100==0:
#         #     print(i)
#         #     print(E)
#         #     print(gradient_norm)
#         # print(compute_overlap_with_ground_state(H_tilde, rho_tilde))
#         gradient_norm = np.linalg.norm(gradients)
#         T_column.append(i)
#         E_column.append(E)
#         Gradient_norm_column.append(gradient_norm)
#         # print(rho_tilde.purity())
# print(E)
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
# print(qt.tracedist(create_spin_state(N,[0,1,2,3,4,5]), rho_tilde))
# plt.show()




# 1 v 1 for all initial states, comparison between SGD and ancilla method.
# for k in range(8): # divide into eight groups, each group contains 8 figures
#     all_rho_tilde = full_rho_tilde[k*8:(k+1)*8]
#     fig_width = 933 / 100
#     fig_height = 1385 / 100
#     fig, axes = plt.subplots(len(all_rho_tilde), 2, figsize=(fig_width, fig_height),constrained_layout=True)
#     steps = 40
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
#         # axes[j][0].set_xlabel('t')
#         axes[j][0].set_ylabel('Energy')
#         axes[j][0].ticklabel_format(style='plain', axis='y', useOffset=False)
#         axes[j][0].set_ylim(-7, 6)
#         axes[j][0].set_title(top_state.removeprefix(top_state[0]))
#         axes[j][0].set_xticks([])

#         axes[j][1].plot(T_column, Gradient_norm_column_SGD,color='blue')
#         axes[j][1].plot(T_column, Gradient_norm_column_ancilla,color='red')
#         axes[j][1].set_xlabel('t')
#         axes[j][1].set_ylabel('Gradient norm ')

#     # print(compute_overlap_with_ground_state(H_tilde,rho_tilde))
#     script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script location
#     filename = '1v1_hx0_100steps_set'+str(k+1) +'.pdf'
#     file_path = os.path.join(script_dir, filename)
#     plt.savefig(file_path)




#1 v 1 for a single state, comparison between SGD and ancilla + SGD method.
# rho_tilde = create_spin_state(N,[0,1,2,5])
# fig, axes = plt.subplots(1, 2, figsize=(16, 5))
# steps = 50
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
# # axes[0].set_ylim(-65, 15)
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