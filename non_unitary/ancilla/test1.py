import qutip as qt
import numpy as np
from ancillaSubroutines import *
import pickle
import matplotlib.pyplot as plt

N = 5
H_tilde = NN_H_tilde(N)
two_qubit_set_tilde = all_two_qubit_set_NN(N)
ancilla_two_qubit_set_tilde = ancilla_two_qubit_set(N)

rho_1_tilde = create_spin_state(N,[0,1])
rho_2_tilde = create_spin_state(N,[1,3])
rho_3_tilde = create_spin_state(N,[])
rho_4_tilde = create_spin_state(N,[0,3])

# rho_tilde = 1/3*rho_1_tilde + 1/3*rho_2_tilde + 1/3*rho_3_tilde 
# rho_tilde = 1/2*rho_1_tilde + 1/2*rho_2_tilde
rho_tilde = rho_4_tilde
# print(energy(rho_to_rho_tilde(rho_tilde), rho_to_rho_tilde(H_tilde)))


T_column = []
E_column = []
Gradient_norm_column = []
second_derivative_negative_column = []
dt = np.pi/100

print(energy(rho_tilde, H_tilde))


for i in range(1):
    gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
    ancilla_gradients = compute_gradient(rho_tilde, H_tilde, ancilla_two_qubit_set_tilde)
    ancilla_second_derivatives = compute_hessian_diagonal(rho_tilde, H_tilde, ancilla_two_qubit_set_tilde)
    ancilla_second_derivatives[ancilla_second_derivatives>0] = 0  
    # rho_tilde =  optimizer_1step_SGD_hessian(rho_tilde,gradients,two_qubit_set_tilde,dt, H_tilde)
    # rho_tilde = opti
    rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
    # rho_tilde = optimizer_1step_SGD_hessian(rho_tilde,gradients,two_qubit_set_tilde, dt, H_tilde)
    rho_tilde = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
    # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
    rho = trace_out_rho_tilde(rho_tilde)
    # E = energy(rho,trace_out_rho_tilde(H_tilde))
    rho_tilde = rho_to_rho_tilde(rho)
    E = energy(rho_tilde, H_tilde)
    
    ancilla_gradient_norm = np.linalg.norm(ancilla_gradients)
    ancilla_second_derivative_norm = np.linalg.norm(ancilla_second_derivatives)
    gradient_norm = np.linalg.norm(gradients)
    print(i)
    print(E)
    # print(gradient_norm)
    print(gradient_norm)
    T_column.append(i)
    E_column.append(E)
    Gradient_norm_column.append(gradient_norm)
    second_derivative_negative_column.append(ancilla_second_derivative_norm)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(T_column, E_column)
axes[0].set_xlabel('t')
axes[0].set_ylabel('Energy')
axes[0].ticklabel_format(style='plain', axis='y', useOffset=False)

axes[1].plot(T_column, Gradient_norm_column)
axes[1].set_xlabel('t')
axes[1].set_ylabel('Gradient norm')

axes[2].plot(T_column, second_derivative_negative_column)
axes[2].set_xlabel('t')
axes[2].set_ylabel('Ancilla second derivative norm')
plt.show()