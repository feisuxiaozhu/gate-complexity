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

rho_tilde = 1/3*rho_1_tilde + 1/3*rho_2_tilde + 1/3*rho_3_tilde 
# rho_tilde = 1/2*rho_1_tilde + 1/2*rho_2_tilde
# rho_tilde = rho_3_tilde
# print(energy(rho_to_rho_tilde(rho_tilde), rho_to_rho_tilde(H_tilde)))


T_column = []
E_column = []
Gradient_norm_column = []
Second_derivative_column = []
dt = np.pi/100

print(energy(rho_tilde, H_tilde))


for i in range(500):
    gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
    ancilla_gradients = compute_gradient(rho_tilde, H_tilde, ancilla_two_qubit_set_tilde)
 
    # rho_tilde =  optimizer_1step_SGD_hessian(rho_tilde,gradients,two_qubit_set_tilde,dt, H_tilde)
    # rho_tilde = opti
    rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
    # rho_tilde = optimizer_1step_SGD_hessian(rho_tilde,gradients,two_qubit_set_tilde, dt, H_tilde)
    rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
    # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
    rho = trace_out_rho_tilde(rho_tilde)
    rho_tilde = rho_to_rho_tilde(rho)


    E = energy(rho_tilde, H_tilde)
    ancilla_gradient_norm = np.linalg.norm(ancilla_gradients)
    gradient_norm = np.linalg.norm(gradients)
    second_derivative_effective_norm = np.linalg.norm(second_derivatives)
    print(i)
    print(E)
    print(gradient_norm)
    print(second_derivative_effective_norm)
    T_column.append(i)
    E_column.append(E)
    Gradient_norm_column.append(gradient_norm)
    Second_derivative_column.append(second_derivative_effective_norm)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(T_column, E_column)
axes[0].set_xlabel('t')
axes[0].set_ylabel('Energy')
axes[0].ticklabel_format(style='plain', axis='y', useOffset=False)

axes[1].plot(T_column, Gradient_norm_column)
axes[1].set_xlabel('t')
axes[1].set_ylabel('Gradient norm (non ancilla gates)')

axes[2].plot(T_column, Second_derivative_column)
axes[2].set_xlabel('t')
axes[2].set_ylabel('Effective second derivative norm')
plt.show()