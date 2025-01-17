import qutip as qt
import numpy as np
from subroutines import *
import pickle
import matplotlib.pyplot as plt

N=5
H = NN_H(N)
two_qubit_set = all_two_qubit_set_NN(N)
dt = np.pi*(1/1000)

rho_1 = create_spin_state(N,[0,1])
# p1 = qt.tensor([sx, sx, qt.qeye(2),qt.qeye(2),  qt.qeye(2)])
# rho_1 = evolve(rho_1,p1,dt)

rho_2 = create_spin_state(N,[1,3])
# p2 = qt.tensor([qt.qeye(2), sx, qt.qeye(2),sx,  qt.qeye(2)])
# rho_2 = evolve(rho_2,p2,dt)
 
rho = 1/2*rho_1 + 1/2*rho_2

rho_3 = create_spin_state(N,[])
rho = 1/3*rho_1 + 1/3*rho_2 +1/3*rho_3

print(energy(rho, H))
# with open('rho.pkl', 'rb') as f:
#     rho = np.load(f, allow_pickle=True)

# new_rho = rho.full()
# diag = new_rho.diagonal().real
# # diag = np.sort(diag)
# k = 3
# indices = np.argpartition(diag, -k)[-k:]
# print(indices)
T_column = []
E_column = []
Gradient_norm_column = []

dt = np.pi/100
for i in range(1):
    gradients = compute_gradient(rho, H, two_qubit_set)
    # print(np.linalg.norm(gradients))
    # rho =  optimizer_1step_SGD_hessian(rho,gradients,two_qubit_set,dt, H)
    rho = optimizer_1step_SGD_no_scheduling(rho, gradients, two_qubit_set, dt)
    E = energy(rho,H)
    gradient_norm = np.linalg.norm(gradients)
    print(i)
    print(E)
    print(gradient_norm)
    T_column.append(i)
    E_column.append(E)
    Gradient_norm_column.append(gradient_norm)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(T_column, E_column)
axes[0].set_xlabel('t')
axes[0].set_ylabel('Energy')
axes[0].ticklabel_format(style='plain', axis='y', useOffset=False)

axes[1].plot(T_column, Gradient_norm_column)
axes[1].set_xlabel('t')
axes[1].set_ylabel('Gradient norm')

plt.show()

# hessian = compute_hessian(rho, H, two_qubit_set)
# print(is_positive_semi_definite(hessian))
# file_name = 'rho_3.pkl'
# with open(file_name, "wb") as f:
#     pickle.dump(rho, f)


