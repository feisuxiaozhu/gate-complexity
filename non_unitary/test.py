import qutip as qt
import numpy as np
from subroutines import *

N=5
H = NN_H(N)
two_qubit_set = all_two_qubit_set_NN(N)
dt = np.pi*(1/1000)

rho_1 = create_spin_state(N,[0,1])
p1 = qt.tensor([sx, sx, qt.qeye(2),qt.qeye(2),  qt.qeye(2)])
rho_1 = evolve(rho_1,p1,dt)

rho_2 = create_spin_state(N,[1,3])
p2 = qt.tensor([qt.qeye(2), sx, qt.qeye(2),sx,  qt.qeye(2)])
rho_2 = evolve(rho_2,p2,dt)

rho = 1/2*rho_1 + 1/2*rho_2


# for i in range(2000):
#     gradients = compute_gradient(rho, H, two_qubit_set)
#     rho = optimize_rho(rho,gradients,two_qubit_set)
#     print(energy(rho,H))
#     # print(gradients)

hessian = compute_hessian(rho, H, two_qubit_set)
print(find_non_symmetric_indices(hessian))
print(hessian[0][22], hessian[22][0])


