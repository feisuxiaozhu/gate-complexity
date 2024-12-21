import qutip as qt
import numpy as np
from subroutines import *

N=5
H = NN_H(N)
two_qubit_set = two_qubit_set(N)
dt = np.pi*(1/10)
##############################################################

# # 0 sanity check to see that operator and state act as expected
# rho = create_spin_state(N,[1,3])
# print(extract_spin_directions_from_rho(rho))
# dt = np.pi*(1/4+1/10000000)
# P = qt.tensor(sx, sx, sx,  sx, sx)
# rho_t = evolve(rho, P, dt)
# print(extract_spin_directions_from_rho(rho_t))

# # 1 check all spin configs and determine their type: min, max, or saddle
# all_states = generate_all_spin_states(N)
# for state in all_states:
#     type = check_rho_type(state, H, two_qubit_set)
#     print(type)
#     if type == 'local max':
#         print(extract_spin_directions_from_rho(state))


# 2 If the state is a saddle point or max, keep envolving it
# manual gradient descent

#(a) demonstration
# rho = create_spin_state(N,[0,1]) #sadle point
# p = qt.tensor([sx, sx, qt.qeye(2),qt.qeye(2),  qt.qeye(2)])
# rho= evolve(rho, p, np.pi/4)
# rho= evolve(rho, p, np.pi/4)
# gradient(rho,H, two_qubit_set)

# #(b) demonstration, the best path should be the one with greatest derivative
# rho = create_spin_state(N,[1,3]) #local maximum
# p = qt.tensor([qt.qeye(2), sx, qt.qeye(2),  sx, qt.qeye(2)])

# rho= evolve(rho, p, dt)
# print(extract_spin_directions_from_rho(rho))
# rho= evolve(rho, p, dt)
# print(extract_spin_directions_from_rho(rho))
# rho= evolve(rho, p, dt)
# print(extract_spin_directions_from_rho(rho))
# rho= evolve(rho, p, dt)
# print(extract_spin_directions_from_rho(rho))
# rho= evolve(rho, p, dt)
# print(extract_spin_directions_from_rho(rho))
# gradient(rho,H, two_qubit_set)



# #(c)
# states = generate_all_spin_states(N)
# for rho in states:
#     print(extract_spin_directions_from_rho(rho))
#     gradient(rho,H, two_qubit_set)


# 3 gradient descent

# rho = 1/2*create_spin_state(N,[1,3]) + 1/2*create_spin_state(N,[4])
rho = create_spin_state(N,[0,1])
p1 = qt.tensor([sx, sx, qt.qeye(2),qt.qeye(2),  qt.qeye(2)])
rho_1 = evolve(rho,p1,dt)
rho_3 = evolve(rho,p1,-dt)
# gradients = compute_gradient(rho_1, H, two_qubit_set)
# print(gradients)
rho_2 = create_spin_state(N,[1,3])
p2 = qt.tensor([qt.qeye(2), sx, qt.qeye(2),sx,  qt.qeye(2)])
rho_2 = evolve(rho_2,p2,dt)
# gradients = compute_gradient(rho_2, H, two_qubit_set)
# print(gradients)
rho = 1/2*rho_1 + 1/2*rho_2
# print(compute_gradient(rho_1, H, two_qubit_set),(rho_1*H).tr())
# print(compute_gradient(rho_3, H, two_qubit_set),(rho_3*H).tr())

# rho_1 = evolve(rho_1,p1,dt)
# rho_3 = evolve(rho_3,p1,-dt)
# print(compute_gradient(rho_1, H, two_qubit_set),(rho_1*H).tr())
# print(compute_gradient(rho_3, H, two_qubit_set),(rho_3*H).tr())
# rho_1 = evolve(rho_1,p1,dt)
# rho_3 = evolve(rho_3,p1,-dt)
# print(compute_gradient(rho_1, H, two_qubit_set),(rho_1*H).tr())
# print(compute_gradient(rho_3, H, two_qubit_set),(rho_3*H).tr())
# rho = rho_1
# for i in range(1):

#     gradients = compute_gradient(rho, H, two_qubit_set)
#     # print(gradients)
#     rho = optimize_rho(rho, gradients, two_qubit_set)
#     print((H*rho).tr().real)
#     # print(gradients)

# for i in range(1000):
#     gradients_1 = compute_gradient(rho_1, H, two_qubit_set)
#     rho_1 = optimize_rho(rho_1, gradients_1, two_qubit_set)
# print(energy(rho_1, H))

# for i in range(1000):
#     gradients_3 = compute_gradient(rho_3, H, two_qubit_set)
#     rho_3 = optimize_rho(rho_3,gradients_3,two_qubit_set)
# print(energy(rho_3,H))

for i in range(1000):
    gradients = compute_gradient(rho, H, two_qubit_set)
    rho = optimize_rho(rho,gradients,two_qubit_set)
    print(energy(rho,H))
    print(gradients)

