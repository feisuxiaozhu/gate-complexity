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


