import qutip as qt
import numpy as np
from subroutines import *

N=4

H = NN_H(N)
two_qubit_set = two_qubit_set(N)

state = create_spin_state(N,[0])
rho = state.proj()


p = two_qubit_set[0]
print(first_derivative(H, rho, p))
print(energy(rho,H))
dt = np.pi/2
rho_t = evolve(rho,p,dt)
print(first_derivative(H, rho_t, p))
print(energy(rho_t,H))