import qutip as qt
import numpy as np
from subroutines import *

N=5

H = NN_H(N)
two_qubit_set = two_qubit_set(N)

state = create_spin_state(N,[1,3])
rho = state
dt = np.pi/10
p = qt.tensor(sx, sx, sx,  sx, sx)

print(check_rho_type(rho, H, two_qubit_set))

