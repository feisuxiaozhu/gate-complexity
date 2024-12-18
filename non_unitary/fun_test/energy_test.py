import qutip as qt
import numpy as np

sx = qt.sigmaz()
H = 0
N = 5
for i in range(N - 1):
    # Create identity operators for all sites
    op_list = [qt.qeye(2) for _ in range(N)]
    # Apply Pauli-Z to the i-th and (i+1)-th sites
    op_list[i] = sx
    op_list[i + 1] = sx
    # Add the term to the Hamiltonian
    H += -qt.tensor(op_list)

spin_up = qt.basis(2, 0)   # |↑⟩ or |0⟩
spin_down = qt.basis(2, 1) # |↓⟩ or |1⟩

state_1 = qt.tensor(spin_up, spin_up, spin_down, spin_down, spin_down)
state_2 = qt.tensor(spin_down, spin_down, spin_down, spin_down, spin_down)


rho_1 = state_1.proj()
rho_2 = state_2.proj()


E_1 = (rho_1*H).tr()
E_2 = (rho_2*H).tr()

P = qt.tensor(qt.sigmax(), qt.sigmax(), qt.qeye(2),qt.qeye(2),qt.qeye(2))
# Q = qt.tensor( qt.qeye(2),qt.qeye(2),qt.sigmax(), qt.sigmax(),  qt.sigmax())
t=0
while t<np.pi:
    t += np.pi/1000
    U_t = (-1j * P * t).expm()  # e^(-i P t)
    U_t_dag = (1j * P * t).expm()  # e^(i P t)
    rho_t = U_t * rho_1 * U_t_dag
    print((rho_t*H).tr())


# state_3 = Q*state_1
# rho_3 = state_3.proj()
# E_3= (rho_3*H).tr()
# print(E_3)


# commutator = -1j * (P * rho_1 - rho_1 * P)
# print(commutator)
# result = (commutator * H).tr()

# print("Derivative at t=0:", result)
