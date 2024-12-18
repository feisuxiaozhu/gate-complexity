import qutip as qt
import numpy as np


# Define the Hamiltonian (Ising + transverse field for asymmetry)
sz = qt.sigmaz()
sx = qt.sigmax()
N = 5  # Number of spins

# Nearest-neighbor Ising interaction
H = 0
for i in range(N - 1):
    op_list = [qt.qeye(2) for _ in range(N)]
    op_list[i] = sz
    op_list[i + 1] = sz
    H += -qt.tensor(op_list)

# Add transverse field to break symmetry
hx = 0.5  # Field strength
for i in range(N):
    op_list = [qt.qeye(2) for _ in range(N)]
    op_list[i] = sx
    H += hx * qt.tensor(op_list)

# Define initial state |↑↑↓↓↓⟩
spin_up = qt.basis(2, 0)   # |↑⟩
spin_down = qt.basis(2, 1) # |↓⟩
state = qt.tensor(spin_up, spin_up, spin_down, spin_down, spin_down)
# state = (qt.tensor(spin_up, spin_up, spin_down, spin_down, spin_down) +
#          qt.tensor(spin_down, spin_down, spin_up, spin_up, spin_up)).unit()
rho = state.proj()



# Define a two-qubit Pauli operator (e.g., X⊗X on qubits 1 and 2)
P = qt.tensor(sx, sx,qt.qeye(2),  qt.qeye(2), qt.qeye(2))

print("initial energy:", (rho*H).tr())

dt = np.pi/10
U_t = (-1j * P * dt).expm()  # e^(-i P t)
U_t_dag = (1j * P * dt).expm()  # e^(i P t)
rho = U_t * rho * U_t_dag

# Compute commutator [-iP, rho]
commutator = -1j * (P * rho - rho * P)
# print("Commutator [-iP, ρ]:\n", commutator)
# Compute the derivative: Tr([-iP, rho] H)
result = (commutator * H).tr()

# Output the result
print("Derivative at t=0:", result)
print("final eneryg:", (rho*H).tr())
