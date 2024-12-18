import qutip as qt
import numpy as np

# System size
N = 5  # Number of spins in the chain

# Define Pauli-X operator
sx = qt.sigmax()

# Initialize Hamiltonian
H = 0  # Start with a zero Hamiltonian

# Construct the Ising Hamiltonian with nearest-neighbor interactions
for i in range(N - 1):
    # Create identity operators for all sites
    op_list = [qt.qeye(2) for _ in range(N)]
    # Apply Pauli-Z to the i-th and (i+1)-th sites
    op_list[i] = sx
    op_list[i + 1] = sx
    # Add the term to the Hamiltonian
    H += -qt.tensor(op_list)

# Define the configuration |↑↑↑↓↓⟩ = |00011⟩
spin_up = qt.basis(2, 0)   # |↑⟩ or |0⟩
spin_down = qt.basis(2, 1) # |↓⟩ or |1⟩

# Tensor product to create the state |00011⟩
state = qt.tensor(spin_up, spin_up, spin_down, spin_down, spin_down)

# Create the density matrix ρ = |ψ⟩⟨ψ|
rho = state.proj()

# Define a two-qubit Pauli operator (e.g., X ⊗ Z ⊗ I ⊗ I ⊗ I)
P = qt.tensor(qt.sigmax(), qt.sigmax(), qt.qeye(2),qt.qeye(2),qt.qeye(2))

# Time parameter
t = 1.0  # Set t to any desired value

# Compute e^(-i P t) and e^(i P t)
U_t = (-1j * P * t).expm()  # e^(-i P t)
U_t_dag = (1j * P * t).expm()  # e^(i P t)

# Compute the evolved density matrix
rho_t = U_t * rho * U_t_dag

# Compute the trace of Tr(rho_t * H)
result = (rho_t * H).tr()

# Compute the commutator [-i P, rho]
commutator = -1j * (P * rho - rho * P)

# Compute Tr(commutator * H)
result = (commutator * H).tr()

# Print the result
print("Derivative at t=0:", result)

