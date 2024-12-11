import qutip as qt
import numpy as np

# System size
N = 5  # Number of spins in the chain

# Define Pauli-Z operator
sz = qt.sigmaz()

# Initialize Hamiltonian
H = 0  # Start with a zero Hamiltonian

# Construct the Ising Hamiltonian with nearest-neighbor interactions
for i in range(N - 1):
    # Create identity operators for all sites
    op_list = [qt.qeye(2) for _ in range(N)]
    # Apply Pauli-Z to the i-th and (i+1)-th sites
    op_list[i] = sz
    op_list[i + 1] = sz
    print(op_list)
    # Add the term to the Hamiltonian
    H += -qt.tensor(op_list)

