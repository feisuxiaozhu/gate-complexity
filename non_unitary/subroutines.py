import qutip as qt
import numpy as np

sz = qt.sigmaz()
sx = qt.sigmax()

# N.N. Ising iteraction Hamiltonian
def NN_H(N):
    H = 0
    for i in range(N - 1):
        op_list = [qt.qeye(2) for _ in range(N)]
        op_list[i] = sz
        op_list[i + 1] = sz
        H += -qt.tensor(op_list)
    return H

# Generate a set of all possible two-qubit Paulis
# For now, we use X⊗X and X⊗I only
def two_qubit_set(N):
    operators = []
    for i in range(N):
        op_list = [qt.qeye(2) for _ in range(N)]  # Start with identity for all sites
        op_list[i] = qt.sigmax()  # Apply X to the i-th site
        single_x = qt.tensor(op_list)
        operators.append(single_x)
    for i in range(N - 1):
        op_list = [qt.qeye(2) for _ in range(N)]  # Start with identity for all sites
        op_list[i] = qt.sigmax()       # Apply X to the i-th site
        op_list[i + 1] = qt.sigmax()   # Apply X to the (i+1)-th site
        two_xx = qt.tensor(op_list)
        operators.append(two_xx)
    return operators

# Create a quantum state of length N with spins up at locations in set A
def create_spin_state(N, A):
    spin_up = qt.basis(2, 0)   
    spin_down = qt.basis(2, 1) 
    spin_states = []
    for i in range(N):
        if i in A:
            spin_states.append(spin_up)
        else:
            spin_states.append(spin_down)
    state = qt.tensor(spin_states)
    return state

# Compute the first time derivative of Tr(exp(-iPt)*rho*exp(iPt)*H)
def first_derivative(H,rho, P):
    # Compute commutator [-iP, rho]
    commutator = -1j * (P * rho - rho * P)
    return((commutator * H).tr())

# evolve the state rho with P for time increment dt
def evolve(rho, P, dt):
    U_t = (-1j * P * dt).expm()
    U_t_dag = (1j * P * dt).expm()
    rho_t = U_t * rho * U_t_dag
    return rho_t

def energy(rho, H):
    return (rho*H).tr()