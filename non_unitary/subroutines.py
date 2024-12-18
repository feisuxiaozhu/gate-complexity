import qutip as qt
import numpy as np
import itertools

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
        op_list = [qt.qeye(2) for _ in range(N)]  
        op_list[i] = qt.sigmax()  
        single_x = qt.tensor(op_list)
        operators.append(single_x)

    # Two-qubit X⊗X operators (allowing non-neighbors)
    for i in range(N):
        for j in range(i + 1, N):  
            op_list = [qt.qeye(2) for _ in range(N)]  
            op_list[i] = qt.sigmax()  
            op_list[j] = qt.sigmax()  
            operators.append(qt.tensor(op_list))
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
    return state.proj()

def generate_all_spin_states(N):
    spin_states = []
    for bits in itertools.product([0, 1], repeat=N):
        # Create a tensor product state for the spin configuration
        state = qt.tensor([qt.basis(2, b) for b in bits])
        spin_states.append(state.proj())
    
    return spin_states


# Compute the first time derivative of Tr(exp(-iPt)*rho*exp(iPt)*H)
def first_derivative(rho,H, P):
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

#decide if rho is local min, max, or a saddle point.
def check_rho_type(rho, H, gateset):
    flags = [0,0,0] # flags for 0, <0, >0 first derivatives 
    dt = np.pi/10
    for p in gateset:
        rho_t = evolve(rho,p,dt)
        d = first_derivative(rho_t, H, p).real
        if d == 0:
            flags[0]=1
        elif d<0:
            flags[1]=1
        else:
            flags[2]=1
    result = ''
    if flags[1]+flags[2] == 2:
        result = 'saddle point'
    elif flags[1] == 1:
        result = 'local max'
    elif flags[2] == 1:
        result = 'local min'
    return result






