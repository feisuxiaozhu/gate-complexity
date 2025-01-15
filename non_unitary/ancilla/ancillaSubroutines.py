import qutip as qt
import numpy as np
import itertools

sz = qt.sigmaz()
sx = qt.sigmax()


def NN_H(N):
    H = 0
    for i in range(N - 1):
        op_list = [qt.qeye(2) for _ in range(N)]
        op_list[i] = sz
        op_list[i + 1] = sz
        identity = [qt.qeye(2)]
        new_list = identity + op_list
        H += -qt.tensor(new_list)
    return H


def all_two_qubit_set_NN(N):
    operators = []
    single_qubit_gates = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]  # Single-qubit X, Y, Z gates
    two_qubit_gates = [(qt.sigmax(), qt.sigmax()), 
                       (qt.sigmax(), qt.sigmay()), 
                       (qt.sigmax(), qt.sigmaz()),
                       (qt.sigmay(), qt.sigmax()), 
                       (qt.sigmay(), qt.sigmay()), 
                       (qt.sigmay(), qt.sigmaz()),
                       (qt.sigmaz(), qt.sigmax()), 
                       (qt.sigmaz(), qt.sigmay()), 
                       (qt.sigmaz(), qt.sigmaz())]  # All two-qubit combinations

    # Single-qubit operators
    for i in range(N):
        for single_gate in single_qubit_gates:
            op_list = [qt.qeye(2) for _ in range(N)]
            op_list[i] = single_gate
            single_op = qt.tensor([qt.qeye(2)]+op_list)
            operators.append(single_op)

    # Two-qubit operators (only on neighboring qubits)
    for i in range(N - 1):  # Restrict to neighbors
        for gate_pair in two_qubit_gates:
            op_list = [qt.qeye(2) for _ in range(N)]
            op_list[i] = gate_pair[0]
            op_list[i + 1] = gate_pair[1]
            two_qubit_op = qt.tensor([qt.qeye(2)]+op_list)
            operators.append(two_qubit_op)

    return operators

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
    rho = state.proj()
    zero = qt.basis(2, 0).proj()
    return qt.tensor(zero, rho)


def trace_out(rho):
    N = len(rho.dims[0])
    rho_reduced = qt.ptrace(rho, list(range(1,N)))
    return rho_reduced

def rho_tilde(rho):
    zero = qt.basis(2, 0).proj()
    return qt.tensor(zero, rho)