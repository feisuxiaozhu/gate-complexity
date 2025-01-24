import qutip as qt
import numpy as np

def H_PXP_tilde(N):
    sz = qt.sigmaz()
    sx = qt.sigmax()
    P = (qt.qeye(2) + sz) / 2
    H = 0
    for i in range(N):
        # Periodic boundary indices
        i_minus_1 = (i - 1) % N
        i_plus_1 = (i + 1) % N
        operators = [qt.qeye(2)] * N
        operators[i_minus_1] = P
        operators[i] = sx
        operators[i_plus_1] = P

        new_list = [qt.qeye(2)] + operators
        H += qt.tensor(new_list)
    return H

def all_two_qubit_set_NN(N):
    operators = []
    single_qubit_gates = [qt.sigmax(), qt.sigmay(),qt.sigmaz()]  # Single-qubit X, Y, Z gates
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

def ancilla_two_qubit_set(N):
    # Two-qubit connecting ancilla and normal qubit
    operators = []
    ancilla_gates = [qt.sigmax(), qt.sigmay()]
    single_qubit_gates = [qt.sigmax(), qt.sigmay(),qt.sigmaz()] 
    for ancilla_gate in ancilla_gates:
        for single_gate in single_qubit_gates:
            for i in range(N):
                op_list = [qt.qeye(2) for _ in range(N)]
                op_list[i] = single_gate
                ancilla_op = qt.tensor([ancilla_gate] + op_list)
                operators.append(ancilla_op)

    return operators

# Assume N even
def create_zero_minus_state(N):
    if N % 2 != 0:
        raise ValueError("N must be even!")

    ket_0 = qt.basis(2, 0)  # |0⟩
    ket_minus = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
    state = qt.tensor(ket_0, ket_minus)
    for i in range(2, N, 2): 
        state = qt.tensor(state, ket_0, ket_minus)
    rho = state.proj()
    zero = qt.basis(2, 0).proj()
    return qt.tensor(zero, rho)


def create_minus_zero_state(N):
    if N % 2 != 0:
        raise ValueError("N must be even!")

    ket_0 = qt.basis(2, 0)  # |0⟩
    ket_minus = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
    state = qt.tensor(ket_minus, ket_0)
    for i in range(2, N, 2): 
        state = qt.tensor(state, ket_minus, ket_0)
    rho = state.proj()
    zero = qt.basis(2, 0).proj()
    return qt.tensor(zero, rho)

def create_CDW_state(N):
    if N % 2 != 0:
        raise ValueError("N must be even to create a perfect CDW state.")
    cdw_basis_states = [qt.basis(2, i % 2) for i in range(N)]
    cdw_state = qt.tensor(cdw_basis_states)
    rho = cdw_state.proj()
    zero = qt.basis(2, 0).proj()
    return qt.tensor(zero, rho)

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

def trace_out_rho_tilde(rho):
    N = len(rho.dims[0])
    rho_reduced = qt.ptrace(rho, list(range(1,N)))
    return rho_reduced

def rho_to_rho_tilde(rho):
    zero = qt.basis(2, 0).proj()
    return qt.tensor(zero, rho)

def adj(P, rho):
    return (P * rho - rho * P)

def first_derivative(rho,H, P):
    commutator = -1j*adj(P, rho)
    result = (commutator * H).tr().real
    return(result)

def evolve(rho, P, dt):
    U_t = (-1j * P * dt).expm()
    U_t_dag = (1j * P * dt).expm()
    rho_t = U_t * rho * U_t_dag
    return rho_t

def energy(rho, H):
    return (rho*H).tr().real

def compute_gradient(rho, H, gateset):
    gradients = []
    for p in gateset:
        gradients.append(first_derivative(rho,H, p))
    return gradients

def compute_hessian(rho, H, gateset):
    d = len(gateset)
    K = np.zeros((d,d))
    for j in range(d):
        for k in range(d):
            P_j = gateset[j]
            P_k = gateset[k]
            Kjk = -1/2*(adj(P_k, adj(P_j, rho))*H+adj(P_j, adj(P_k, rho))*H).tr().real
            K[j][k] = Kjk
    return K

def optimizer_1step_pure_GD(rho, gradients, gateset, dt):
    num_qubits = len(gateset[0].dims[0])
    P = qt.tensor([qt.qzero(2) for _ in range(num_qubits)])
    for i in range(len(gradients)):
        P += (gradients[i])* gateset[i]
    rho = evolve(rho, P, -dt)
    return rho

def optimizer_1step_SGD_no_scheduling(rho, gradients, gateset, dt):
    num_qubits = len(gateset[0].dims[0])
    P = qt.tensor([qt.qzero(2) for _ in range(num_qubits)])
    for i in range(len(gradients)):
        epsilon = np.random.normal(0,np.sqrt(dt))
        P += (gradients[i]+epsilon)* gateset[i]
    rho = evolve(rho, P, -dt)
    return rho

def optimizer_1step_SGD_ancilla_no_scheduling(rho, ancilla_gateset, dt0, H):
    tolerance=1e-10
    dt=np.sqrt(dt0)
    num_qubits = len(ancilla_gateset[0].dims[0])
    
    Hessian = compute_hessian(rho, H, ancilla_gateset)

    eigenvalues, eigenvectors = np.linalg.eigh(Hessian)
    eigenvalues[np.abs(eigenvalues) < tolerance] = 0
    Q = eigenvectors
    
    negative_indices = np.where(eigenvalues < 0)[0]
    second_derivatives = [0 for _ in range(len(eigenvectors[0]))]
    
    for index in negative_indices:
        second_derivatives[index] = eigenvalues[index]

    second_derivatives = Q @ second_derivatives

    P = qt.tensor([qt.qzero(2) for _ in range(num_qubits)])
    for i in range(len(second_derivatives)):
        P += second_derivatives[i]* ancilla_gateset[i]
    rho = evolve(rho, P, dt)
    return rho, second_derivatives

def optimizer_1step_SGD_hessian(rho, gradients, gateset, dt0, H):
    tolerance=1e-10
    dt = dt0
    num_qubits = len(gateset[0].dims[0])
    Hessian = compute_hessian(rho, H, gateset)
    eigenvalues, eigenvectors = np.linalg.eigh(Hessian)
    eigenvalues[np.abs(eigenvalues) < tolerance] = 0
    D = np.diag(eigenvalues)
    Q = eigenvectors
    eigenvalues = eigenvalues.real
    negative_indices = np.where(eigenvalues < 0)[0]
    epsilon_col = [0 for _ in range(len(gradients))]
    for index in negative_indices:
        epsilon = np.random.normal(0,np.sqrt(dt))
        epsilon_col[index] = epsilon
  
    epsilon_col = Q @ epsilon_col *10

    P = qt.tensor([qt.qzero(2) for _ in range(num_qubits)])
    for i in range(len(gradients)):
        P += (gradients[i]+epsilon_col[i])* gateset[i]
    rho = evolve(rho, P, -dt)
    return rho

def compute_overlap_with_ground_state(H, rho):
    if not H.isherm:
        raise ValueError("Hamiltonian H must be Hermitian.")
    if not rho.isoper:
        raise ValueError("rho must be a density matrix.")
    eigenenergies, eigenstates = H.eigenstates()
    ground_state = eigenstates[0]
    rho_tilde = rho_to_rho_tilde(rho)
    rho_gs = ground_state.proj()
    overlap = (rho_gs*rho).tr().real

    return overlap


def decompose_into_product_state(state):
    # Ensure input is a ket or density matrix
    if not (state.isket or state.isoper):
        raise ValueError("Input state must be a ket or a density matrix.")

    # If the state is a density matrix, convert it to a pure state (if possible)
    if state.isoper:
        purity = state.purity()
        if purity < 1 - 1e-10: 
            raise ValueError("The state is not pure, cannot decompose into a product state.")
        state = state.eigenstates()[1][0]  # Extract the dominant eigenstate (pure state)

    # Get the dimensions of the subsystems
    dims = state.dims[0]
    if len(dims) < 2:
        raise ValueError("The state it not a multipartite system.")

    # Decompose into subsystems
    subsystem_states = []
    for i in range(len(dims)):
        reduced_rho = state.ptrace(i)
        if reduced_rho.purity() < 1 - 1e-10:
            return None  
        subsystem_states.append(reduced_rho)

    # Return the list of subsystem states
    return subsystem_states