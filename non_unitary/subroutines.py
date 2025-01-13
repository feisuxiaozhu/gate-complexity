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

def Z_H(N):
    H = 0
    for i in range(N):
        op_list = [qt.qeye(2) for _ in range(N)]
        op_list[i] = sz
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

# Two qubit operator on nearest neightbor
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
            single_op = qt.tensor(op_list)
            operators.append(single_op)

    # Two-qubit operators (only on neighboring qubits)
    for i in range(N - 1):  # Restrict to neighbors
        for gate_pair in two_qubit_gates:
            op_list = [qt.qeye(2) for _ in range(N)]
            op_list[i] = gate_pair[0]
            op_list[i + 1] = gate_pair[1]
            two_qubit_op = qt.tensor(op_list)
            operators.append(two_qubit_op)

    return operators

# Include all possible gates, not only restricted to NN operators
def all_two_qubit_set_complete(N):
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
            single_op = qt.tensor(op_list)
            operators.append(single_op)
    
    # Two-qubit operators (allowing non-neighboring qubits)
    for i in range(N):
        for j in range(i + 1, N):
            for gate_pair in two_qubit_gates:
                op_list = [qt.qeye(2) for _ in range(N)]
                op_list[i] = gate_pair[0]
                op_list[j] = gate_pair[1]
                two_qubit_op = qt.tensor(op_list)
                operators.append(two_qubit_op)
    
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


def adj(P, rho):
    return (P * rho - rho * P)

# Compute the first time derivative of Tr(exp(-iPt)*rho*exp(iPt)*H) at t=0
def first_derivative(rho,H, P):
    # Compute commutator [-iP, rho]
    # commutator = -1j * (P * rho - rho * P)
    commutator = -1j*adj(P, rho)
    result = (commutator * H).tr().real
    return(result)

# evolve the state rho with P for time increment dt
def evolve(rho, P, dt):
    U_t = (-1j * P * dt).expm()
    U_t_dag = (1j * P * dt).expm()
    rho_t = U_t * rho * U_t_dag
    return rho_t

def energy(rho, H):
    return (rho*H).tr().real

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


def extract_pure_state(rho): #bad code don't use
    # Validate that rho has trace 1
    

    # Get the eigenstates of rho
    eigenvalues, eigenstates = rho.eigenstates()

    # Extract the eigenstate with eigenvalue ~1
    for eigval, eigvec in zip(eigenvalues, eigenstates):
        if abs(eigval - 1) < 1e-10:  # Tolerance for numerical precision
            return eigvec


def extract_spin_directions_from_rho(rho):

    # Check if the density matrix is pure: ρ^2 = ρ
    if not np.isclose((rho * rho - rho).norm(), 0):
        raise ValueError("The input density matrix is not a pure state.")
    
    # Extract the dominant eigenstate of the density matrix
    eigenvalues, eigenstates = rho.eigenstates()
    index = np.argmax(eigenvalues)  # Find the eigenvector with eigenvalue ~1
    psi = eigenstates[index]  # Corresponding pure state vector |ψ⟩

    # Find the spin directions
    state_vector = psi.full().flatten()
    index = np.argmax(np.abs(state_vector))  # Find the index of the largest amplitude

    # Convert index to binary representation
    num_qubits = int(np.log2(psi.shape[0]))
    binary_rep = format(index, f"0{num_qubits}b")  # Binary string representing the state
    # Interpret binary representation as spin directions

    spin_directions = ["↑" if b == "0" else "↓" for b in binary_rep]
    beauty = "|"+"".join(spin_directions)+">"

    return beauty


def compute_gradient(rho, H, gateset):
    gradients = []
    for p in gateset:
        gradients.append(first_derivative(rho,H, p))
    return gradients


# Output new parameter rho 
def optimize_rho(rho, gradients, gateset):
    dt = np.pi/100
    num_qubits = len(gateset[0].dims[0])
    P = qt.tensor([qt.qzero(2) for _ in range(num_qubits)])
    for i in range(len(gradients)):
            P += gradients[i]* gateset[i]
    rho = evolve(rho, P, -dt)
    return rho

# Output new parameter rho with noise added to gradient
def optimize_rho_with_noise(rho, gradients, gateset):
    dt = np.pi/100
    num_qubits = len(gateset[0].dims[0])
    P = qt.tensor([qt.qzero(2) for _ in range(num_qubits)])
    for i in range(len(gradients)):
            epsilon = np.random.normal(0,np.sqrt(dt))
            P += (gradients[i]+epsilon)* gateset[i]
    rho = evolve(rho, P, -dt)
    return rho


def optimize_rho_noise_dynamicdt(rho, gradients, gateset,dt):
    num_qubits = len(gateset[0].dims[0])
    P = qt.tensor([qt.qzero(2) for _ in range(num_qubits)])
    for i in range(len(gradients)):
            # if i> 50000:
            #     dt = np.pi/10000
            # if i> 100000:
            #     dt = np.pi/1000000
            # if i>150000:
            #     dt = np.pi/100000000
            epsilon = np.random.normal(0,np.sqrt(dt))
            P += (gradients[i])* gateset[i]

    rho = evolve(rho, P, -dt)
    return rho


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
        epsilon = np.random.normal(0,np.sqrt(dt))/1000
        P += (gradients[i]+epsilon)* gateset[i]
    rho = evolve(rho, P, -dt)
    return rho

def optimizer_1step_SGD_dt_scheduling(rho, gradients, gateset, dt0, round):
    dt = dt0/100
    # if round > 5000:
    #     dt = dt0/10
    # if round > 10000:
    #     dt=dt0/100
    # if round > 15000:
    #     dt = dt0/1000
    num_qubits = len(gateset[0].dims[0])
    P = qt.tensor([qt.qzero(2) for _ in range(num_qubits)])
    for i in range(len(gradients)):
        epsilon = np.random.normal(0,np.sqrt(dt))
        P += (gradients[i]+epsilon)* gateset[i]
    rho = evolve(rho, P, -dt)
    return rho

def optimizer_1step_SGD_noise_scheduling(rho, gradients, gateset, dt0, round):
    dt = dt0
    num_qubits = len(gateset[0].dims[0])
    P = qt.tensor([qt.qzero(2) for _ in range(num_qubits)])
    for i in range(len(gradients)):
        epsilon = np.random.normal(0,np.sqrt(dt))
        # epsilon = np.sign(epsilon)* np.sqrt(np.abs(epsilon))
        epsilon = epsilon/1000
        P += (gradients[i]+epsilon)* gateset[i]
    rho = evolve(rho, P, -dt)
    return rho


def optimizer_1step_SGD_hessian(rho, gradients, gateset, dt0, H):
    tolerance=1e-10
    dt = dt0
    num_qubits = len(gateset[0].dims[0])
    Hessian = compute_hessian(rho, H, gateset)
    eigenvalues, eigenvectors = np.linalg.eigh(Hessian)
    eigenvalues[np.abs(eigenvalues) < tolerance] = 0
    D = np.diag(eigenvalues)
    Q = eigenvectors
    # residual = Hessian - Q @ D @ Q.T
    # print(np.allclose(residual, np.zeros_like(Hessian), atol=1e-8)) 
    eigenvalues = eigenvalues.real
    negative_indices = np.where(eigenvalues < 0)[0]
    # print(eigenvalues)
    # print(negative_indices)
    epsilon_col = [0 for _ in range(len(gradients))]
    for index in negative_indices:
        epsilon = np.random.normal(0,np.sqrt(dt))
        epsilon_col[index] = epsilon
    epsilon_col = Q @ epsilon_col

    P = qt.tensor([qt.qzero(2) for _ in range(num_qubits)])
    for i in range(len(gradients)):
        P += (gradients[i]+epsilon_col[i])* gateset[i]
    rho = evolve(rho, P, -dt)
    return rho


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


def find_non_symmetric_indices(matrix):
    non_symmetric_indices = []
    rows, cols = matrix.shape

    # Check for equality only in the upper triangular part
    for i in range(rows):
        for j in range(i + 1, cols):  # Avoid checking the diagonal
            if not np.isclose(matrix[i, j], matrix[j, i]):  # Use isclose for floating-point comparison
                non_symmetric_indices.append((i, j))

    return non_symmetric_indices

def is_positive_semi_definite(matrix):
    tolerance=1e-10
    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        print("Matrix is not symmetric.")
        return False
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(matrix)
    # Set eigenvalues close to zero to exactly zero
    eigenvalues[np.abs(eigenvalues) < tolerance] = 0 
    # print(eigenvalues)
    # Check if all eigenvalues are non-negative
    return np.all(eigenvalues >= 0)