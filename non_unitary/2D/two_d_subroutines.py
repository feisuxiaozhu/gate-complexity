import qutip as qt
import numpy as np
import itertools
import time
from multiprocessing import Pool

def ising_2d_hamiltonian(M, N, J=1.0, hx=0, hz=0, periodic=False):
    num_sites = M * N  
    sx, sz, I = qt.sigmax(), qt.sigmaz(), qt.qeye(2)  
    def interaction_term(i, j):
        op_list = [I] * num_sites  
        op_list[i] = sz  
        op_list[j] = sz  
        return qt.tensor(op_list)
    def transverse_field_term_x(i):
        op_list = [I] * num_sites
        op_list[i] = sx  
        return qt.tensor(op_list)
    def transverse_field_term_z(i):
        op_list = [I] * num_sites
        op_list[i] = sz  
        return qt.tensor(op_list)
    H = 0
    for i in range(M):
        for j in range(N):
            site = i * N + j  
            # Interaction with right neighbor
            if j < N - 1 or periodic:
                neighbor = i * N + (j + 1) % N  # Right neighbor
                H += -J * interaction_term(site, neighbor)
            # Interaction with bottom neighbor (vertical)
            if i < M - 1 or periodic:
                neighbor = ((i + 1) % M) * N + j  # Bottom neighbor (wrap if periodic)
                H += -J * interaction_term(site, neighbor)
            # Transverse field term
            H += -hx * transverse_field_term_x(site)
            H += -hz * transverse_field_term_z(site)
    return qt.tensor(I,H)

def all_two_qubit_set_NN(M, N):
    num_sites = M * N  
    sx, sy, sz, I = qt.sigmax(), qt.sigmay(), qt.sigmaz(), qt.qeye(2)  
    pauli_ops = [sx, sy, sz]  
    one_qubit_gates = []
    two_qubit_gates = []
    gate_descriptions = []
    def single_site_operator(op, site):
        return qt.tensor([op if k == site else I for k in range(num_sites)])
    def two_site_operator(op1, site1, op2, site2):
        return qt.tensor([op1 if k == site1 else op2 if k == site2 else I for k in range(num_sites)])
    # Generate all single-qubit gates
    for site in range(num_sites):
        one_qubit_gates.append(qt.tensor(I,single_site_operator(sx, site)))  
        one_qubit_gates.append(qt.tensor(I,single_site_operator(sy, site)))  
        one_qubit_gates.append(qt.tensor(I,single_site_operator(sz, site)))  
        gate_descriptions.append(f"Single-qubit X gate on site {site}")
        gate_descriptions.append(f"Single-qubit Y gate on site {site}")
        gate_descriptions.append(f"Single-qubit Z gate on site {site}")

    # Generate all nearest-neighbor two-qubit gates
    for i in range(M):
        for j in range(N):
            site = i * N + j  
            # Interaction with right neighbor (horizontal)
            if j < N - 1:
                neighbor = i * N + (j + 1)
                for op1, label1 in zip(pauli_ops, ["X", "Y", "Z"]):
                    for op2, label2 in zip(pauli_ops, ["X", "Y", "Z"]):
                        two_qubit_gates.append(qt.tensor(I,two_site_operator(op1, site, op2, neighbor)))
                        gate_descriptions.append(f"Two-qubit {label1} ⊗ {label2} gate between site {site} and site {neighbor}")
            # Interaction with bottom neighbor (vertical)
            if i < M - 1:
                neighbor = (i + 1) * N + j
                for op1, label1 in zip(pauli_ops, ["X", "Y", "Z"]):
                    for op2, label2 in zip(pauli_ops, ["X", "Y", "Z"]):
                        two_qubit_gates.append(qt.tensor(I,two_site_operator(op1, site, op2, neighbor)))
                        gate_descriptions.append(f"Two-qubit {label1} ⊗ {label2} gate between site {site} and site {neighbor}")

    return one_qubit_gates + two_qubit_gates, gate_descriptions


def ancilla_two_qubit_set(M,N):
    # Two-qubit connecting ancilla and normal qubit
    # We chose only connect the first qubit of each row to the ancilla
    operators = []
    ancilla_gates = [qt.sigmax(), qt.sigmay()]
    single_qubit_gates = [qt.sigmax(), qt.sigmay(),qt.sigmaz()] 
    for ancilla_gate in ancilla_gates:
        for single_gate in single_qubit_gates:
            for i in range(M):
                op_list = [qt.qeye(2) for _ in range(M*N)]
                op_list[i*N] = single_gate
                ancilla_op = qt.tensor([ancilla_gate] + op_list)
                operators.append(ancilla_op)

    return operators

def generate_spin_state(M, N, state_type="up", custom_state=None):
    num_sites = M * N  
    up = qt.basis(2, 0)  
    down = qt.basis(2, 1)  

    if state_type == "up":
        state = qt.tensor([up] * num_sites)
    elif state_type == "down":
        state = qt.tensor([down] * num_sites)
    elif state_type == "custom":
        if custom_state is None or len(custom_state) != num_sites:
            raise ValueError(f"custom_state must be a list of length {num_sites} containing 0 (up) and 1 (down)")
        state = qt.tensor([up if bit == 0 else down for bit in custom_state])
    else:
        raise ValueError("Invalid state_type. Choose from 'up', 'down', or 'custom'.")
    rho = state.proj()
    zero = qt.basis(2, 0).proj()
    return qt.tensor(zero, rho)




def generate_all_spin_states(M,N):
    spin_states = []
    for bits in itertools.product([0, 1], repeat=M*N):
        # Create a tensor product state for the spin configuration
        state = qt.tensor([qt.basis(2, b) for b in bits])
        rho = state.proj()
        zero = qt.basis(2, 0).proj()

        spin_states.append(qt.tensor(zero,rho))
    
    return spin_states


def energy(rho, H):
    return (rho*H).tr().real

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

def compute_gradient(rho, H, gateset):
    gradients = []
    for p in gateset:
        gradients.append(first_derivative(rho,H, p))
    return gradients


def compute_hessian(rho, H, gateset):
    d = len(gateset)
    K = np.zeros((d, d))
    # Store adjoint operators as a list (avoiding dictionary hash issues)
    adj_rho = [adj(P, rho) for P in gateset]
    for j in range(d):
        P_j = gateset[j]
        adj_Pj_rho = adj_rho[j]
        for k in range(j, d):  # Exploit symmetry
            P_k = gateset[k]
            adj_Pk_rho = adj_rho[k]
            adj_Pk_Pj_rho = adj(P_k, adj_Pj_rho)
            Kjk = -0.5 * (adj_Pk_Pj_rho * H + adj(P_j, adj_Pk_rho) * H).tr().real
            K[j, k] = Kjk
            K[k, j] = Kjk
    return K

def compute_hessian_diagonal(rho, H, gateset):
    d = len(gateset)
    K = np.zeros((d,d))
    for j in range(d):
        P_j = gateset[j]
        P_k = gateset[j]
        Kjk = -1/2*(adj(P_k, adj(P_j, rho))*H+adj(P_j, adj(P_k, rho))*H).tr().real
        K[j][j] = Kjk
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
    Hessian = compute_hessian_diagonal(rho, H, gateset)
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
  
    epsilon_col = Q @ epsilon_col *10
    P = qt.tensor([qt.qzero(2) for _ in range(num_qubits)])
    for i in range(len(gradients)):
        P += (gradients[i]+epsilon_col[i])* gateset[i]
    rho = evolve(rho, P, -dt)
    return rho

def driver(rho_tilde,H_tilde,two_qubit_set_tilde,ancilla_two_qubit_set_tilde ,dt):
    for i in range(1000):
        if i%100==0:
            print('iteration: '+ str(i))
        # time_in = time.time()
        gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
        # time_final = time.time()
        # print('time to find gradient: ' + str(time_final-time_in))
        rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
        gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
        rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
        # time_in = time.time()
        # rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
        # time_final = time.time()
        # print('time for 1step ancilla update: ' + str(time_final-time_in))
        # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
        rho = trace_out_rho_tilde(rho_tilde)
        rho_tilde = rho_to_rho_tilde(rho)

        E = energy(rho_tilde, H_tilde)
        
        # print(E)
    return (E, np.linalg.norm(gradients))


