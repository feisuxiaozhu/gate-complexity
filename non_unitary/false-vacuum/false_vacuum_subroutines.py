import qutip as qt
import numpy as np
import itertools
import matplotlib.pyplot as plt

def H_TFIM(N,hx,hz):
    si = qt.qeye(2)  
    sx = qt.sigmax() 
    sz = qt.sigmaz()
    H=0
    for i in range(N):
        # Nearest neighbor interaction sigma_z^i sigma_z^(i+1)
        if i < N - 1:
            H += -qt.tensor([sz if j == i or j == i + 1 else si for j in range(N)])
        H += -hx * qt.tensor([sx if j == i else si for j in range(N)])
        H += -hz * qt.tensor([sz if j == i else si for j in range(N)])
    
    return qt.tensor(si,H)

def rydberg_hamiltonian_periodic(N, Omega, C6, r0, Delta_glob,Delta_loc):
    if N % 2 != 0:
        raise ValueError("N must be even for periodic boundary conditions.")

    sx, sz, I = qt.sigmax(), qt.sigmaz(), qt.qeye(2)
    n_op = (I - sz) / 2
    V_nn = C6 / (r0 ** 6)       # Nearest-neighbor interaction 
    V_nnn = C6 / ((2 * r0) ** 6)  # Next-nearest-neighbor interaction
    H = 0
    # First term
    for j in range(N):
        H += (Omega / 2) * qt.tensor([sx if k == j else I for k in range(N)])
    # Second term: Global detuning term 
    for j in range(N):
        H -= (Delta_glob+ (-1)**j*Delta_loc) * qt.tensor([n_op if k == j else I for k in range(N)])
    # Third term: Interaction term with periodic boundary conditions
    for j in range(N):
        k_nn = (j + 1) % N  # Nearest neighbor
        k_nnn = (j + 2) % N  # Next-nearest neighbor
        # Nearest-neighbor interaction
        H += V_nn * qt.tensor([
            n_op if site == j or site == k_nn else I for site in range(N)
        ])
        # Next-nearest-neighbor interaction
        H += V_nnn * qt.tensor([
            n_op if site == j or site == k_nnn else I for site in range(N)
        ])
    return qt.tensor(I,H)

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


def create_spin_state(N, A):
    spin_up = qt.basis(2, 0)   
    spin_down = qt.basis(2, 1) 
    spin_states = []
    for i in range(N):
        if i in A:
            spin_states.append(spin_down)
        else:
            spin_states.append(spin_up)
    state = qt.tensor(spin_states)
    rho = state.proj()
    zero = qt.basis(2, 0).proj()
    return qt.tensor(zero, rho)

def generate_all_spin_states(N):
    spin_states = []
    for bits in itertools.product([0, 1], repeat=N):
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
    # dt = dt0*10 # for Rydberg
    dt=np.sqrt(dt0) # for TFIM
    # dt = dt0
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

def driver(rho_tilde,H_tilde,two_qubit_set_tilde,ancilla_two_qubit_set_tilde ,dt, steps):
    for i in range(steps):
        gradients = compute_gradient(rho_tilde, H_tilde, two_qubit_set_tilde)
        rho_tilde = optimizer_1step_SGD_no_scheduling(rho_tilde, gradients, two_qubit_set_tilde, dt)
        # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
        # rho_tilde, second_derivatives = optimizer_1step_SGD_ancilla_no_scheduling(rho_tilde, ancilla_two_qubit_set_tilde , dt, H_tilde)
        # rho_tilde = optimizer_1step_pure_GD(rho_tilde, gradients, two_qubit_set_tilde, dt)
        rho = trace_out_rho_tilde(rho_tilde)
        rho_tilde = rho_to_rho_tilde(rho)

        E = energy(rho_tilde, H_tilde)
        if i%50==0:
            print('iteration: '+ str(i))
        # print(E)
    return E

    # print(rho_tilde.purity())

def rydberg_landscape(N, Omega, C6, r0, Delta_glob,Delta_loc):
    H = rydberg_hamiltonian_periodic(N, Omega, C6, r0, Delta_glob,Delta_loc)
    all_rho_tilde = generate_all_spin_states(N)
    E_column = []
    for rho_tilde in all_rho_tilde:
        E_column.append(energy(rho_tilde, H))
    plt.figure(figsize=(8, 5))
    plt.plot(E_column, marker='o', linestyle='-', color='b')
    plt.xlabel('Spin')
    plt.ylabel('Energy')
    title_text = (f"Energy Landscape (N={N}, Ω={Omega:.2f}, C₆={C6:.2f}, r₀={r0:.2f}, "
                  f"Δ_glob={Delta_glob:.2f}, Δ_loc={Delta_loc:.2f})")
    plt.title(title_text)
    # plt.grid(True)
    plt.show()


def top_three_spin_configurations(rho):
    N = int(np.log2(rho.shape[0]))  # Number of qubits
    eigenvalues, eigenvectors = rho.eigenstates()

    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_indices = sorted_indices[:3]  # Get indices of the top three configurations

    # Convert eigenvectors to computational basis representation
    top_configurations = []
    for idx in top_indices:
        basis_state = eigenvectors[idx].full().flatten()  # Get eigenvector as an array
        index = np.argmax(np.abs(basis_state))
        spin_string = format(index, f"0{N}b")
        top_configurations.append((spin_string, eigenvalues[idx].real))
    top_state = top_configurations[0][0]
    return top_configurations, top_state