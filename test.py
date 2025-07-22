from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt

# Create named quantum registers
p = QuantumRegister(2, name='p')       # Register for p (p[1], p[0])
q = QuantumRegister(2, name='q')       # Register for q (q[1], q[0])
anc = QuantumRegister(2, name='anc')   # Ancilla qubits
flag = QuantumRegister(1, name='flag') # Output flag qubit

# Create circuit
qc = QuantumCircuit(p, q, anc, flag)

# Step 1: Compare MSB (p[1] vs q[1])
qc.cx(p[1], anc[1])              # anc[1] = p[1]
qc.cx(q[1], anc[1])              # anc[1] = p[1] XOR q[1]
qc.ccx(p[1], anc[1], flag[0])    # if p[1]=1, q[1]=0 => flag=1

# Step 2: Compare LSB (p[0] vs q[0]) only if MSBs equal
qc.cx(p[0], anc[0])              # anc[0] = p[0]
qc.cx(q[0], anc[0])              # anc[0] = p[0] XOR q[0]
qc.x(anc[1])                     # Invert anc[1] for control
qc.ccx(p[0], anc[0], flag[0])    # if p[0]=1, q[0]=0 and MSBs same => flag=1
qc.x(anc[1])                     # Reset anc[1]

# Draw the circuit
qc.draw(output='mpl')
plt.show()
