import qutip as qt
import numpy as np
from PXPsubroutines import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle

N = 6
H_tilde = H_PXP_tilde(N)

# rho_tilde = find_ground_state(H_tilde)
# rho_tilde = create_minus_zero_state(N)
rho_tilde = create_CDW_state(N)
# with open('rho_pxp_pureSGD_5000.pkl', 'rb') as f:
#     rho_tilde = rho_to_rho_tilde(np.load(f, allow_pickle=True))





# Create operator O = X I I I... (X on position 0, and I on other registers including ancilla)
op_list = [qt.qeye(2) for _ in range(N)]
op_list[0] = qt.sigmax()
O = qt.tensor([qt.qeye(2)]+op_list)

dt = np.pi/100

value_column = []
T_column = []
# evolve:
for i in range(800):
    print(i)
    T = dt*i    
    value = (O*evolve(rho_tilde,H_tilde,T)).tr()
    T_column.append(T)
    value_column.append(value)


# fig, axes = plt.subplots(1, 1, figsize=(16, 5))
plt.plot(T_column,value_column)
plt.xlabel('t')
plt.ylabel('Expectation of O')
plt.ticklabel_format(style='plain', axis='y', useOffset=False)
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.show()


