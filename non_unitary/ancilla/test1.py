import qutip as qt
import numpy as np
from ancillaSubroutines import *
import pickle
import matplotlib.pyplot as plt

N = 3
H = NN_H(N)
two_qubit_set = all_two_qubit_set_NN(N)

rho = create_spin_state(N,[])

trace_out(rho)








