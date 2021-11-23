# We want to generate all S108 group elements
import numpy as np
import _pickle as pickle
import math as math
from numpy.linalg import matrix_power

w = math.cos(2*math.pi/3) + 1j*math.sin(2*math.pi/3)
C = np.array([[1, 0, 0], [0, w, 0], [0, 0, w**2]])
E = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
V = 1/(1j*math.sqrt(3)) * np.array([[1, 1, 1], [1, w, w**2], [1, w**2, w]])

result = []

for p in range(3):
    for q in range(3):
        for r in range(3):
            for s in range(2):
                for t in range(2):
                    matrix = w**p * \
                        matrix_power(C, q) * matrix_power(E, r) * \
                        matrix_power(V, 2*s+t)
                    result.append(matrix)

with open('./S108_element.npy', 'wb') as f:
    np.save(f, result)


