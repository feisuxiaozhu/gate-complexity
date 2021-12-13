# We want to generate all S108 group elements
import numpy as np
import _pickle as pickle
import math as math
from numpy.linalg import matrix_power

# Check whether matrix A equals matrix B


def check_equal(A, B):
    C = np.subtract(A, B)
    epsilon = 0.000000001
    sum = 0
    for i in range(len(A)):
        for j in range(len(A)):
            sum += abs(C[i][j])
    if sum < epsilon:
        return True
    else:
        return False


def mult(a, b):
    return np.matmul(a, b)

def inv(a):
    return np.linalg.inv(a) 

def trace(a):
     return np.matrix(a).trace()

# Return the key of matrix A in dic


def check_dict_key(A, dic):
    for key, value in dic.items():
        if check_equal(value, A):
            return key


def g(p, q, r, s, t):
    return w**p * \
        mult(mult(matrix_power(C, q), matrix_power(E, r)), matrix_power(V, 2*s+t))


w = math.cos(2*math.pi/3) + 1j*math.sin(2*math.pi/3)
C = np.array([[1, 0, 0], [0, w, 0], [0, 0, w**2]])
E = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
V = 1/(1j*math.sqrt(3)) * np.array([[1, 1, 1], [1, w, w**2], [1, w**2, w]])

S108 = []
S108_dict = {}


for p in range(3):
    for q in range(3):
        for r in range(3):
            for s in range(2):
                for t in range(2):
                    matrix = g(p, q, r, s, t)
                    S108 .append(matrix)
                    key = str(p)+str(q)+str(r)+str(s)+str(t)
                    S108_dict[key] = matrix

# Find inverse             
# pattern = set()
# for index, A in S108_dict.items():
#     B = inv(A)
#     new_index = check_dict_key(B, S108_dict)
#     # print(index, new_index)
#     print(index[:4], new_index[:4])
#     p = str(index[:4])+str(new_index[:4])
#     pattern.add(p)


# Find trace
pattern = {}
for index, A in S108_dict.items():
    t  = trace(A)
    real = t.real.round(5)
    img = t.imag.round(5)
    new_t = complex(real, img)
    print(index, new_t)
    if str(new_t) not in pattern.keys():
        pattern[str(new_t)] = [index]
    else:
        pattern[str(new_t)].append(index)
print(pattern)
print(len(pattern))