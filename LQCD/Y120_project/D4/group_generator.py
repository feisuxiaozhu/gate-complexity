# We want to generate all D4 group elements
import numpy as np
import math as math
from numpy.linalg import matrix_power

def matrix(a,b,c):
    A = np.array([[0,1],[1,0]])
    B = np.array([[1j,0],[0,-1j]])
    left =  matrix_power(A,a)
    right = matrix_power(B,2*b+c)
    result = np.matmul(left,right)
    return result

def check_equal(A,B):
    C = np.subtract(A,B)
    epsilon = 0.0001
    sum = 0
    for i in range(2):
        for j in range(2):
            sum += abs(C[i][j])
    if sum < epsilon:
        return True
    else:
        return False

def find_index_in_list(matrix, matrix_list):
    counter = 0
    for right in matrix_list:
        if check_equal(matrix, right):
            return counter
        counter += 1

D_4 = []
for a in range(2):
    for b in range(2):
        for c in range(2):
            D_4.append(matrix(a,b,c))

for i in range(len(D_4)):
    print(i)
    matrix =  np.linalg.inv(D_4[i])
    j = find_index_in_list(matrix, D_4)
    print(j)











