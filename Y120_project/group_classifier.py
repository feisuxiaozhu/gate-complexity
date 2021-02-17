import numpy as np
from collections import Counter
from sympy import *
from sympy.functions import exp

def check_equal(A,B):
    C = np.subtract(A,B)
    epsilon = 0.000000001
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

def tr(u):
    return np.trace(u)

def mult(a,b):
    return np.matmul(a,b)

def det(u):
    return np.linalg.det(u)

def purifier(number):
    epsilon = 0.00000001
    if abs(number) < epsilon:
        return 0
    elif abs(number-1)<epsilon:
        return 1
    elif abs(number+1)<epsilon:
        return -1
    else:
        return number

Y_120 = np.load('./Y120_element.npy',allow_pickle='TRUE')

# # Find the distribution of traces
# set_trace = []
# for u in Y_120:
#     trace_real = round(purifier(tr(u).real),4)
#     trace_img =  round(purifier(tr(u).imag),4)
#     trace = trace_real + 1j*trace_img
#     set_trace.append(trace)
# print(Counter(set_trace))

# # Find the distribution of products
# set_product = []
# for u in Y_120:
#     for v in Y_120:
#         product = mult(u,v)
#         index = find_index_in_list(product,Y_120)
#         set_product.append(index)
# print(Counter(set_product))

# Find the parametrizaiton of each group element

alpha=Symbol('alpha',real=True)
beta =Symbol('beta', real=True)
theta=Symbol('theta',real=True)

matrix = Y_120[1]

print(matrix[0][1])
print(matrix[1][0])

eq1 = Eq(exp(I*alpha)*cos(theta), matrix[0][0])
eq2 = Eq(exp(I*beta)*sin(theta), matrix[0][1])
eq3 = Eq(-exp(I*beta)*sin(theta),matrix[1][0])
eq4 = Eq(exp(-I*alpha)*cos(theta), matrix[1][1])
result_1=solve([eq1],alpha,beta,theta)
result_2=solve([eq2],alpha,beta,theta)
result_3=solve([eq3],alpha,beta,theta)
result_4=solve([eq4],alpha,beta,theta)

print(result_1)
print(result_2)
print(result_3)
print(result_4)