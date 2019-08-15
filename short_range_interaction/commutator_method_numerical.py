from numpy import *
from matplotlib import pylab as plt
import math as math
import random 
from scipy.linalg import expm, norm
 
 
commutator_coefficients = {'ABA': 0.00463891008158979, 'ABB': 0.00737205664595221, 'AAA': 0.0047013343101698826, 'AAB': 0.005703818762629349, 'BBB': 0.028373434405425904, 'BBA': 0.01732815305710953, 'BAB': 0.009726162358456837, 'BAA': 0.009689668290122171}
# define Pauli matrices
sigma_x = array([[0, 1],[ 1, 0]])
sigma_y = array([[0, -1j],[1j, 0]])
sigma_z = array([[1, 0],[0, -1]])
I = array([[1,0],[0,1]])

def generate_random_h(n):
    h = []
    for i in range(n):
        h.append(random.uniform(0,1))
    return h

def tensor_product_generator(sigma,dic,length):
    for j in range(length):
        if j == 0:
            running_multiple = sigma
        else:
            running_multiple = I
        for i in range(1,length):
            if i == j:
                running_multiple = kron(running_multiple,sigma)
            else:
                running_multiple = kron(running_multiple,I)
        dic[j] = running_multiple.astype(complex)

def construct_H_even(h,n,X,Y,Z):
    H = zeros((2**n,2**n)).astype(complex)
    for i in range(n-1):
        if (i+1) % 2== 0:
            H += matmul(X[i],X[i+1])+matmul(Y[i],Y[i+1])+matmul(Z[i],Z[i+1])+h[i]*Z[i]
    return H

def construct_H_odd(h,n,X,Y,Z):
    H = zeros((2**n,2**n)).astype(complex)
    for i in range(n-1):
        if (i+1) % 2== 1:
            H += matmul(X[i],X[i+1])+matmul(Y[i],Y[i+1])+matmul(Z[i],Z[i+1])+h[i]*Z[i]
    return H

def LP(A,B): #as Lie product
    return subtract(matmul(A,B),matmul(B,A))

def MN(A): #as matrix norm
    return norm(A,ord=2)

n = 20
X={};Y={};Z={}
h = generate_random_h(n)
tensor_product_generator(sigma_x,X,n)
tensor_product_generator(sigma_y,Y,n)
tensor_product_generator(sigma_z,Z,n)
A = construct_H_odd(h,n,X,Y,Z)
B = construct_H_even(h,n,X,Y,Z)
print(A)

# def Nomr_matrix(['A','A','B'']):
#     return norm(LP(A,LP(A,LP(B,LP(A,B)))),ord=2)
# res = norm(LP(A,LP(A,B)),ord=2)
# print(res*0.00463891008158979*n**5)








