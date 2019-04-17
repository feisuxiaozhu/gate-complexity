from numpy import *
from scipy.linalg import expm, norm
n = 10
h = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

# define Pauli matrices
sigma_x = array([[0, 1],[ 1, 0]])
sigma_y = array([[0, -1j],[1j, 0]])
sigma_z = array([[1, 0],[0, -1]])
I = array([[1,0],[0,1]])
# create dictionary for  X_j, Y_j, and Z_j, note they are tensor products!
X={};Y={};Z={}
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
tensor_product_generator(sigma_x,X,n)
tensor_product_generator(sigma_y,Y,n)
tensor_product_generator(sigma_z,Z,n)

def construct_expitH(h,n,X,Y,Z):
    t = n
    H = zeros((2**n,2**n)).astype(complex)
    for i in range(n-1):
        H += matmul(X[i],X[i+1])+matmul(Y[i],Y[i+1])+matmul(Z[i],Z[i+1])+h[i]*Z[i]
    H = expm(-1j*t*H)
    return H

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

def construct_S2(h,n,X,Y,Z,t):
    H_odd = construct_H_odd(h,n,X,Y,Z)
    H_even = construct_H_even(h,n,X,Y,Z)
    H_odd_exponent = -1j*t*H_odd/2.
    H_even_exponent = -1j*t*H_even
    A = expm(H_odd_exponent)
    B = expm(H_even_exponent)
    return matmul(matmul(A,B),A)

def construct_S4(h,n,X,Y,Z,t): #r is used for trotter step
    p = 1/float(4-4**(1./(2*2-1))) 
    A = construct_S2(h,n,X,Y,Z,p*t)
    B = construct_S2(h,n,X,Y,Z,(1-4*p)*t)
    C = matmul(A,A)
    return matmul(matmul(C,B),C)

def findError(h,n,X,Y,Z,r):
    t = n/r
    A = construct_expitH(h,n,X,Y,Z)
    B = construct_S4(h,n,X,Y,Z,t)
    B_exponent = B 
    for i in range(r):
        if i!=0:
            B_exponent = matmul(B_exponent,B)
    return norm(subtract(A,B_exponent),ord=2)

print(findError(h,n,X,Y,Z,1000))






