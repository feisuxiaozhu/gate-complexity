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

# def det(u):
#     return np.linalg.det(u)

def purifier(number):
    epsilon = 0.000001
    if abs(number) < epsilon:
        return 0
    elif abs(number-1)<epsilon:
        return 1
    elif abs(number+1)<epsilon:
        return -1
    else:
        return number

# We follow the parametrization in the following paper
# https://www.tamagawa.jp/research/quantum/bulletin/pdf/Tamagawa.Vol.5-5.pdf
def get_beta_eta_zeta(matrix):

    # eta=Symbol('eta',real=True)
    # beta =Symbol('beta', real=True)
    # zeta=Symbol('zeta',real=True)

    beta =purifier(acos(2*(matrix[0][0]*matrix[1][1]).real-1))
    eta = arg(matrix[1][1]/cos(beta/2)).evalf(15)
    zeta = arg(-matrix[0][1]/sin(beta/2)).evalf(15)
    
    return beta, eta, zeta

Y_120 = np.load('./Y120_element.npy',allow_pickle='TRUE')




A = Y_120[20]
B = Y_120[2]
print(get_beta_eta_zeta(A))
print(get_beta_eta_zeta(B))
print(get_beta_eta_zeta(mult(A,B)))
    

# print(matrix[0][1])
# print(matrix[1][0])

# eq1 = Eq(exp(I*alpha)*cos(theta), matrix[0][0])
# eq2 = Eq(exp(I*beta)*sin(theta), matrix[0][1])
# eq3 = Eq(-exp(I*beta)*sin(theta),matrix[1][0])
# eq4 = Eq(exp(-I*alpha)*cos(theta), matrix[1][1])
# result_1=solve([eq1],alpha,beta,theta)
# result_2=solve([eq2],alpha,beta,theta)
# result_3=solve([eq3],alpha,beta,theta)
# result_4=solve([eq4],alpha,beta,theta)

