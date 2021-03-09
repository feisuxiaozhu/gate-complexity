import numpy as np
from collections import Counter
from sympy import *
from sympy.functions import exp
import math

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

def inv(a):
    return np.linalg.inv(a) 

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

def radian_to_deg(r):
    return r/3.14159265358979*180.0

# We follow the parametrization in the following paper
# https://www.tamagawa.jp/research/quantum/bulletin/pdf/Tamagawa.Vol.5-5.pdf
def get_beta_eta_zeta(matrix):
    beta =round(purifier(acos(2*(matrix[0][0]*matrix[1][1]).real-1)) / np.pi * 180,4)
    eta = round(arg(matrix[1][1]/cos(beta/2)).evalf(15) / np.pi * 180,4)
    zeta = round(arg(-matrix[0][1]/sin(beta/2)).evalf(15) / np.pi * 180,4)
    return beta, eta, zeta

def get_beta_eta_zeta_integer(matrix):
    beta,eta,zeta = get_beta_eta_zeta(matrix)
    if not math.isnan(eta):
        eta = int(eta/18)
    if not math.isnan(zeta):
        zeta = int(zeta/18)
    epsilon = 0.0001
    if beta-0<epsilon:
        beta = 0
    elif beta- 63.4349 < epsilon:
        beta = 1
    elif beta-116.5651< epsilon:
        beta = 2
    elif beta-180<epsilon:
        beta = 3
    return beta,eta,zeta

Y_120 = np.load('./Y120_element.npy',allow_pickle='TRUE')





beta_set=set()
eta_set=set()
zeta_set=set()
for i in range(120):
    A = Y_120[i]
    beta,eta,zeta = get_beta_eta_zeta_integer(A)
    beta_set.add(beta)
    eta_set.add(eta)
    zeta_set.add(zeta)
print(beta_set)
print(eta_set)
print(zeta_set)


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

