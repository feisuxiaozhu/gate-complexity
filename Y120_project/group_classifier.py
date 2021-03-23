import numpy as np
from collections import Counter
from sympy import *
from sympy.functions import exp
import math
from collections import Counter

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

def tr(u):
    return np.trace(u)

def mult(a,b):
    return np.matmul(a,b)

def inv(a):
    return np.linalg.inv(a) 

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
    return round(r/np.pi*180.0,2)

def deg_to_radian(deg):
    return deg/180.0 * np.pi

# We follow the parametrization in the following paper
# https://www.tamagawa.jp/research/quantum/bulletin/pdf/Tamagawa.Vol.5-5.pdf
def get_beta_eta_zeta_radian(matrix): #return in radian
    beta =purifier(acos(2*(matrix[0][0]*matrix[1][1]).real-1)) 
    eta = arg(matrix[1][1]/cos(beta/2)).evalf(15)
    zeta = arg(-matrix[0][1]/sin(beta/2)).evalf(15) 
    return beta, eta, zeta

def get_beta_eta_zeta(matrix): # return in degree
    beta,eta,zeta = get_beta_eta_zeta_radian(matrix)
    beta = round(beta/np.pi*180,4)
    eta = round(eta/np.pi*180,4)
    zeta = round(zeta/np.pi*180,4)
    return beta,eta,zeta

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

def construct_Y120_matrix(beta_int, eta_int, zeta_int):
    if math.isnan(eta_int): # assign 0 as value of eta if eta is NaN
        eta = 0
    else:
        eta = int(18*eta_int)

    if math.isnan(zeta_int): # assign 0 as value of zeta if zeta is NaN
        zeta = 0
    else:
        zeta = int(18*zeta_int)
    if beta_int == 0:
        beta = 0
    elif beta_int == 1:
        beta = 63.4349
    elif beta_int ==2:
        beta = 116.5651
    elif beta_int ==3:
        beta = 180
    beta = deg_to_radian(beta)
    eta = deg_to_radian(eta)
    zeta = deg_to_radian(zeta)
    M_11 = purifier(np.exp(-1j*eta)*np.cos(beta/2))
    M_12 = purifier(-np.exp(1j*zeta)*np.sin(beta/2))
    M_21 = purifier(np.exp(-1j*zeta)*np.sin(beta/2))
    M_22 = purifier(np.exp(1j*eta)*np.cos(beta/2))
    matrix = np.array( [[M_11,M_12],[M_21,M_22]])
    return matrix



# Y_120 = np.load('./Y120_element.npy',allow_pickle='TRUE')
# beta_pool = [0,1,2,3]
# eta_pool = [-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
# zeta_pool=[-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
# beta_dict_radian = {0:0,1:63.4349/180*np.pi,2:116.5651/180*np.pi,3:180/180*np.pi}

# prod_set=set()
# for beta_1 in beta_pool:
#     for beta_2 in beta_pool:
#         beta_1_new = beta_dict_radian[beta_1]
#         beta_2_new = beta_dict_radian[beta_2]
#         prod_sin =round(float( (sin(beta_1_new)*sin(beta_2_new))),3)
#         prod_cos =round(float( (cos(beta_1_new)*cos(beta_2_new))),3)
#         prod_set.add((prod_cos,prod_sin))
# print(prod_set)


# for left_beta in beta_pool:
#     for right_beta in beta_pool:
#         print('-----------------------------------')
#         print(left_beta,right_beta)
#         experiment = []
#         for matrix_left in Y_120:
#             for matrix_right in Y_120:
#                 if get_beta_eta_zeta_integer(matrix_left)[0]==left_beta and get_beta_eta_zeta_integer(matrix_right)[0]==right_beta:
#                     prod = mult(matrix_left,matrix_right)
#                     new_beta,new_eta,new_zeta = get_beta_eta_zeta_integer(prod)
#                     experiment.append(new_beta)
#         print(Counter(experiment))

# pool_set=set()
# for left_matrix in Y_120:
#     for right_matrix in Y_120:
#         beta1,eta1,zeta1 = get_beta_eta_zeta_radian(left_matrix)
#         beta2,eta2,zeta2 = get_beta_eta_zeta_radian(right_matrix)
#         if math.isnan(eta1):
#             eta1=0
#         if math.isnan(eta2):
#             eta2=0
#         if math.isnan(zeta1):
#             zeta1=0
#         if math.isnan(zeta2):
#             zeta2=0
#         test = round(cos(beta1)*cos(beta2)*sqrt(5),2)
#         # test = radian_to_deg(test)
#         print(test)
#         pool_set.add(test)
# print(pool_set)





