import numpy as np
import math as math

read_dictionary = np.load('c:/Users/feisu/Desktop/gate-complexity/third_order_contractor/higher_order_characters/result_class_element.npy',allow_pickle='TRUE').item()
def dag(u):
    return np.transpose(u.conjugate())

def tr(u):
    return np.trace(u)

def mult(a,b):
    return np.matmul(a,b)

def chi_2_m1(u):
    return 1/2*(tr(u)**2*tr(dag(u))+tr(mult(u,u))*tr(dag(u)))-tr(u)

def chi_3_1(u):
    return 1/8*(tr(u)**4+2*tr(mult(u,u))*tr(u)**2-tr(mult(u,u))**2-2*tr(mult(u,mult(u,mult(u,u)))))

u = read_dictionary['5'][0]
print(chi_3_1(u)-chi_2_m1(u))