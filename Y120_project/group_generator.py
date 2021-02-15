# We want to generate all Y120 group elements
import numpy as np
import math as math

# Check whether matrix A equals matrix B
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

# Check whether matrix A exists in list matrix_list, if not return new list, o.w. return old list
def check_whether_in_list(A,matrix_list):
    duplicate = False    
    for B in matrix_list:
        if check_equal(A,B):
            duplicate = True
    return duplicate

def mult(a,b):
    return np.matmul(a,b)

def epsilon(k):
    return  math.cos(2*math.pi*k/5) + 1j*math.sin(2*math.pi*k/5)

zero_matrix = np.array([[0,0],[0,0]])
A_dict = {}
C_dict = {}    
generators = []
# generators of Y120
B = np.array([[0,1],[-1,0]])
generators.append(B)
for k in range(2):
    index = str(k+1)
    A_dict[index] = np.array([[epsilon(k)**2,0],[0,epsilon(k)**2]])
    generators.append(A_dict[index])
    C_dict[index] = 1/math.sqrt(5)*np.array([[-epsilon(k)+epsilon(k)**4, epsilon(k)**2-epsilon(k)**3],[epsilon(k)**2-epsilon(k)**3,epsilon(k)-epsilon(k)**4]])
    if not check_equal(zero_matrix,C_dict[index]):
        generators.append(C_dict[index])
        generators.append(mult(A_dict[index],C_dict[index]))
        generators.append(mult(B,C_dict[index]))


Y120_result = []
# Y120_result.append(zero_matrix)

for generator in generators:
    if not check_whether_in_list(generator, Y120_result):
        Y120_result.append(generator)
    for initial in generators:
        stop_sign = False
        previous_element = initial
        while not stop_sign:
            current_element = mult(generator,previous_element)
            if not check_whether_in_list(current_element, Y120_result):
                Y120_result.append(current_element)
            if check_equal(initial,current_element):
                stop_sign = True
            previous_element = current_element
            print(len(Y120_result))
        
print(len(Y120_result))




