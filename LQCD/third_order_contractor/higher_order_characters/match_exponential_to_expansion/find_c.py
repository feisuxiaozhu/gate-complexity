import numpy as np
import math as math
from numpy import linalg
import _pickle as pickle

read_dictionary = np.load('../result_class_element.npy',allow_pickle='TRUE').item()

# Check whether matrix A equals matrix B
def check_equal(A,B):
    C = np.subtract(A,B)
    epsilon = 0.000000001
    sum = 0
    for i in range(3):
        for j in range(3):
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

# Return the class number an S1080 element belongs
def check_class_number(u, read_dictionary):
    for i,matrix_list in read_dictionary.items():
        for matrix in matrix_list:
            if check_equal(u,matrix):
                return i
    return 'n/a'


def dag(u):
    return np.transpose(u.conjugate())

def tr(u):
    return np.trace(u)

def mult(a,b):
    return np.matmul(a,b)

def chi_prime_1(u):
    return 1.0

def chi_prime_2(u):
    return np.trace(u)

def chi_prime_3(u):
    return chi_prime_2(dag(u))

def chi_prime_4(u):
    return np.trace(u)*np.trace(np.transpose(u.conjugate()))-1

def chi_prime_5(u):
    return 1/2*(tr(mult(u,u))+tr(u)**2)

def chi_prime_6(u):
    return chi_prime_5(dag(u))

def chi_prime_8(u):
    return 1/2*tr(u)*(tr(dag(u))**2+tr(mult(dag(u),dag(u))))-tr(dag(u))

def chi_prime_7(u):
    return chi_prime_8(dag(u))

def chi_prime_9(u):
    return 1/6*(np.trace(u)*np.trace(u)*np.trace(u)+2*np.trace(np.matmul(u,np.matmul(u,u)))+3*np.trace(u)*np.trace(np.matmul(u,u)))

def chi_prime_10(u):
    output=[5.0,0.0,-1.0,1.0,1.0,0.0,0.0,-1.0,2.0,-1.0,-1.0,0.0,0.0,0.0,1.0,5.0,5.0]
    class_number = check_class_number(u,read_dictionary)
    index = int(class_number)-1
    return output[index]

def chi_prime_11(u):
    output=[5.0,0.0,-1.0,1.0,1.0,0.0,0.0,2.0,-1.0,-1.0,-1.0,0.0,0.0,0.0,1.0,5.0,5.0]
    class_number = check_class_number(u,read_dictionary)
    index = int(class_number)-1
    return output[index]


def chi_prime_12(u):
    first_term = -tr(u)*tr(dag(u)) + 1/4*(tr(dag(u))**2+tr(mult(dag(u),dag(u))))*(tr(u)**2+tr(mult(u,u)))
    second_term = 1/4*(tr(u)**2*tr(dag(u))**2 + tr(u)**2*tr(mult(dag(u),dag(u))) + tr(dag(u))**2*tr(mult(u,u))+tr(mult(dag(u),dag(u)))*tr(mult(u,u)))
    third_term = 1/6*(-tr(u)**3-3*tr(u)*tr(mult(u,u))-2*tr(mult(u,mult(u,u))))
    fourth_term = 1/6*(tr(u)**3+3*tr(u)*tr(mult(u,u))+2*tr(mult(u,mult(u,u))))
    fifth_term = -1/36*(tr(dag(u))**3+3*tr(dag(u))*tr(mult(dag(u),dag(u)))+2*tr(mult(dag(u),mult(dag(u),dag(u)))))*(tr(u)**3+3*tr(u)*tr(mult(u,u))+2*tr(mult(u,mult(u,u))))
    sixth_term = 1/24*tr(dag(u))*(tr(u)**4+6*tr(u)**2*tr(mult(u,u))+3*tr(mult(u,u))**2+8*tr(u)*tr(mult(u,mult(u,u)))+6*tr(mult(u,mult(u,mult(u,u)))))
    return first_term + second_term + third_term + fourth_term + fifth_term + sixth_term

def chi_prime_13(u):
    first_term = 1-1/4*(tr(dag(u))**2+tr(mult(dag(u),dag(u))))*(tr(u)**2+tr(mult(u,u)))
    second_term = 1/4*(-tr(u)**2*tr(dag(u))**2-tr(u)**2*tr(mult(dag(u),dag(u)))-tr(dag(u))**2*tr(mult(u,u))-tr(mult(dag(u),dag(u)))*tr(mult(u,u)))
    third_term = 1/3*(-tr(u)**3-3*tr(u)*tr(mult(u,u))-2*tr(mult(u,mult(u,u))))
    fourth_term = 1/36*(tr(dag(u))**3+3*tr(dag(u))*tr(mult(dag(u),dag(u)))+2*tr(mult(dag(u),mult(dag(u),dag(u)))))*(tr(u)**3+3*tr(u)*tr(mult(u,u))+2*tr(mult(u,mult(u,u))))
    return first_term+second_term+third_term+fourth_term

def chi_prime_14(u):
    first_term = 1/6*np.trace(np.transpose(u.conjugate()))*(np.trace(u)**3+2*np.trace(np.matmul(u,np.matmul(u,u)))+3*np.trace(np.matmul(u,u))*np.trace(u))  
    second_term = -1/2*(np.trace(u)**2+np.trace(np.matmul(u,u)))
    third_term = -1/2*np.trace(u)*(np.trace(np.transpose(u.conjugate()))**2+np.trace(np.matmul(np.transpose(u.conjugate()),np.transpose(u.conjugate()))))
    fourth_term = np.trace(np.transpose(u.conjugate()))
    return first_term+second_term+third_term+fourth_term

def chi_prime_15(u):
    first_term = 1/6*tr(u)*(tr(dag(u))**3+2*tr(mult(dag(u),mult(dag(u),dag(u))))+3*tr(mult(dag(u),dag(u)))*tr(dag(u)))
    second_term = -1/2*(tr(dag(u))**2+tr(mult(dag(u),dag(u))))
    third_term = -1/2*tr(dag(u))*(tr(u)**2+tr(mult(u,u)))+tr(u)
    return first_term+second_term+third_term

def chi_prime_16(u):
    zero_term = -1/2*tr(dag(u))*(tr(u)**2+tr(mult(u,u)))
    zero_five_term = 1/12*(tr(dag(u))**2+tr(mult(dag(u),dag(u))))*(tr(u)**3+3*tr(u)*tr(mult(u,u))+2*tr(mult(u,mult(u,u))))
    first_term = -1/2*tr(dag(u))*(tr(u)**2+tr(mult(u,u))) + tr(u)
    second_term = -1/6*tr(u)*(tr(dag(u))**3+2*tr(mult(dag(u),mult(dag(u),dag(u))))+3*tr(mult(dag(u),dag(u)))*tr(dag(u)))
    third_term = 1/2*(tr(dag(u))**2 + tr(mult(dag(u),dag(u))))
    return first_term+second_term+third_term+zero_five_term+zero_term

def chi_prime_17(u):
    return chi_prime_16(dag(u))


def chi_prime(r,u):
    if r == '1':
        return chi_prime_1(u)
    elif r=='2':
        return chi_prime_2(u)
    elif r=='3':
        return chi_prime_3(u)
    elif r=='4':
        return chi_prime_4(u)
    elif r=='5':
        return chi_prime_5(u)
    elif r=='6':
        return chi_prime_6(u)
    elif r=='7':
        return chi_prime_7(u)
    elif r=='8':
        return chi_prime_8(u)
    elif r=='9':
        return chi_prime_9(u)
    elif r=='10':
        return chi_prime_10(u)
    elif r=='11':
        return chi_prime_11(u)
    elif r=='12':
        return chi_prime_12(u)
    elif r=='13':
        return chi_prime_13(u)
    elif r=='14':
        return chi_prime_14(u)
    elif r=='15':
        return chi_prime_15(u)
    elif r=='16':
        return chi_prime_16(u)
    elif r=='17':
        return chi_prime_17(u)

# find c_1, c_2, ..., c_17 for given i,j
def create_vector(i,j):
    vector = []
    for k in range(17):
        key = str(k+1)
        u = read_dictionary[key][0]
        result = chi_prime(i,u) * chi_prime(j,u)
        vector.append([result])
    return vector

# create a matrix that we need to invert in the end
def create_matrix():
    matrix = []
    for k in range(17):
        row = []
        key = str(k+1)
        u = read_dictionary[key][0]
        for ell in range(17):
            key_2 = str(ell+1)
            row.append(chi_prime(key_2,u))
        matrix.append(row)
    return matrix

# calculate c_1,...,c_17 for given i,j pair
# i,j are string choosen from set {'1','2',...,'17'}
def compute_c(i,j):
    result = []
    A = np.array(create_matrix())
    x = np.array(create_vector(i,j))
    answer = linalg.solve(A, x) 
    real_part = answer.real
    for i in real_part:
        number = i[0]
        if number < 0.0000000001:
            result.append(0.0)
        else:
            result.append(round(number,2))
    return result


result_dict = {}
for i in range(17):
    for j in range(17):
        index_1 = str(i+1)
        index_2 = str(j+1)
        key = index_1+'_'+index_2
        result_dict[key] = compute_c(index_1,index_2)

print(result_dict)
with open('./c_for_i_j_result.npy', 'wb') as f:
    np.save(f, result_dict)
