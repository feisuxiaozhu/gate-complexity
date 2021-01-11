# This file contain functions that check all chi's are correct
import numpy as np
import math as math
import _pickle as pickle

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

def convert_string_to_matrix(string):
    temp = string.split(',')
    temp1=[]
    for i in temp:
        temp1.append(complex(i))
    temp1=np.reshape(temp1,(3,3))
    return temp1

def dag(u):
    return np.transpose(u.conjugate())

def tr(u):
    return np.trace(u)

def mult(a,b):
    return np.matmul(a,b)

def chi_prime_2(u):
    return np.trace(u)

def chi_prime_4(u):
    return np.trace(u)*np.trace(np.transpose(u.conjugate()))-1

def chi_prime_8(u):
    return 1/2*tr(u)*(tr(dag(u))**2+tr(mult(dag(u),dag(u))))-tr(dag(u))

def chi_prime_9(u):
    return 1/6*(np.trace(u)*np.trace(u)*np.trace(u)+2*np.trace(np.matmul(u,np.matmul(u,u)))+3*np.trace(u)*np.trace(np.matmul(u,u)))

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

def find_category_number(result, category_list):
    epsilon = 0.00001
    counter = 0
    for i in category_list:
        counter += 1
        if abs(result - i) < epsilon:
            return str(counter)


file1 = open(
    'c:/Users/feisu/Desktop/gate-complexity/third_order_contractor/higher_order_characters/data_1080.txt', 'r')
Lines = file1.readlines()
result = []
for line in Lines:
    line=line.strip('\n')
    new_string=''
    for char in line:
        if char == '{':
            new_string += ''
        elif char == '}':
            new_string += ''
        elif char == '*':
            new_string+= 'j'
        elif char == 'I':
            new_string+=''
        else:
            new_string += char
    new_string = new_string.replace(' ','')
    new_matrix = convert_string_to_matrix(new_string)
    result.append(new_matrix)

u1 = (1-math.sqrt(5))/2
u2 = (1+math.sqrt(5))/2
w = (1+math.sqrt(3)*1j)/2
w_c = (1-math.sqrt(3)*1j)/2

category_2 = [3,u2,1,w,w_c,-u1*w,-u1*w_c,0,0,-w_c,-w,u1,-u2*w_c,-u2*w,-1,-3*w_c,-3*w]
category_2_result = {}
for i in range(17):
    category_2_result[str(i+1)]=[]
for matrix in result:
    category_number = find_category_number(chi_prime_2(matrix),category_2)
    category_2_result[category_number].append(matrix)

# This part checks the values in table 1 is correct in [Fly85a]
# for i in category_2_result['2']:
#     print(chi_prime_16(i))

# This part tries to divide class 8 and class 9 into two 120=element groups
# Pick any x\in c_8\cup c_9, check the size of the set S={uxu^{-1} | u\in s1080} is 120 


repeated_set = []
for i in range(17):
    index = str(i+1)
    if index != '8' and index != '9':
        repeated_set += category_2_result[index]
# print(len(repeated_set))
new_set = []
new_set_2 = []
x = category_2_result['8'][1]
for u in result:
    y = mult(u,mult(x,dag(u)))
    if not check_whether_in_list(y, repeated_set) and not check_whether_in_list(y,new_set):
        new_set.append(y)
    elif  not check_whether_in_list(y, repeated_set) and not check_whether_in_list(y,new_set_2):
        new_set_2.append(y)


category_2_result['8'] = new_set
category_2_result['9'] = new_set_2


with open('c:/Users/feisu/Desktop/gate-complexity/third_order_contractor/higher_order_characters/result_class_element.npy', 'wb') as f:
    np.save(f, category_2_result)
   


