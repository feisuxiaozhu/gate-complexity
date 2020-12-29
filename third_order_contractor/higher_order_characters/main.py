# This file contain functions that check all chi's are correct
import numpy as np
import math as math

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

def chi_prime_9(u):
    return 1/6*(np.trace(u)*np.trace(u)*np.trace(u)+2*np.trace(np.matmul(u,np.matmul(u,u)))+3*np.trace(u)*np.trace(np.matmul(u,u)))

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

for i in category_2_result['16']:
    print(chi_prime_16(i))

