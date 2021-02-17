# We want to generate all Y120 group elements
import numpy as np
import math as math
import _pickle as pickle


zero_matrix = np.array([[0,0],[0,0]])
identity_matrix = np.array([[1,0],[0,1]])
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

def dag(u):
    return np.transpose(u.conjugate())

def det(u):
    return np.linalg.det(u)

def epsilon(k):
    return  math.cos(2*math.pi*k/5) + 1j*math.sin(2*math.pi*k/5)

# return 0 if the number is small enough to be considered as zero. 
def purifier(number):
    epsilon = 0.00000001
    if abs(number) < epsilon:
        return 0
    else:
        return number

def python_matrix_to_mathematica(matrix):
    a=matrix
    first_real = purifier(a[0][0].real)
    first_img  = purifier(a[0][0].imag)
    if first_img < 0:
        first = str(first_real) + str(first_img)
    else:
        first =  str(first_real) + '+' + str(first_img)+'*I'

    second_real= purifier(a[0][1].real)
    second_img = purifier(a[0][1].imag)
    if second_img < 0:
        second = str(second_real) + str(second_img)+'*I'
    else:
        second =  str(second_real) + '+' + str(second_img)+'*I'

    third_real = purifier(a[1][0].real)
    third_img  = purifier(a[1][0].imag)
    if third_img < 0:
        third = str(third_real) + str(third_img)+'*I'
    else:
        third =  str(third_real) + '+' + str(third_img)+'*I'

    forth_real = purifier(a[1][1].real)
    forth_img  = purifier(a[1][1].imag) 
    if forth_img < 0:
        forth = str(forth_real) + str(forth_img)+'*I'
    else:
        forth =  str(forth_real) + '+' + str(forth_img)+'*I'

    matrix_string = '{'+'{' +first+','+second+'}'+','+'{' +third+','+forth+'}'+'}'    
    return matrix_string


A_dict = {}
C_dict = {}    
generators = []
# generators of Y120
k = 1
B = np.array([[0,1],[-1,0]])
A =  np.array([[epsilon(k)**3,0],[0,epsilon(k)**2]])
C = 1/math.sqrt(5)*np.array([[-epsilon(k)+epsilon(k)**4, epsilon(k)**2-epsilon(k)**3],[epsilon(k)**2-epsilon(k)**3,epsilon(k)-epsilon(k)**4]])
generators.append(B)
generators.append(C)
generators.append(A)
generators.append(dag(A))
generators.append(dag(C))


k=2
D = np.array([[epsilon(k)**2,0],[0,epsilon(k)**2]])
E = 1/math.sqrt(5)*np.array([[-epsilon(k)+epsilon(k)**4, epsilon(k)**2-epsilon(k)**3],[epsilon(k)**2-epsilon(k)**3,epsilon(k)-epsilon(k)**4]])


Y120_result = []
Y120_result.append(A)
Y120_result.append(B)
Y120_result.append(C)
Y120_result.append(dag(A))
Y120_result.append(dag(C))
# Y120_result.append(D)
# Y120_result.append(dag(D))
# Y120_result.append(E)
# Y120_result.append(dag(E))

counter = 1
while counter <4:
    counter += 1
    new_Y120_result = []
    for left in Y120_result:
        new_Y120_result.append(left)
        for right in Y120_result:

            product = mult(left, right)
            if not check_whether_in_list(product, Y120_result) and not check_whether_in_list(product, new_Y120_result):
                new_Y120_result.append(product)          
    Y120_result = new_Y120_result

with open('./Y120_element.npy', 'wb') as f:
    np.save(f, Y120_result)

# result = ''
# for matrix in Y120_result:
#     det = np.linalg.det(matrix)
#     # print(det)
#     result += python_matrix_to_mathematica(matrix) + '\n'

# print(result)



# counter = 0
# for i in Y120_result:
#     counter += 1
#     print(counter)
#     find_inverse = False
#     for j in Y120_result:
#         product = mult(i,j)
#         if check_equal(product, identity_matrix):
#             find_inverse = True
#             print(product)
#     if not find_inverse:
#         print('cannot find inverse')
#         break


# while True:
#     new_Y120_result = []
#     for element in Y120_result:
#         new_Y120_result.append(element) 
#         right_product_1 = mult(element,A)
#         right_product_2 = mult(element,B)
#         right_product_3 = mult(element,C)
#         if not check_whether_in_list(right_product_1, Y120_result) and not check_whether_in_list(right_product_1, new_Y120_result):
#             new_Y120_result.append(right_product_1)
#         if not check_whether_in_list(right_product_2, Y120_result) and not check_whether_in_list(right_product_2, new_Y120_result):
#             new_Y120_result.append(right_product_2)
#         if not check_whether_in_list(right_product_3, Y120_result) and not check_whether_in_list(right_product_3, new_Y120_result):
#             new_Y120_result.append(right_product_3)
#     Y120_result = new_Y120_result
#     print(len(Y120_result))






