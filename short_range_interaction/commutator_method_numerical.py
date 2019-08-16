from numpy import *
from matplotlib import pylab as plt
import math as math
import random 
from scipy.linalg import expm, norm
 
 
commutator_coefficients = {'ABA': 0.00463891008158979, 'ABB': 0.00737205664595221, 'AAA': 0.0047013343101698826, 'AAB': 0.005703818762629349, 'BBB': 0.028373434405425904, 'BBA': 0.01732815305710953, 'BAB': 0.009726162358456837, 'BAA': 0.009689668290122171}
# define Pauli matrices
sigma_x = array([[0, 1],[ 1, 0]])
sigma_y = array([[0, -1j],[1j, 0]])
sigma_z = array([[1, 0],[0, -1]])
I = array([[1,0],[0,1]])

def generate_random_h(n):
    h = []
    for i in range(n-1):
        h.append(random.uniform(0,1))
    return h

def generate_local_terms(h,local_terms):
    counter = 1
    for b_field in h:
        term = kron(sigma_x,sigma_x)+kron(sigma_y,sigma_y)+kron(sigma_z,sigma_z)+b_field*kron(sigma_z,I)
        local_terms[counter] = term
        counter+= 1
    zero_term = zeros((4,4)).astype(complex)
    local_terms[0] = zero_term
    local_terms[-1] = zero_term
    local_terms[-2] = zero_term
    local_terms[counter] = zero_term
    local_terms[counter+1] = zero_term
    local_terms[counter+2] = zero_term
    local_terms[counter+3] = zero_term

def kron_helper(n,i,local_term):
    if i == 1:
        running_multiple = local_term
    else:
        running_multiple = I
    for j in range(2,n+1):
        if j==i:
            running_multiple = kron(running_multiple,local_term)
        else:
            running_multiple = kron(running_multiple,I)
    return running_multiple


def AAA(n,r):
    total_result = 0
    i = 2
    t=n
    while (i<n):
        Hp1 = kron_helper(3,3,local_terms[i+1])
        H = kron_helper(3,2,local_terms[i])
        Hm1 = kron_helper(3,1,local_terms[i-1])
        first_level_matrix = Hp1 + Hm1
        second_level_matrix = Hp1 + Hm1
        third_level_matrix = Hp1 + Hm1
        fourth_level_matrix = Hp1 + Hm1
        temp = MN(LP(fourth_level_matrix,LP(third_level_matrix,LP(second_level_matrix,LP(first_level_matrix,H)))))
        total_result+= abs(temp)
        i+=2
    total_result = total_result*commutator_coefficients['AAA']*t**5/(r**4)
    return total_result

def BAB(n,r):
    total_result = 0
    i = 2
    t = n

    while i<n:
        H = kron_helper(9,5,local_terms[i])
        Hm1 = kron_helper(9,4,local_terms[i-1])
        Hm2 = kron_helper(9,3,local_terms[i-2])
        Hm3 = kron_helper(9,2,local_terms[i-3])
        Hm4 = kron_helper(9,1,local_terms[i-4])
        Hp1 = kron_helper(9,6,local_terms[i+1])
        Hp2 = kron_helper(9,7,local_terms[i+2])
        Hp3 = kron_helper(9,8,local_terms[i+3])
        Hp4 = kron_helper(9,9,local_terms[i+4])
        first_level_matrix = Hm1 + Hp1
        second_level_matrix = Hm2 + H + Hp2
        third_level_matrix = Hm3 + Hm1 + Hp1 + Hp3
        fourth_level_matrix = Hm4 + Hm2 + H + Hp2 + Hp4
        temp = MN(LP(fourth_level_matrix,LP(third_level_matrix,LP(second_level_matrix,LP(first_level_matrix,H)))))
        total_result+= abs(temp)
        i+=2
    total_result = total_result * commutator_coefficients['BAB']*t**5/(r**4)
    return total_result

def AAB(n,r):
    total_result = 0
    i = 2
    t = n
    while i<n:
        H = kron_helper(7,4,local_terms[i])
        Hm1 = kron_helper(7,3,local_terms[i-1])
        Hm2 = kron_helper(7,2,local_terms[i-2])
        Hm3 = kron_helper(7,1,local_terms[i-3])
        Hp1 = kron_helper(7,5,local_terms[i+1])
        Hp2 = kron_helper(7,6,local_terms[i+2])
        Hp3 = kron_helper(7,7,local_terms[i+3])
        first_level_matrix = Hm1 + Hp1
        second_level_matrix = Hm2 + H + Hp2
        third_level_matrix = Hm3 + Hm1 + Hp1 + Hp3
        fourth_level_matrix = Hm3 + Hm1 + Hp1 + Hp3
        temp = MN(LP(fourth_level_matrix,LP(third_level_matrix,LP(second_level_matrix,LP(first_level_matrix,H)))))
        total_result+= abs(temp)
        i+=2
    total_result = total_result * commutator_coefficients['AAB']*t**5/(r**4)
    return total_result

def ABA(n,r):
    total_result = 0
    i = 2
    t = n
    while i<n:
        H = kron_helper(7,4,local_terms[i])
        Hm1 = kron_helper(7,3,local_terms[i-1])
        Hm2 = kron_helper(7,2,local_terms[i-2])
        Hm3 = kron_helper(7,1,local_terms[i-3])
        Hp1 = kron_helper(7,5,local_terms[i+1])
        Hp2 = kron_helper(7,6,local_terms[i+2])
        Hp3 = kron_helper(7,7,local_terms[i+3])
        first_level_matrix = Hm1 + Hp1
        second_level_matrix = Hm1 + Hp1
        third_level_matrix = Hm2 + H + Hp2
        fourth_level_matrix = Hm3 + Hm1 + Hp1 + Hp3
        temp = MN(LP(fourth_level_matrix,LP(third_level_matrix,LP(second_level_matrix,LP(first_level_matrix,H)))))
        total_result+= abs(temp)
        i+=2
    total_result = total_result * commutator_coefficients['ABA']*t**5/(r**4)
    return total_result

def ABB(n,r):
    total_result = 0
    i = 2
    t = n
    while i<n:
        H = kron_helper(7,4,local_terms[i])
        Hm1 = kron_helper(7,3,local_terms[i-1])
        Hm2 = kron_helper(7,2,local_terms[i-2])
        Hm3 = kron_helper(7,1,local_terms[i-3])
        Hp1 = kron_helper(7,5,local_terms[i+1])
        Hp2 = kron_helper(7,6,local_terms[i+2])
        Hp3 = kron_helper(7,7,local_terms[i+3])
        first_level_matrix = Hm1 + Hp1
        second_level_matrix = Hm2 + H + Hp2
        third_level_matrix = Hm2 + H + Hp2
        fourth_level_matrix = Hm3 + Hm1 + Hp1 + Hp3
        temp = MN(LP(fourth_level_matrix,LP(third_level_matrix,LP(second_level_matrix,LP(first_level_matrix,H)))))
        total_result+= abs(temp)
        i+=2
    total_result = total_result * commutator_coefficients['ABB']*t**5/(r**4)
    return total_result

def BBB(n,r):
    total_result = 0
    i = 2
    t = n
    while i<n:
        H = kron_helper(5,3,local_terms[i])
        Hm1 = kron_helper(5,2,local_terms[i-1])
        Hm2 = kron_helper(5,1,local_terms[i-2])
        Hp1 = kron_helper(5,4,local_terms[i+1])
        Hp2 = kron_helper(5,5,local_terms[i+2])
        first_level_matrix = Hm1 + Hp1
        second_level_matrix = Hm2 + H + Hp2
        third_level_matrix = Hm2 + H + Hp2
        fourth_level_matrix = Hm2 + H + Hp2
        temp = MN(LP(fourth_level_matrix,LP(third_level_matrix,LP(second_level_matrix,LP(first_level_matrix,H)))))
        total_result+= abs(temp)
        i+=2
    total_result = total_result * commutator_coefficients['BBB']*t**5/(r**4)
    return total_result

def BBA(n,r):
    total_result = 0
    i = 2
    t = n
    while i<n:
        H = kron_helper(5,3,local_terms[i])
        Hm1 = kron_helper(5,2,local_terms[i-1])
        Hm2 = kron_helper(5,1,local_terms[i-2])
        Hp1 = kron_helper(5,4,local_terms[i+1])
        Hp2 = kron_helper(5,5,local_terms[i+2])
        first_level_matrix = Hm1 + Hp1
        second_level_matrix = Hm1 + Hp1
        third_level_matrix = Hm2 + H + Hp2
        fourth_level_matrix = Hm2 + H + Hp2
        temp = MN(LP(fourth_level_matrix,LP(third_level_matrix,LP(second_level_matrix,LP(first_level_matrix,H)))))
        total_result+= abs(temp)
        i+=2
    total_result = total_result * commutator_coefficients['BBA']*t**5/(r**4)
    return total_result


def BAA(n,r):
    total_result = 0
    i = 2
    t = n
    while i<n:
        H = kron_helper(5,3,local_terms[i])
        Hm1 = kron_helper(5,2,local_terms[i-1])
        Hm2 = kron_helper(5,1,local_terms[i-2])
        Hp1 = kron_helper(5,4,local_terms[i+1])
        Hp2 = kron_helper(5,5,local_terms[i+2])
        first_level_matrix = Hm1 + Hp1
        second_level_matrix = Hm1 + Hp1
        third_level_matrix = Hm1 + Hp1
        fourth_level_matrix = Hm2 + H + Hp2
        temp = MN(LP(fourth_level_matrix,LP(third_level_matrix,LP(second_level_matrix,LP(first_level_matrix,H)))))
        total_result+= abs(temp)
        i+=2
    total_result = total_result * commutator_coefficients['BAA']*t**5/(r**4)
    return total_result

def total_error(n,r):
    return AAA(n,r)+BAB(n,r)+AAB(n,r)+ABA(n,r)+ABB(n,r)+BBB(n,r)+BBA(n,r)+BAA(n,r)

def LP(A,B): #as Lie product
    return subtract(matmul(A,B),matmul(B,A))

def MN(A): #as matrix norm
    return norm(A,ord=2)

def find_r(n):
    error = 10**(-3)
    r = 100
    previous_r = 0
    while (total_error(n,r) > error):
        previous_r = r
        r = 2 * r
    result = binary_search(previous_r,r,n,error,1,5)   
    return result

def binary_search(low,up,n,error,counter,constraint):
    if counter == constraint:
        return up
    else:
        counter += 1
    if low + 1 >= up:
        return int(up)
    mid = ceil((low+up)/2)
    if total_error(n,mid)< error:
        return binary_search(low,mid,n,error,counter,constraint)
    else:
        return binary_search(mid,up,n,error,counter,constraint)

n = 10
while n<= 100:
    print(n)
    local_terms = {}
    h = generate_random_h(n)
    generate_local_terms(h,local_terms)
    r = find_r(n)
    print(r)
    n+=10








