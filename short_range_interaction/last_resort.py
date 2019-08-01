from numpy import *
from matplotlib import pylab as plt
import math as math
import random 
from scipy.linalg import expm, norm

# define Pauli matrices
sigma_x = array([[0, 1],[ 1, 0]])
sigma_y = array([[0, -1j],[1j, 0]])
sigma_z = array([[1, 0],[0, -1]])
I = array([[1,0],[0,1]])

k = 2
p=4
p_k = 1/float(4-4**(1./(2*k-1)))
a_coefficients = [1/2.*p_k,p_k,1/2.*(1-3.*p_k),1/2.*(1.-3.*p_k),p_k,1/2.*p_k]
b_coefficients = [p_k,p_k,1-4.*p_k,p_k,p_k,0]




def generate_random_h(n):
    h = []
    for i in range(n):
        h.append(random.uniform(0,1))
    return h

#take the current support, and return sum relevant norms 
def get_relevant_norms(support,H_1_array,H_2_array,j):
    current_support = set()
    query_support = set() #new additional support from H_1 and H_2
    result=0
    j = j-1
    #round 1 
    if 1 in H_1_array:
        coefficient = abs(a_coefficients[j])
    else:
        coefficient = abs(b_coefficients[j])
    for i in support:
        current_support.add(i)
    for i in support:
        if i+1 in H_1_array:
            current_support.add(i+1)
            query_support.add(i+1)
        if i-1 in H_1_array:
            current_support.add(i-1)
            query_support.add(i-1)
    for i in query_support:
        if i%2==0:
            result += norm(H_even[i],ord=2)*coefficient
            #result += coefficient
        else:
            result += norm(H_odd[i],ord=2)*coefficient
            #result += coefficient
    query_support = set()

    #round 2
    if 1 in H_2_array:
        coefficient = abs(a_coefficients[j])
    else:
        coefficient = abs(b_coefficients[j])
    for i in current_support:
        if i not in support:
            support.append(i)
    for i in support:
        if i+1 in H_2_array:
            query_support.add(i+1)
            current_support.add(i+1)
        if i-1 in H_2_array:
            query_support.add(i-1)
            current_support.add(i-1)
    for i in query_support:
        if i%2==0:
            result += norm(H_even[i],ord=2)*coefficient
            #result += coefficient
        else:
            result += norm(H_odd[i],ord=2)*coefficient
            #result += coefficient
    support_result = []
    for i in current_support:
        support_result.append(i)
    return (result,support_result)

def A_1(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        second_matrix = zeros((4,4),dtype=complex)
        if i-1 >0: 
            current_support.append(i-1)
            second_matrix += H_even[i-1]
        if i+1 <=n: 
            current_support.append(i+1)
            second_matrix += H_even[i+1]
        b_1 = b_coefficients[0]
        a_sup_1 = a_coefficients[0]
        first_matrix = H_odd[i]
        inner_norm = abs(b_1*a_sup_1)*norm(matmul(second_matrix,first_matrix)-matmul(first_matrix,second_matrix),ord=2)

        result,current_support = get_relevant_norms(current_support,H_even_array,[],1)
        running_sum += result*2
        for j in range(2,4):
            result,current_support = get_relevant_norms(current_support,H_odd_array,H_even_array,j)
            running_sum += result*2
        running_sum = inner_norm*running_sum**(p-1)
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def A_2(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        second_matrix = zeros((4,4),dtype=complex)
        if i-1 >0: 
            current_support.append(i-1)
            second_matrix += H_even[i-1]
        if i+1 <=n: 
            current_support.append(i+1)
            second_matrix += H_even[i+1]
        b_2 = b_coefficients[1]
        a_sup_2 = 0
        for c in range(2):
            a_sup_2 += a_coefficients[c]
        first_matrix = H_odd[i]
        inner_norm = abs(b_2*a_sup_2)*norm(matmul(second_matrix,first_matrix)-matmul(first_matrix,second_matrix),ord=2)

        result,current_support = get_relevant_norms(current_support,H_even_array,[],2)
        running_sum += result*2
        for j in range(3,4):
            result,current_support = get_relevant_norms(current_support,H_odd_array,H_even_array,j)
            running_sum += result*2
        running_sum = inner_norm*running_sum**(p-1)
        total_sum += running_sum   
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def A_3(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        second_matrix = zeros((4,4),dtype=complex)
        if i-1 >0: 
            current_support.append(i-1)
            second_matrix += H_even[i-1]
        if i+1 <=n: 
            current_support.append(i+1)
            second_matrix += H_even[i+1]
        b_3 = b_coefficients[2]
        a_sup_3 = 0
        for c in range(3):
            a_sup_3 += a_coefficients[c]
        first_matrix = H_odd[i]
        inner_norm = abs(b_3*a_sup_3)*norm(matmul(second_matrix,first_matrix)-matmul(first_matrix,second_matrix),ord=2)

        result,current_support = get_relevant_norms(current_support,H_even_array,[],3)
        running_sum += result*2
        running_sum = inner_norm*running_sum**(p-1)
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def A_4(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        second_matrix = zeros((4,4),dtype=complex)
        if i-1 >0: 
            current_support.append(i-1)
            second_matrix += H_even[i-1]
        if i+1 <=n: 
            current_support.append(i+1)
            second_matrix += H_even[i+1]
        b_4 = b_coefficients[4]
        a_sup_4 = 0
        for c in range(4):
            a_sup_4 += a_coefficients[c]
        first_matrix = H_odd[i]
        inner_norm = abs(b_4*a_sup_4)*norm(matmul(second_matrix,first_matrix)-matmul(first_matrix,second_matrix),ord=2)


        result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,4)
        running_sum += result*2
        running_sum = inner_norm*running_sum**(p-1)
        total_sum += running_sum 
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def A_5(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        second_matrix = zeros((4,4),dtype=complex)
        if i-1 >0: 
            current_support.append(i-1)
            second_matrix += H_even[i-1]
        if i+1 <=n: 
            current_support.append(i+1)
            second_matrix += H_even[i+1]
        b_5 = b_coefficients[1]
        a_sup_5 = 0
        for c in range(5):
            a_sup_5 += a_coefficients[c]
        first_matrix = H_odd[i]
        inner_norm = abs(b_5*a_sup_5)*norm(matmul(second_matrix,first_matrix)-matmul(first_matrix,second_matrix),ord=2)

        result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,5)
        running_sum += result*2
        result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,4)
        running_sum += result*2

        running_sum = inner_norm*running_sum**(p-1)
        total_sum += running_sum 
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum


def B_1(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        second_matrix = zeros((4,4),dtype=complex)
        if i-1 >0: 
            current_support.append(i-1)
            second_matrix += H_odd[i-1]
        if i+1 <=n: 
            current_support.append(i+1)
            second_matrix += H_odd[i+1]
        a_2 = a_coefficients[1]
        b_sup_1 = b_coefficients[0]
        first_matrix = H_even[i]
        inner_norm = abs(a_2*b_sup_1)*norm(matmul(second_matrix,first_matrix)-matmul(first_matrix,second_matrix),ord=2)

        result,current_support = get_relevant_norms(current_support,H_odd_array,H_even_array,2)
        running_sum += result*2
        result,current_support = get_relevant_norms(current_support,H_odd_array,H_even_array,3)
        running_sum += result*2

        running_sum = inner_norm*running_sum**(p-1)
        total_sum += running_sum 
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def B_2(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        second_matrix = zeros((4,4),dtype=complex)
        if i-1 >0: 
            current_support.append(i-1)
            second_matrix += H_odd[i-1]
        if i+1 <=n: 
            current_support.append(i+1)
            second_matrix += H_odd[i+1]
        a_3 = a_coefficients[2]
        b_sup_2 = 0
        for c in range(2):
            b_sup_2+= b_coefficients[c]
        first_matrix = H_even[i]
        inner_norm = abs(a_3*b_sup_2)*norm(matmul(second_matrix,first_matrix)-matmul(first_matrix,second_matrix),ord=2)


        result,current_support = get_relevant_norms(current_support,H_odd_array,H_even_array,3)
        running_sum += result*2
        running_sum = inner_norm*running_sum**(p-1)
        total_sum += running_sum
        
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def B_3(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        second_matrix = zeros((4,4),dtype=complex)
        if i-1 >0: 
            current_support.append(i-1)
            second_matrix += H_odd[i-1]
        if i+1 <=n: 
            current_support.append(i+1)
            second_matrix += H_odd[i+1]
        a_4 = a_coefficients[3]
        b_sup_3 = 0
        for c in range(3):
            b_sup_3+= b_coefficients[c]
        first_matrix = H_even[i]
        inner_norm = abs(a_4*b_sup_3)*norm(matmul(second_matrix,first_matrix)-matmul(first_matrix,second_matrix),ord=2)

        result,current_support = get_relevant_norms(current_support,[],H_odd_array,4)
        running_sum += result*2
        running_sum = inner_norm*running_sum**(p-1)
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def B_4(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        second_matrix = zeros((4,4),dtype=complex)
        if i-1 >0: 
            current_support.append(i-1)
            second_matrix += H_odd[i-1]
        if i+1 <=n: 
            current_support.append(i+1)
            second_matrix += H_odd[i+1]
        a_5 = a_coefficients[4]
        b_sup_4 = 0
        for c in range(4):
            b_sup_4+= b_coefficients[c]
        first_matrix = H_even[i]
        inner_norm = abs(a_5*b_sup_4)*norm(matmul(second_matrix,first_matrix)-matmul(first_matrix,second_matrix),ord=2)

        result,current_support = get_relevant_norms(current_support,[],H_odd_array,5)
        running_sum += result*2
        result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,4)
        running_sum += result*2
        running_sum = inner_norm*running_sum**(p-1)
        total_sum += running_sum

    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def B_5(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        second_matrix = zeros((4,4),dtype=complex)
        if i-1 >0: 
            current_support.append(i-1)
            second_matrix += H_odd[i-1]
        if i+1 <=n: 
            current_support.append(i+1)
            second_matrix += H_odd[i+1]
        a_6 = a_coefficients[5]
        b_sup_5 = 0
        for c in range(5):
            b_sup_5+= b_coefficients[c]
        first_matrix = H_even[i]
        inner_norm = abs(a_6*b_sup_5)*norm(matmul(second_matrix,first_matrix)-matmul(first_matrix,second_matrix),ord=2)

        result,current_support = get_relevant_norms(current_support,[],H_odd_array,6)
        running_sum += result*2
        running_sum = inner_norm*running_sum**(p-1)
        result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,5)
        running_sum += result*2
        result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,4)
        running_sum += result*2

        running_sum = inner_norm*running_sum**(p-1)
        total_sum += running_sum 
    
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def total_error(r,n):
    A = A_1(H_odd,r)+A_2(H_odd,r)+A_3(H_odd,r)+A_4(H_odd,r)+A_5(H_odd,r)
    B = B_1(H_even,r)+B_2(H_even,r)+B_3(H_even,r)+B_4(H_even,r)+B_5(H_even,r)
    return A+B

def find_r(n):
    error = 10**(-3)
    r = 1
    previous_r = 0
    while (total_error(r,n) > error):
        previous_r = r
        r = 2 * r
    result = binary_search(previous_r,r,n,error)   
    return result

def binary_search(low,up,n,error):
    if low + 1 >= up:
        return int(up)
    mid = ceil((low+up)/2)
    if total_error(mid,n)< error:
        return binary_search(low,mid,n,error)
    else:
        return binary_search(mid,up,n,error)

print(a_coefficients)
n = 10
result_n=[]
result_r = []
while n<= 100:
    global t
    t=n
    global h
    h = generate_random_h(n)
    global H_even
    H_even = {}
    global H_odd
    H_odd = {}
    global H_even_array
    H_even_array = []
    global H_odd_array
    H_odd_array= []
    for i in range(1,n+1):
        if i%2 == 0:
            H_even[i] = matmul(kron(sigma_x,I),kron(I,sigma_x))+matmul(kron(sigma_y,I),kron(I,sigma_y))+matmul(kron(sigma_z,I),kron(I,sigma_z))+h[i-1]*kron(sigma_z,I)
            H_even_array.append(i)
        else:
            H_odd[i] = matmul(kron(sigma_x,I),kron(I,sigma_x))+matmul(kron(sigma_y,I),kron(I,sigma_y))+matmul(kron(sigma_z,I),kron(I,sigma_z))+h[i-1]*kron(sigma_z,I)
            H_odd_array.append(i)

    r = find_r(n)
    print(n)
    print(r)
    result_n.append(n)
    result_r.append(r)
    n+= 1

filename = 'last_resort_result.csv'
savetxt(filename,c_[result_n,result_r],delimiter=',')

