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

    # if j% 2 == 0:
    #     coefficient = b_coefficients[j-1]
    # else:
    #     coefficient = a_coefficients[j-1]
    # print(query_support)
    # for i in query_support:
    #     if i%2==0:
    #         #result += norm(H_even[i],ord=2)*coefficient
    #         result += coefficient
    #     else:
    #         #result += norm(H_odd[i],ord=2)*coefficient
    #         result += coefficient
    support_result = []
    for i in current_support:
        support_result.append(i)
    return (result,support_result)


def A_1(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        for j in range(1,4):
            result,current_support = get_relevant_norms(current_support,H_odd_array,H_even_array,j)
            running_sum += result*2
        running_sum = norm(H[i],ord=2)*abs(a_coefficients[0])*running_sum**p
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def A_2(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        for j in range(2,4):
            result,current_support = get_relevant_norms(current_support,H_odd_array,H_even_array,j)
            running_sum += result*2
        running_sum = norm(H[i],ord=2)*abs(a_coefficients[1])*running_sum**p
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def A_3(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        for j in range(3,4):
            result,current_support = get_relevant_norms(current_support,H_odd_array,H_even_array,j)
            running_sum += result*2
        running_sum = norm(H[i],ord=2)*abs(a_coefficients[2])*running_sum**p
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def A_4(H,r):
    total_sum = 0
    for i in H:
        total_sum += norm(H[i],ord=2)*abs(a_coefficients[3])
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum
    
def A_5(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        for j in range(4,5):
            result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,j)
            running_sum += result*2
        running_sum = norm(H[i],ord=2)*abs(a_coefficients[4])*running_sum**p
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def A_6(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        for j in range(5,3,-1):
            result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,j)
            running_sum += result*2
        running_sum = norm(H[i],ord=2)*abs(a_coefficients[5])*running_sum**p
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def B_1(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        for j in range(2,4):
            result,current_support = get_relevant_norms(current_support,H_odd_array,H_even_array,j)
            running_sum += result*2
        running_sum = norm(H[i],ord=2)*abs(b_coefficients[0])*running_sum**p
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def B_2(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        for j in range(3,4):
            result,current_support = get_relevant_norms(current_support,H_odd_array,H_even_array,j)
            running_sum += result*2
        running_sum = norm(H[i],ord=2)*abs(b_coefficients[1])*running_sum**p
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def B_3(H,r):
    total_sum = 0
    for i in H:
        total_sum += norm(H[i],ord=2)*abs(b_coefficients[2])
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def B_4(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        for j in range(4,5):
            result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,j)
            running_sum += result*2
        running_sum = norm(H[i],ord=2)*abs(b_coefficients[3])*running_sum**p
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def B_5(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        for j in range(5,3,-1):
            result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,j)
            running_sum += result*2
        running_sum = norm(H[i],ord=2)*abs(b_coefficients[4])*running_sum**p
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def B_6(H,r):
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        for j in range(6,3,-1):
            result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,j)
            running_sum += result*2
        running_sum = norm(H[i],ord=2)*abs(b_coefficients[5])*running_sum**p
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

def C(H,r): #H_even or H_odd is the input
    total_sum = 0
    for i in H:
        running_sum = 0
        current_support = [i]
        for j in range(6,3,-1):
            result,current_support = get_relevant_norms(current_support,H_even_array,H_odd_array,j)
            running_sum += result*2
        running_sum = norm(H[i],ord=2)*running_sum**p
        total_sum += running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return total_sum

k = 2
p=4
p_k = 1/float(4-4**(1/(2*k-1)))
a_coefficients = [1/2.*p_k,p_k,1/2.*(1-3.*p_k),1/2.*(1.-3.*p_k),p_k,1/2.*p_k]
b_coefficients = [p_k,p_k,1-4.*p_k,p_k,p_k,0]

def total_error(r,n):
    A = A_1(H_odd,r)+A_2(H_odd,r)+A_3(H_odd,r)+A_4(H_odd,r)+A_5(H_odd,r)+A_6(H_odd,r)
    B = B_1(H_even,r)+B_2(H_even,r)+B_3(H_even,r)+B_4(H_even,r)+B_5(H_even,r)+B_6(H_even,r)
    C_final = C(H_even,r)+C(H_odd,r)
    return A+B+C_final

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
            H_even[i] = matmul(sigma_x,sigma_x)+matmul(sigma_y,sigma_y)+matmul(sigma_z,sigma_z)+h[i-1]*sigma_z
            H_even_array.append(i)
        else:
            H_odd[i] = matmul(sigma_x,sigma_x)+matmul(sigma_y,sigma_y)+matmul(sigma_z,sigma_z)+h[i-1]*sigma_z
            H_odd_array.append(i)
    r = find_r(n)
    print(n)
    print(r)
    result_n.append(n)
    result_r.append(r)
    n+= 1

filename = 'result.csv'
savetxt(filename,c_[result_n,result_r],delimiter=',')
# plt.loglog(result_n,result_r,basex=10)
# plt.grid(True)
# plt.show()

# n=20
# t=n
# h = generate_random_h(n)
# H_even = {}
# H_odd = {}
# H_even_array = []
# H_odd_array= []
# for i in range(1,n+1):
#     if i%2 == 0:
#         H_even[i] = matmul(sigma_x,sigma_x)+matmul(sigma_y,sigma_y)+matmul(sigma_z,sigma_z)+h[i-1]*sigma_z
#         H_even_array.append(i)
#     else:
#         H_odd[i] = matmul(sigma_x,sigma_x)+matmul(sigma_y,sigma_y)+matmul(sigma_z,sigma_z)+h[i-1]*sigma_z
#         H_odd_array.append(i)
# total = 0
# print(a_coefficients)
# print(b_coefficients)
# a,current_support = get_relevant_norms([1],[1,3,5,7,9],[2,4,6,8,10],1)
# total += a*2
# a,current_support = get_relevant_norms(current_support,[1,3,5,7,9],[2,4,6,8,10],2)
# total += a*2
# a,current_support = get_relevant_norms(current_support,[1,3,5,7,9],[2,4,6,8,10],3)
# total += a*2
# print(total)
