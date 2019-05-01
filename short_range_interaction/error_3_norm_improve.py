from numpy import *
from scipy.linalg import expm, norm
import random 
import timeit

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
def get_relevant_norms(support,H_1_array,H_2_array):
    current_support = set()
    query_support = set() #new additional support from H_1 and H_2
    #round 1 
    for i in support:
        current_support.add(i)
    for i in support:
        if i+1 in H_1_array:
            current_support.add(i+1)
            query_support.add(i+1)
        if i-1 in H_1_array:
            current_support.add(i-1)
            query_support.add(i-1)
    #round 2
    for i in current_support:
        if i not in support:
            support.append(i)
    for i in support:
        if i+1 in H_2_array:
            query_support.add(i+1)
        if i-1 in H_2_array:
            query_support.add(i-1)
    result=0
    for i in query_support:
        if i%2==0:
            result += norm(H_even[i],ord=2)
        else:
            result += norm(H_odd[i],ord=2)
    return result
    
def evolution_error_norm_1(r,n):
    t= n
    s= 10
    p=4
    k = 2
    p_k = 1/float(4-4**(1/(2*k-1)))
    mu = max(abs(p_k)/2,abs(1-4*p_k)/2)
    total_sum = 0
    for j in range(1,int(s/2)+1):
        running_sum = 0
        if j%2==0:
            H = H_even
        else:
            H= H_odd
        for i in H:
            current_support = i
            for l in range(j,int(s/2)+1):
                if l % 2 == 0:
                    first_array = H_even_array
                    second_array = H_odd_array
                else:
                    first_array=H_odd_array
                    second_array = H_even_array



n=10
h = generate_random_h(n)
H_even = {}
H_odd = {}
H_even_array=[]
H_odd_array = []
for i in range(1,n+1):
    if i%2 == 0:
        H_even[i] = matmul(sigma_x,sigma_x)+matmul(sigma_y,sigma_y)+matmul(sigma_z,sigma_z)+h[i-1]*sigma_z
        H_even_array.append(i)
    else:
        H_odd[i] = matmul(sigma_x,sigma_x)+matmul(sigma_y,sigma_y)+matmul(sigma_z,sigma_z)+h[i-1]*sigma_z
        H_odd_array.append(i)

evolution_error_norm_1(10,10)






