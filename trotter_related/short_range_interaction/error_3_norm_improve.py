from numpy import *
from scipy.linalg import expm, norm
import random 
import timeit

# define Pauli matrices
sigma_x = array([[0, 1],[ 1, 0]])
sigma_y = array([[0, -1j],[1j, 0]])
sigma_z = array([[1, 0],[0, -1]])
I = array([[1,0],[0,1]])

def calculate_norm(H):
    result= 0
    for i,j in H.items():
        result += norm(j,ord=2)
    return result

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
            current_support.add(i+1)
        if i-1 in H_2_array:
            query_support.add(i-1)
            current_support.add(i-1)
    result=0
    for i in query_support:
        if i%2==0:
            result += norm(H_even[i],ord=2)
        else:
            result += norm(H_odd[i],ord=2)
    support_result = []
    for i in current_support:
        support_result.append(i)
    return (result,support_result)
    
def evolution_error_norm_1(r,n):
    t= n
    s= 10
    p=4
    k = 2
    p_k = 1/float(4-4**(1/(2*k-1)))
    mu = max(abs(p_k)/2,abs(1-4*p_k)/2)
    total_sum = 0
    for j in range(1,int(s/2)+1):
        if j%2==0:
            H = H_even
        else:
            H= H_odd
        for i in H:
            running_sum = 0
            current_support = [i]
            for l in range(j,int(s/2)+1):
                if l % 2 == 0:
                    first_array = H_even_array
                    second_array = H_odd_array
                else:
                    first_array=H_odd_array
                    second_array = H_even_array
                result,support_result = get_relevant_norms(current_support,first_array,second_array)
                current_support = support_result
                running_sum+= result*2*mu
            running_sum = norm(H[i],ord=2)*mu*running_sum**p
            total_sum+= running_sum
    
    for j in range(int(s/2)+1,s+1):
        if j%2==0:
            H = H_even
        else:
            H= H_odd
        for i in H:
            running_sum = 0
            current_support = [i]
            for l in range(j-1,int(s/2),-1):
                if l % 2 == 0:
                    first_array = H_even_array
                    second_array = H_odd_array
                else:
                    first_array=H_odd_array
                    second_array = H_even_array
                result,support_result = get_relevant_norms(current_support,first_array,second_array)
                current_support = support_result
                running_sum+= result*2*mu
            running_sum = norm(H[i],ord=2)*mu*running_sum**p
            total_sum+= running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return(total_sum)

def evolution_error_norm_2(r,n):
    return evolution_error_norm_1(r,n)

def evolution_error_norm_3(r,n):
    t= n
    s= 10
    p=4
    k = 2
    p_k = 1/float(4-4**(1/(2*k-1)))
    mu = max(abs(p_k)/2,abs(1-4*p_k)/2)
    total_sum = 0
    for i in H_total:
        running_sum = 0
        current_support=[i]
        for l in range(s,int(s/2),-1):
            if l % 2 == 0:
                    first_array = H_even_array
                    second_array = H_odd_array
            else:
                first_array=H_odd_array
                second_array = H_even_array
            result,support_result = get_relevant_norms(current_support,first_array,second_array)
            current_support = support_result
            running_sum+= result*2*mu
        running_sum=norm(H_total[i],ord=2)*mu*running_sum**p
        total_sum+= running_sum
    total_sum = total_sum*t**(p+1)/float(math.factorial(p+1)*r**(p))
    return(total_sum)

def evolution_error_norm(r,n):
    return evolution_error_norm_1(r,n)+evolution_error_norm_2(r,n)+evolution_error_norm_3(r,n)

def find_r_norm(n):
    error = 10**(-3)
    r = 1
    previous_r = 0
    while (evolution_error_norm(r,n) > error):
        previous_r = r
        r = 2 * r
    result = binary_search_norm(previous_r,r,n,error)   
    return result

def binary_search_norm(low,up,n,error):
    if low + 1 >= up:
        return int(up)
    mid = ceil((low+up)/2)
    if evolution_error_norm(mid,n)< error:
        return binary_search_norm(low,mid,n,error)
    else:
        return binary_search_norm(mid,up,n,error)


result_r_norm = []
result_n=[]
n=5
while n<=100:
    global h 
    h = generate_random_h(n)
    global H_even 
    H_even = {}
    global H_odd 
    H_odd = {}
    global H_total
    H_total ={}
    global H_even_array
    H_even_array = []
    global H_odd_array 
    H_odd_array= []
    for i in range(1,n+1):
        if i%2 == 0:
            H_even[i] = matmul(sigma_x,sigma_x)+matmul(sigma_y,sigma_y)+matmul(sigma_z,sigma_z)+h[i-1]*sigma_z
            H_total[i] = matmul(sigma_x,sigma_x)+matmul(sigma_y,sigma_y)+matmul(sigma_z,sigma_z)+h[i-1]*sigma_z
            H_even_array.append(i)
        else:
            H_odd[i] = matmul(sigma_x,sigma_x)+matmul(sigma_y,sigma_y)+matmul(sigma_z,sigma_z)+h[i-1]*sigma_z
            H_total[i] = matmul(sigma_x,sigma_x)+matmul(sigma_y,sigma_y)+matmul(sigma_z,sigma_z)+h[i-1]*sigma_z
            H_odd_array.append(i)

    r_norm = find_r_norm(n)
    print(n)
    print(r_norm)
    result_n.append(n)
    result_r_norm.append(r_norm)
    n += 1
filename = 'result.csv'
savetxt(filename,c_[result_n,result_r_norm],delimiter=',')



