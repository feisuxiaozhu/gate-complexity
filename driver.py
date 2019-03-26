import sys
from numpy import *
from matplotlib import pylab as plt
seterr(over='ignore')

def evolution_error(r,n):
    t = n
    L = 4*(n-1)
    k = 2
    gamma = 1
    exponential_term = exp(2*L*5**(k-1)*gamma*t/r)
    product_term = (2*L*5**(k-1)*gamma*t)**(2*k+1)/float((3*r**(2*k)))
    return exponential_term * product_term

def evolution_error_1(r,n):
    t = n
    s=10
    p=4
    gamma = 4
    k = 2
    p_k = 1/float(4-4**(1/(2*k-1)))
    mu = max(abs(p_k)/2,abs(1-4*p_k)/2)
    first_factor = floor(n/2)*gamma*mu
    third_factor = t**(p+1)/float(math.factorial(p+1)*r**(p))
    total_sum = 0
    for j in range(1,s+1):
        running_sum = 0
        for i in range(1,j+1):
            if i == 1:
                running_sum += 2
            elif i <= floor(n/2):
                running_sum += i + i+1
            else:
                running_sum += 2*floor(n/2)
        total_sum += first_factor*(2*gamma*mu*running_sum)**p * third_factor
    return total_sum

def evolution_error_2(r,n):
    return evolution_error_1(r,n)

def evolution_error_3(r,n):
    t = n
    s=10
    p=4
    gamma = 4
    k = 2
    p_k = 1/float(4-4**(1/(2*k-1)))
    mu = max(abs(p_k)/2,abs(1-4*p_k)/2)
    first_factor = n*gamma
    third_factor = t**(p+1)/float(math.factorial(p+1)*r**(p))
    running_sum = 0
    for i in range(1,s+1):
        if i == 1:
            running_sum += 2
        elif i <= floor(n/2):
            running_sum += i + i+1
        else:
            running_sum += 2*floor(n/2)
    total_sum = first_factor*(2*gamma*mu*running_sum)**p * third_factor
    return total_sum

def evolution_error_product(r,n):
    return evolution_error_1(r,n) + evolution_error_2(r,n) + evolution_error_3(r,n)

def find_r (n):
    error = 10**(-3)
    r = 1
    previous_r = 0
    while (evolution_error(r,n) > error):
        previous_r = r
        r = 2 * r
    result = binary_search(previous_r,r,n,error)   
    return result

def find_r_product (n):
    error = 10**(-3)
    r = 1
    previous_r = 0
    while (evolution_error_product(r,n) > error):
        previous_r = r
        r = 2 * r
    result = binary_search_product(previous_r,r,n,error)   
    return result

def binary_search(low,up,n,error):
    if low + 1 >= up:
        return int(up)
    mid = ceil((low+up)/2)
    if evolution_error(mid,n)< error:
        return binary_search(low,mid,n,error)
    else:
        return binary_search(mid,up,n,error)

def binary_search_product(low,up,n,error):
    if low + 1 >= up:
        return int(up)
    mid = ceil((low+up)/2)
    if evolution_error_product(mid,n)< error:
        return binary_search_product(low,mid,n,error)
    else:
        return binary_search_product(mid,up,n,error)

n = 10
result_n = []
result_r = []
result_r_product = []
while n <= 100:
    r = find_r(n)
    r_product = find_r_product(n)
    # print('n is: ' + str(n))
    # print('r: ' + str(r))
    result_n.append(n)
    result_r.append(r)
    result_r_product.append(r_product)
    n += 1

plt.loglog(result_n,result_r,basex = 10)
plt.loglog(result_n,result_r_product,basex = 10)
plt.grid(True)
plt.show()
