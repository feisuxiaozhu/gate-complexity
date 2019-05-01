from numpy import *
from matplotlib import pylab as plt
import math as math
def evolution_error_better_1(r,n):
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
    for j in range(1,s/2+1):
        running_sum = 0
        for i in range(1,s/2+1-(j)+1):
            if i == 1:
                running_sum += 2
            elif i <= floor(n/2):
                running_sum += i + i+1
            else:
                running_sum += 2*floor(n/2)
        total_sum += first_factor*(2*gamma*mu*running_sum)**p * third_factor

        running_sum = 0
        for i in range(1,s/2+1-(j)+1):
            if i == 1:
                running_sum += 5
            elif i <= floor(n/2):
                running_sum += i+1 + i+2
            else:
                running_sum += 2*floor(n/2)
        total_sum += first_factor*(2*gamma*mu*running_sum)**p * third_factor

        return total_sum

def evolution_error_better_2(r,n):
    return evolution_error_better_1(r,n)

def evolution_error_better_3(r,n):
    t = n
    s=10
    p=4
    gamma = 4
    k = 2
    p_k = 1/float(4-4**(1/(2*k-1)))
    mu = max(abs(p_k)/2.,abs(1-4*p_k)/2.)
    first_factor = n*gamma/ 2.
    third_factor = t**(p+1)/float(math.factorial(p+1)*r**(p))
    running_sum = 0
    for i in range(1,s/2+1):
        if i == 1:
            running_sum += 2
        elif i <= floor(n/2):
            running_sum += i + i+1
        else:
            running_sum += 2*floor(n/2)
    for i in range(1,s/2+1):
        if i == 1:
            running_sum += 5
        elif i <= floor(n/2):
            running_sum += i+1 + i+2
        else:
            running_sum += 2*floor(n/2)
    total_sum = first_factor*(2*gamma*mu*running_sum)**p * third_factor
    return total_sum

def evolution_error_better(r,n):
    return evolution_error_better_1(r,n)+evolution_error_better_2(r,n)+evolution_error_better_3(r,n)
print(evolution_error_better(1800,10))
