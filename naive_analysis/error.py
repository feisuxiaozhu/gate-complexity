from numpy import *
seterr(over='ignore')
def evolution_error(r,n):
    t = n
    L = 4*(n-1)
    k = 2
    gamma = 1
    exponential_term = exp(2*L*5**(k-1)*gamma*t/r)
    product_term = (2*L*5**(k-1)*gamma*t)**(2*k+1)/float((3*r**(2*k)))
    return exponential_term * product_term

def find_r (n):
    error = 10**(-3)
    r = 1
    previous_r = 0
    while (evolution_error(r,n) > error):
        previous_r = r
        r = 2 * r
    result = binary_search(previous_r,r,n,error)   
    return result
    
def binary_search(low,up,n,error):
    if low + 1 >= up:
        return int(up)
    mid = ceil((low+up)/2)
    if evolution_error(mid,n)< error:
        return binary_search(low,mid,n,error)
    else:
        return binary_search(mid,up,n,error)
    
print(find_r(1000))
