import sympy as sym
import numpy as np

read_dictionary = np.load('c:/Users/feisu/Desktop/gate-complexity/third_order_contractor/higher_order_characters/match_exponential_to_expansion/c_for_i_j_result.npy',allow_pickle='TRUE').item()

def beta(i):
    string = 'beta_'+str(i)
    return sym.Symbol(string)

cutoff = 17
coefficient_dict ={}
for i in range(cutoff):
    key = str(i+1)
    if key == '1':
        coefficient_dict[key] = beta(i+1) - 1
    else:
        coefficient_dict[key] = beta(i+1)
for i in range(cutoff):
    for j in range(cutoff):
        key = str(i+1)
        key_2 = str(j+1)
        combo_key = key+ '_'+key_2
        if key == key_2:
            coefficient_dict[combo_key] = -1/2* coefficient_dict[key]**2
        else:
            coefficient_dict[combo_key] = -1/2*coefficient_dict[key]*coefficient_dict[key_2]

# Find each chi_i*chi_j's coefficients, and add them to corresponding chi_i's.
for i in range(cutoff):
    for j in range(cutoff):
        key = str(i+1)
        key_2 = str(j+1)
        combo_key = key+ '_'+key_2
        simplification_vector = read_dictionary[combo_key]
        constant = coefficient_dict[combo_key]
        for k in range(cutoff):
            key_3 = str(k+1)
            coefficient_dict[key_3] += simplification_vector[k]*constant
        
# print result 
for i in range(cutoff):
    key = str(i+1)
    coefficient = sym.simplify(sym.expand(coefficient_dict[key]))
    print(key)
    print(coefficient)















