import sympy as sym
from sympy import *

# This function takes input delta_{i,j}*delta_{jk} and outputs delta_{i,k}
def tiny_contractor(delta):
    res=[]
    dict_res = {}
    for i in delta:
        if str(i) in dict_res.keys():
            dict_res[str(i)] += 1
        else:
            dict_res[str(i)] = 1
    for i,j in dict_res.items():
        if j == 1:
            res.append(i)

    return res

#takes input [delta_ij, delta_jk, delta_mn] and outputs [delta_ik, delta_mn]
def purifier(delta):
    used_index=[]
    delta_len = len(delta)
    res = []
    for index in range(delta_len):
        found=False
        items = delta[index]
        for item in items:
            for index2 in range(index+1, delta_len):
                items_2 = delta[index2]
                if item in items_2:
                    if not found and index2 not in used_index:
                        temp = items+items_2
                        res.append(tiny_contractor(temp))
                        found=True
                        used_index.append(index2)
        if not found and index not in used_index:
            res.append(items)
    return res


# Input of this function looks like (C_1*delta*delta, C_2*delta*delta)
# It outputs C_1*C_2*delta*delta with all delta's contracted
def contractor(A,B):
    res_temp = []
    len_A = len(A)
    len_B = len(B)
    checked_B_index=[]
    for i in range(len_A):
        temp = False
        for item in A[i]:
            for j in range(len_B):
                if item in B[j]:
                    if j not in checked_B_index and not temp:
                        checked_B_index.append(j)
                        new_item = A[i] + B[j]
                        temp = True
        if temp:
            res_temp.append(tiny_contractor(new_item))
        else:
            res_temp.append(A[i])

    for i in range(len_B):
        if i not in checked_B_index:
            res_temp.append(B[i])
    return purifier(res_temp)

N = sym.Symbol('N')
#take input like [[],[],[]] and return N**3 where the exponent depends on the number of []'s 
def getNs(A):
    counter = 0
    for i in A:
        if i==[]:
            counter += 1
    return N**counter

N = sym.Symbol('N')

rhs = [ [['1','2'],['3','4'],['5','6'],['7','8']], [['1','2'],['3','4'],['5','8'],['7','6']], [['1','2'],['3','6'],['5','4'],['7','8']], [['1','2'],['3','6'],['5','8'],['7','4']], [['1','2'],['3','8'],['5','4'],['7','6']], [['1','2'],['3','8'],['5','6'],['7','4']] ]
rhs+= [ [['1','4'],['3','2'],['5','6'],['7','8']], [['1','4'],['3','2'],['5','8'],['7','6']], [['1','4'],['3','6'],['5','2'],['7','8']], [['1','4'],['3','6'],['5','8'],['7','2']], [['1','4'],['3','8'],['5','2'],['7','6']], [['1','4'],['3','8'],['5','6'],['7','2']] ]
rhs+= [ [['1','6'],['3','2'],['5','4'],['7','8']], [['1','6'],['3','2'],['5','8'],['7','4']], [['1','6'],['3','4'],['5','2'],['7','8']], [['1','6'],['3','4'],['5','8'],['7','2']], [['1','6'],['3','8'],['5','2'],['7','4']], [['1','6'],['3','8'],['5','4'],['7','2']] ]
#obtain the rows of the matrix N in temp.
temp = []
for tiny in rhs:
    temp1 = []
    for piece in rhs:
        temp1.append(getNs(contractor(tiny, piece)))
    temp.append(temp1)
print(temp)