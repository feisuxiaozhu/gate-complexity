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
    res = []
    res_temp = []
    res.append(A[0]+B[0])
    len_A = len(A)
    len_B = len(B)
    checked_B_index=[]
    for i in range(1,len_A):
        temp = False
        for item in A[i]:
            for j in range(1,len_B):
                if item in B[j]:
                    if j not in checked_B_index:
                        checked_B_index.append(j)
                        new_item = A[i] + B[j]
                        temp = True
        if temp:
            res_temp.append(tiny_contractor(new_item))
        else:
            res_temp.append(A[i])

    for i in range(1,len_B):
        if i not in checked_B_index:
            res_temp.append(B[i])

    return res+purifier(res_temp)


# It takes input as E= C*delta*delta+ ... and F= C*delta*delta+... 
# Outputs contracted multiple between E and F
def Multiply(E,F):
    res = []
    for item_E in E:
        for item_F in F:
            res.append(contractor(item_E,item_F))
    return res


def u_contractor(U,D):
    res_dict=['Tr(u)^2','Tr(u^2)']
    D_used_index=[]
    res=[]
    for item_U in U:
        added=False
        for item in item_U:
            for j in range(len(D)):
                if item in D[j]:
                    if j not in D_used_index and not added:
                        D_used_index.append(j)
                        new_item=item_U + D[j]
                        res.append(tiny_contractor(new_item))
                        added=True
    counter = 0
    for i in res:
        if len(i)==0:
            counter+=1
    if counter == 0:
        return res_dict[1]
    elif counter == 2:
        return res_dict[0]


A = [ [['1'],['2','3'],['7','8']], [['2'],['2','8'],['3','7']] ]
B = [ [['1'],['3','4'],['8','9']], [['2'],['3','9'],['4','8']] ]
C = [ [['1'],['4','5'],['9','alpha']], [['2'],['4','alpha'],['5','9']] ]
D = [ [['1'],['5','1'],['alpha','6']], [['2'],['5','6'],['1','alpha']] ]
temp1= Multiply(A,B)
temp2 = Multiply(C,D)
temp3 = Multiply(temp1, temp2)


U = [['1','2'],['6','7']]

zero_res = {}
for item in temp3:
    D=[]
    for i in range(1,len(item)):
        D.append(item[i])
    u = u_contractor(U,D)
    C = item[0]
    if u in zero_res.keys():
        zero_res[u].append(C)
    else:
        zero_res[u]=[C]

V_2 = Symbol('V_2')
V_11 = Symbol('V_11')
c_dict={'1':1/2*(V_2+V_11),'2':1/2*(V_2-V_11)}

zero_res_final = {}
for i,j in zero_res.items():
    temp1=0
    for k in j:
        temp = 1
        for l in k:
            temp *= c_dict[l]
        temp1 += temp
    zero_res_final[i] = sym.simplify(temp1)

print(zero_res_final)

