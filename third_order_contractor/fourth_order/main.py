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
                    if j not in checked_B_index and not temp:
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

# It takes input as U=u*u*u and D=delta*delta*delta
# Outputs contracted u as trace.
# For this case we do not need to consider the case receiving delta_ii, or [].


def u_contractor(U,D):
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
        for i in range(1,len(res)):
            if sorted(res[0]) == sorted(res[i]):
                return 'Tr^2(u^2)'    
        return 'Tr(u^4)'
    elif counter == 1:
        return 'Tr(u)Tr(u^3)'
    elif counter == 2:
        return 'Tr^2(u)Tr(u^2)'
    elif counter == 3:
        return 'fuck you'
    elif counter == 4:
        return 'Tr^4(u)'

A = [  [['1'],['2','3'],['7','8'],['gamma','eta'],['b','c']], [['2'],['2','3'],['7','8'],['gamma','c'],['b','eta']], [['3'],['2','3'],['7','eta'],['gamma','8'],['b','c']], [['4'],['2','3'],['7','eta'],['gamma','c'],['b','8']], [['5'],['2','3'],['7','c'],['gamma','8'],['b','eta']], [['6'],['2','3'],['7','c'],['gamma','eta'],['b','8']]  ]
A+= [  [['7'],['2','8'],['7','3'],['gamma','eta'],['b','c']], [['8'],['2','8'],['7','3'],['gamma','c'],['b','eta']], [['9'],['2','8'],['7','eta'],['gamma','3'],['b','c']], [['10'],['2','8'],['7','eta'],['gamma','c'],['b','3']], [['11'],['2','8'],['7','c'],['gamma','3'],['b','eta']], [['12'],['2','8'],['7','c'],['gamma','eta'],['b','3']] ]
A+= [  [['13'],['2','eta'],['7','3'],['gamma','8'],['b','c']], [['14'],['2','eta'],['7','3'],['gamma','c'],['b','8']], [['15'],['2','eta'],['7','8'],['gamma','3'],['b','c']], [['16'],['2','eta'],['7','8'],['gamma','c'],['b','3']], [['17'],['2','eta'],['7','c'],['gamma','3'],['b','8']], [['18'],['2','eta'],['7','c'],['gamma','8'],['b','3']] ]
A+= [  [['19'],['2','c'],['7','3'],['gamma','8'],['b','eta']], [['20'],['2','c'],['7','3'],['gamma','eta'],['b','8']], [['21'],['2','c'],['7','8'],['gamma','3'],['b','eta']], [['22'],['2','c'],['7','8'],['gamma','eta'],['b','3']], [['23'],['2','c'],['7','eta'],['gamma','3'],['b','8']], [['24'],['2','c'],['7','eta'],['gamma','8'],['b','3']]   ]

B = [  [['1'],['3','4'],['8','9'],['eta','mu'],['c','d']], [['2'],['3','4'],['8','9'],['eta','d'],['c','mu']], [['3'],['3','4'],['8','mu'],['eta','9'],['c','d']], [['4'],['3','4'],['8','mu'],['eta','d'],['c','9']], [['5'],['3','4'],['8','d'],['eta','9'],['c','mu']], [['6'],['3','4'],['8','d'],['eta','mu'],['c','9']]   ]
B+= [  [['7'],['3','9'],['8','4'],['eta','mu'],['c','d']], [['8'],['3','9'],['8','4'],['eta','d'],['c','mu']], [['9'],['3','9'],['8','mu'],['eta','4'],['c','d']], [['10'],['3','9'],['8','mu'],['eta','d'],['c','4']], [['11'],['3','9'],['8','d'],['eta','4'],['c','mu']], [['12'],['3','9'],['8','d'],['eta','mu'],['c','4']]   ]
B+= [  [['13'],['3','mu'],['8','4'],['eta','9'],['c','d']], [['14'],['3','mu'],['8','4'],['eta','d'],['c','9']], [['15'],['3','mu'],['8','9'],['eta','4'],['c','d']], [['16'],['3','mu'],['8','9'],['eta','d'],['c','4']], [['17'],['3','mu'],['8','d'],['eta','4'],['c','9']], [['18'],['3','mu'],['8','d'],['eta','9'],['c','4']]  ]
B+= [  [['19'],['3','d'],['8','4'],['eta','9'],['c','mu']], [['20'],['3','d'],['8','4'],['eta','mu'],['c','9']], [['21'],['3','d'],['8','9'],['eta','4'],['c','mu']], [['22'],['3','d'],['8','9'],['eta','mu'],['c','4']], [['23'],['3','d'],['8','mu'],['eta','4'],['c','9']], [['24'],['3','d'],['8','mu'],['eta','9'],['c','4']]  ]

C = [  [['1'],['4','5'],['9','alpha'],['mu','nu'],['d','e']], [['2'],['4','5'],['9','alpha'],['mu','e'],['d','nu']], [['3'],['4','5'],['9','nu'],['mu','alpha'],['d','e']], [['4'],['4','5'],['9','nu'],['mu','e'],['d','alpha']], [['5'],['4','5'],['9','e'],['mu','alpha'],['d','nu']], [['6'],['4','5'],['9','e'],['mu','nu'],['d','alpha']]     ]
C+= [  [['7'],['4','alpha'],['9','5'],['mu','nu'],['d','e']], [['8'],['4','alpha'],['9','5'],['mu','e'],['d','nu']], [['9'],['4','alpha'],['9','nu'],['mu','5'],['d','e']], [['10'],['4','alpha'],['9','nu'],['mu','e'],['d','5']], [['11'],['4','alpha'],['9','e'],['mu','5'],['d','nu']], [['12'],['4','alpha'],['9','e'],['mu','nu'],['d','5']]  ]
C+= [  [['13'],['4','nu'],['9','5'],['mu','alpha'],['d','e']], [['14'],['4','nu'],['9','5'],['mu','e'],['d','alpha']], [['15'],['4','nu'],['9','alpha'],['mu','5'],['d','e']], [['16'],['4','nu'],['9','alpha'],['mu','e'],['d','5']], [['17'],['4','nu'],['9','e'],['mu','5'],['d','alpha']], [['18'],['4','nu'],['9','e'],['mu','alpha'],['d','5']]   ]
C+= [  [['19'],['4','e'],['9','5'],['mu','alpha'],['d','nu']], [['20'],['4','e'],['9','5'],['mu','nu'],['d','alpha']], [['21'],['4','e'],['9','alpha'],['mu','5'],['d','nu']], [['22'],['4','e'],['9','alpha'],['mu','nu'],['d','5']], [['23'],['4','e'],['9','nu'],['mu','5'],['d','alpha']], [['24'],['4','e'],['9','nu'],['mu','alpha'],['d','5']]  ]

D = [  [['1'],['5','1'],['alpha','6'],['nu','beta'],['e','a']], [['2'],['5','1'],['alpha','6'],['nu','a'],['e','beta']], [['3'],['5','1'],['alpha','beta'],['nu','6'],['e','a']], [['4'],['5','1'],['alpha','beta'],['nu','a'],['e','6']], [['5'],['5','1'],['alpha','a'],['nu','6'],['e','beta']], [['6'],['5','1'],['alpha','a'],['nu','beta'],['e','6']]   ]
D+= [  [['7'],['5','6'],['alpha','1'],['nu','beta'],['e','a']], [['8'],['5','6'],['alpha','1'],['nu','a'],['e','beta']], [['9'],['5','6'],['alpha','beta'],['nu','1'],['e','a']], [['10'],['5','6'],['alpha','beta'],['nu','a'],['e','1']], [['11'],['5','6'],['alpha','a'],['nu','1'],['e','beta']], [['12'],['5','6'],['alpha','a'],['nu','beta'],['e','1']] ]
D+= [  [['13'],['5','beta'],['alpha','1'],['nu','6'],['e','a']], [['14'],['5','beta'],['alpha','1'],['nu','a'],['e','6']], [['15'],['5','beta'],['alpha','6'],['nu','1'],['e','a']], [['16'],['5','beta'],['alpha','6'],['nu','a'],['e','1']], [['17'],['5','beta'],['alpha','a'],['nu','1'],['e','6']], [['18'],['5','beta'],['alpha','a'],['nu','6'],['e','1']] ]
D+= [  [['19'],['5','a'],['alpha','1'],['nu','6'],['e','beta']], [['20'],['5','a'],['alpha','1'],['nu','beta'],['e','6']], [['21'],['5','a'],['alpha','6'],['nu','1'],['e','beta']], [['22'],['5','a'],['alpha','6'],['nu','beta'],['e','1']], [['23'],['5','a'],['alpha','beta'],['nu','1'],['e','6']], [['24'],['5','a'],['alpha','beta'],['nu','6beta'],['e','1']]  ]




