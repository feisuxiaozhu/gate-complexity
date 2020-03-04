import sympy as sym

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

# It takes input as U=u*u*u and D=delta*delta*delta
# Outputs contracted u as trace.
# For this case we do not need to consider the case receiving delta_ii, or [].
def u_contractor(U,D):
    res_dict=['(Tr(u))^3','Tr(u^2)Tr(u)','Tr(u^3)']
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
        return res_dict[2]
    elif counter == 1:
        return res_dict[1]
    else:
        return res_dict[0]

# T1= [['6'],['2','8'],['7','gamma'],['3','eta']] 
# T2= [['6'],['3','9'],['8','eta'],['4','mu']] 
# print(contractor(T1,T2))


# It takes input as U=u*u*u^dagger and D=delta*delta*delta
# Outputs contraced multiple of u as trace.
# Need to consider the case that D inlcudes [], or delta_ii
def u_dagger_contractor(U,D):
    #first count the number of delta_ii's ([]) in D.
    counter2=0
    for i in D:
        if i== []:
            counter2 += 1
    if counter2 == 0:
        prefactor = 1
    else:
        prefactor = 3*counter2

    res_dict=['(Tr(u))^2Tr(u^*)','Tr(u^2)Tr(u^*)','Tr(uu^*)Tr(u)','Tr(u^2u^*)']
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
        return res_dict[3], prefactor
    elif counter == 1:
        if res[2]==[]:
            return res_dict[1], prefactor
        else:
            return res_dict[2], prefactor
    else:
        return res_dict[0], prefactor

# U=[['1','2'],['6','7'],['beta','gamma']]
# D=[['2', '1'], ['7', '6'], ['gamma', 'beta']]
# E=[['2','6'],['7','1'],['gamma','beta']]
# F=[['2','beta'],['7','6'],['gamma','1']]
# print(u_contractor(U,D))

A=[  [['1'],['2','3'],['7','8'],['gamma','eta']], [['2'],['2','8'],['3','7'],['gamma','eta']], [['3'],['2','eta'],['3','gamma'],['7','8']], [['4'],['2','3'],['7','eta'],['8','gamma']], [['5'],['2','eta'],['3','7'],['8','gamma']], [['6'],['2','8'],['7','eta'],['3','gamma']]  ]
B=[  [['1'],['3','4'],['8','9'],['eta','mu']], [['2'],['3','9'],['4','8'],['eta','mu']], [['3'],['3','mu'],['4','eta'],['8','9']], [['4'],['3','4'],['8','mu'],['9','eta']], [['5'],['3','mu'],['4','8'],['9','eta']], [['6'],['3','9'],['8','mu'],['4','eta']]   ]
C=[  [['1'],['4','5'],['9','alpha'],['mu','nu']], [['2'],['4','alpha'],['5','9'],['mu','nu']], [['3'],['4','nu'],['5','mu'],['9','alpha']], [['4'],['4','5'],['9','nu'],['alpha','mu']], [['5'],['4','nu'],['5','9'],['alpha','mu']], [['6'],['4','alpha'],['9','nu'],['5','mu']]  ] 
D=[  [['1'],['5','1'],['alpha','6'],['nu','beta']], [['2'],['5','6'],['1','alpha'],['nu','beta']], [['3'],['5','beta'],['1','nu'],['alpha','6']], [['4'],['5','1'],['alpha','beta'],['6','nu']], [['5'],['5','beta'],['1','alpha'],['6','nu']], [['6'],['5','6'],['alpha','beta'],['1','nu']]    ]
temp1 = Multiply(A,B)
temp2 = Multiply(C,D)
temp3 = Multiply(temp1,temp2)

U=[['1','2'],['6','7'],['beta','gamma']]
A_res={}
for item in temp3:
    D=[]
    for i in range(1,len(item)):
        D.append(item[i])
    u = u_contractor(U,D)
    C = item[0]
    if u in A_res.keys():
        A_res[u].append(C)
    else:
        A_res[u]=[C]
    
V_3 = sym.Symbol('V_3')
V_21 = sym.Symbol('V_21')
V_111 = sym.Symbol('V_111')
c_dict = {'1': V_3+2*V_21+V_111, '2':V_3-V_111, '3':V_3-V_111, '4':V_3-V_111, '5':V_3-V_21+V_111, '6':V_3-V_21+V_111}

A_res_final = {}
for i,j in A_res.items():
    temp1=0
    for k in j:
        temp = 1
        for l in k:
            temp *= c_dict[l]
        temp1 += temp
    A_res_final[i] = sym.expand(temp1)


E=[  [['1'],['2','3'],['7','8'],['eta','gamma']], [['2'],['2','8'],['3','7'],['eta','gamma']], [['3'],['2','gamma'],['3','eta'],['7','8']], [['4'],['2','3'],['7','gamma'],['8','eta']], [['5'],['2','gamma'],['3','7'],['8','eta']], [['6'],['2','8'],['7','gamma'],['3','eta']]  ]
F=[  [['1'],['3','4'],['8','9'],['mu','eta']], [['2'],['3','9'],['4','8'],['mu','eta']], [['3'],['3','eta'],['4','mu'],['8','9']], [['4'],['3','4'],['8','eta'],['9','mu']], [['5'],['3','eta'],['4','8'],['9','mu']], [['6'],['3','9'],['8','eta'],['4','mu']]   ]
G=[  [['1'],['4','5'],['9','alpha'],['nu','mu']], [['2'],['4','alpha'],['5','9'],['nu','mu']], [['3'],['4','mu'],['5','nu'],['9','alpha']], [['4'],['4','5'],['9','mu'],['alpha','nu']], [['5'],['4','mu'],['5','9'],['alpha','nu']], [['6'],['4','alpha'],['9','mu'],['5','nu']]  ] 
H=[  [['1'],['5','1'],['alpha','6'],['beta','nu']], [['2'],['5','6'],['1','alpha'],['beta','nu']], [['3'],['5','nu'],['1','beta'],['alpha','6']], [['4'],['5','1'],['alpha','nu'],['6','beta']], [['5'],['5','nu'],['1','alpha'],['6','beta']], [['6'],['5','6'],['alpha','nu'],['1','beta']]    ]
temp4 = Multiply(E,F)
temp5 = Multiply(G,H)
temp6 = Multiply(temp4,temp5)
V = [['1','2'],['6','7'],['gamma','beta']]
B_res={}
for item in temp6:
    D=[]
    for i in range(1,len(item)):
        D.append(item[i])

    u, prefactor = u_dagger_contractor(V,D)
    C = item[0]
    C.append(prefactor) #append the frefactor to the end of multple of C's
    if u in B_res.keys():
        B_res[u].append(C)
    else:
        B_res[u]=[C]

V_1 = sym.Symbol('V_1')
V_2_1 = sym.Symbol('V_2-1')
V_11_1 = sym.Symbol('V_11-1')
N = sym.Symbol('N')
c_dagger_dict={'1':V_2_1+V_11_1,'2':V_2_1-V_11_1,'3':N*V_1-V_2_1-V_11_1,'4':N*V_1-V_2_1-V_11_1,'5':-V_1-V_2_1+V_11_1,'6':-V_1-V_2_1+V_11_1}

B_res_final = {}
for i,j in B_res.items():
    temp1=0
    for k in j:
        temp = 1
        for l in k:
            if type(l) is str:
                temp *= c_dagger_dict[l]
        temp1 += temp*k[-1]
    B_res_final[i] = sym.expand(temp1)

print(A_res_final)
print(B_res_final)
# But notice that Tr(uu^\dagger)*Tr(u) = 3Tr(u) and Tr(uuu^\dagger)= Tr(u), so B_res_final can be simplified.

print(3*B_res_final['Tr(uu^*)Tr(u)']+B_res_final['Tr(u^2u^*)'])



