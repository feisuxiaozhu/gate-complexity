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

# Input of this function looks like (C_1*delta*delta, C_2*delta*delta)
# It outputs C_1*C_2*delta*delta with all delta's contracted
def contractor(A,B):
    res = []
    res.append(A[0]+B[0])
    len_A = len(A)
    len_B = len(B)
    checked_B_index=[]
    for i in range(1,len_A):
        temp = False
        for item in A[i]:
            for j in range(1,len_B):
                if item in B[j]:
                    checked_B_index.append(j)
                    new_item = A[i] + B[j]
                    temp = True
        if temp:
            res.append(tiny_contractor(new_item))
        else:
            res.append(A[i])

    for i in range(1,len_B):
        if i not in checked_B_index:
            res.append(B[i])

    return res

# It takes input as E= C*delta*delta+ ... and F= C*delta*delta+... 
# Outputs contracted multiple between E and F
def Multiply(E,F):
    res = []
    for item_E in E:
        for item_F in F:
            res.append(contractor(item_E,item_F))
    return res



A=[  [['1'],['2','3'],['7','8'],['gamma','eta']], [['2'],['2','8'],['3','7'],['gamma','eta']], [['3'],['2','eta'],['3','gamma'],['7','8']], [['4'],['2','3'],['7','eta'],['8','gamma']], [['5'],['2','eta'],['3','7'],['8','gamma']], [['6'],['2','8'],['7','eta'],['3','gamma']]  ]
B=[  [['1'],['3','4'],['8','9'],['eta','mu']], [['2'],['3','9'],['4','8'],['eta','mu']], [['3'],['3','mu'],['4','eta'],['8','9']], [['4'],['3','4'],['8','mu'],['9','eta']], [['5'],['3','mu'],['4','8'],['9','eta']], [['6'],['3','9'],['8','mu'],['4','eta']]   ]
C=[  [['1'],['4','5'],['9','alpha'],['mu','nu']], [['2'],['4','alpha'],['5','9'],['mu','nu']], [['3'],['4','nu'],['5','mu'],['9','alpha']], [['4'],['4','5'],['9','nu'],['alpha','mu']], [['5'],['4','nu'],['5','9'],['alpha','mu']], [['6'],['4','alpha'],['9','nu'],['5','mu']]  ] 
D=[  [['1'],['5','1'],['alpha','6'],['nu','beta']], [['2'],['5','6'],['1','alpha'],['nu','beta']], [['3'],['5','beta'],['1','nu'],['alpha','6']], [['4'],['5','1'],['alpha','beta'],['6','nu']], [['5'],['5','beta'],['1','alpha'],['6','nu']], [['6'],['5','6'],['alpha','beta'],['1','nu']]    ]
temp1 = Multiply(A,B)
temp2 = Multiply(C,D)
temp3 = Multiply(temp1,temp2)
print(temp3)
# contractor(A,B)
# C = [1,'alpha','alpha',2]
# print(tiny_contractor(C))